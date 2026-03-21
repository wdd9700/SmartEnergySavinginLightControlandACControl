#!/usr/bin/env python3
"""
速度估计模块

基于单应矩阵的像素坐标到世界坐标转换，实现车辆速度估计。
支持多区域速度统计和速度平滑。

Example:
    >>> from traffic_energy.detection.speed_estimator import SpeedEstimator
    >>> estimator = SpeedEstimator(fps=30)
    >>> # 设置标定点
    >>> estimator.set_calibration_points(
    ...     pixel_points=[[100, 500], [500, 500], [100, 800], [500, 800]],
    ...     world_points=[[0, 0], [10, 0], [0, 10], [10, 10]]
    ... )
    >>> speed = estimator.estimate_speed(track_trajectory)
"""

from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import time

import numpy as np
import cv2

from shared.logger import setup_logger

logger = setup_logger("speed_estimator")


@dataclass
class SpeedMeasurement:
    """速度测量结果
    
    Attributes:
        speed_kmh: 速度 (km/h)
        speed_ms: 速度 (m/s)
        direction: 行驶方向 (度，0=北)
        confidence: 置信度
        timestamp: 测量时间戳
    """
    speed_kmh: float
    speed_ms: float
    direction: float
    confidence: float
    timestamp: float


class SpeedEstimator:
    """速度估计器
    
    使用单应矩阵将像素坐标转换为世界坐标，计算车辆速度。
    
    Attributes:
        fps: 视频帧率
        smoothing_window: 平滑窗口大小
        pixels_per_meter: 像素到米的转换比例（简化模式）
        homography_matrix: 单应矩阵
        
    Example:
        >>> estimator = SpeedEstimator(fps=30)
        >>> estimator.set_calibration_points(
        ...     pixel_points=[[960, 540], [1000, 540]],
        ...     world_points=[[0, 0], [10, 0]]
        ... )
        >>> measurement = estimator.estimate_speed_from_positions(
        ...     [(100, 500), (120, 500)], [0.0, 0.033]
        ... )
        >>> print(f"速度: {measurement.speed_kmh:.1f} km/h")
    """
    
    def __init__(
        self,
        fps: float = 30.0,
        smoothing_window: int = 5,
        pixels_per_meter: Optional[float] = None
    ) -> None:
        """初始化速度估计器
        
        Args:
            fps: 视频帧率
            smoothing_window: 速度平滑窗口大小
            pixels_per_meter: 像素到米的转换比例（可选）
        """
        self.fps = fps
        self.smoothing_window = smoothing_window
        self.pixels_per_meter = pixels_per_meter
        self.homography_matrix: Optional[np.ndarray] = None
        
        # 速度历史（用于平滑）
        self._speed_history: Dict[int, List[float]] = {}
        
        logger.info(f"初始化速度估计器，FPS: {fps}")
    
    def set_calibration_points(
        self,
        pixel_points: List[List[float]],
        world_points: List[List[float]]
    ) -> bool:
        """设置标定点并计算单应矩阵
        
        Args:
            pixel_points: 像素坐标点列表 [[x1, y1], [x2, y2], ...]
            world_points: 世界坐标点列表 [[X1, Y1], [X2, Y2], ...]
            
        Returns:
            是否成功
        """
        if len(pixel_points) < 4 or len(world_points) < 4:
            logger.error("至少需要4个标定点")
            return False
        
        if len(pixel_points) != len(world_points):
            logger.error("像素点和世界点数量不匹配")
            return False
        
        try:
            pixel_array = np.array(pixel_points, dtype=np.float32)
            world_array = np.array(world_points, dtype=np.float32)
            
            # 计算单应矩阵
            self.homography_matrix, _ = cv2.findHomography(
                pixel_array[:4], world_array[:4]
            )
            
            if self.homography_matrix is not None:
                logger.info("单应矩阵计算成功")
                return True
            else:
                logger.error("单应矩阵计算失败")
                return False
                
        except Exception as e:
            logger.error(f"标定失败: {e}")
            return False
    
    def set_homography_matrix(self, matrix: np.ndarray) -> None:
        """直接设置单应矩阵
        
        Args:
            matrix: 3x3单应矩阵
        """
        if matrix.shape != (3, 3):
            raise ValueError("单应矩阵必须是3x3")
        
        self.homography_matrix = matrix.copy()
        logger.info("单应矩阵已设置")
    
    def pixel_to_world(
        self,
        pixel_point: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """将像素坐标转换为世界坐标
        
        Args:
            pixel_point: 像素坐标 (x, y)
            
        Returns:
            世界坐标 (X, Y) 或 None
        """
        if self.homography_matrix is not None:
            # 使用单应矩阵转换
            pixel = np.array([[pixel_point[0], pixel_point[1], 1.0]])
            world = np.dot(self.homography_matrix, pixel.T)
            world = world / world[2]  # 齐次坐标归一化
            return (float(world[0]), float(world[1]))
        
        elif self.pixels_per_meter is not None:
            # 使用简单比例转换
            return (
                pixel_point[0] / self.pixels_per_meter,
                pixel_point[1] / self.pixels_per_meter
            )
        
        else:
            logger.warning("未设置标定参数")
            return None
    
    def estimate_speed_from_positions(
        self,
        positions: List[Tuple[float, float]],
        timestamps: List[float],
        track_id: Optional[int] = None
    ) -> Optional[SpeedMeasurement]:
        """从位置序列估计速度
        
        Args:
            positions: 像素位置列表 [(x1, y1), (x2, y2), ...]
            timestamps: 时间戳列表
            track_id: 轨迹ID（用于平滑）
            
        Returns:
            SpeedMeasurement或None
        """
        if len(positions) < 2 or len(timestamps) < 2:
            return None
        
        if len(positions) != len(timestamps):
            logger.error("位置和时间戳数量不匹配")
            return None
        
        # 转换为世界坐标
        world_positions = []
        for pos in positions:
            world_pos = self.pixel_to_world(pos)
            if world_pos is not None:
                world_positions.append(world_pos)
        
        if len(world_positions) < 2:
            return None
        
        # 计算速度
        speeds = []
        directions = []
        
        for i in range(1, len(world_positions)):
            dx = world_positions[i][0] - world_positions[i-1][0]
            dy = world_positions[i][1] - world_positions[i-1][1]
            dt = timestamps[i] - timestamps[i-1]
            
            if dt <= 0:
                continue
            
            distance = np.sqrt(dx**2 + dy**2)
            speed_ms = distance / dt
            
            # 计算方向（0=北，顺时针）
            direction = np.degrees(np.arctan2(dx, -dy))  # 注意Y轴向下
            if direction < 0:
                direction += 360
            
            speeds.append(speed_ms)
            directions.append(direction)
        
        if not speeds:
            return None
        
        # 平均速度
        avg_speed_ms = np.mean(speeds)
        avg_direction = np.mean(directions)
        
        # 速度平滑
        if track_id is not None:
            avg_speed_ms = self._smooth_speed(track_id, avg_speed_ms)
        
        # 转换为km/h
        speed_kmh = avg_speed_ms * 3.6
        
        # 计算置信度（基于速度一致性）
        if len(speeds) > 1:
            speed_std = np.std(speeds)
            confidence = max(0.0, 1.0 - speed_std / (avg_speed_ms + 0.1))
        else:
            confidence = 0.5
        
        return SpeedMeasurement(
            speed_kmh=speed_kmh,
            speed_ms=avg_speed_ms,
            direction=avg_direction,
            confidence=confidence,
            timestamp=time.time()
        )
    
    def estimate_speed_from_trajectory(
        self,
        trajectory: List,
        track_id: Optional[int] = None
    ) -> Optional[SpeedMeasurement]:
        """从轨迹估计速度
        
        Args:
            trajectory: 轨迹点列表（需要有center和timestamp属性）
            track_id: 轨迹ID
            
        Returns:
            SpeedMeasurement或None
        """
        if len(trajectory) < 2:
            return None
        
        positions = []
        timestamps = []
        
        for point in trajectory:
            if hasattr(point, 'center') and hasattr(point, 'timestamp'):
                positions.append(point.center)
                timestamps.append(point.timestamp)
            elif isinstance(point, dict):
                positions.append(point.get('center', (0, 0)))
                timestamps.append(point.get('timestamp', 0))
        
        return self.estimate_speed_from_positions(
            positions, timestamps, track_id
        )
    
    def _smooth_speed(
        self,
        track_id: int,
        speed_ms: float
    ) -> float:
        """平滑速度值
        
        Args:
            track_id: 轨迹ID
            speed_ms: 当前速度
            
        Returns:
            平滑后的速度
        """
        if track_id not in self._speed_history:
            self._speed_history[track_id] = []
        
        history = self._speed_history[track_id]
        history.append(speed_ms)
        
        # 保持窗口大小
        if len(history) > self.smoothing_window:
            history.pop(0)
        
        # 使用中值滤波减少异常值影响
        if len(history) >= 3:
            return float(np.median(history))
        else:
            return float(np.mean(history))
    
    def get_average_speed(
        self,
        measurements: List[SpeedMeasurement]
    ) -> Tuple[float, float]:
        """计算平均速度
        
        Args:
            measurements: 速度测量列表
            
        Returns:
            (平均速度km/h, 标准差)
        """
        if not measurements:
            return 0.0, 0.0
        
        speeds = [m.speed_kmh for m in measurements]
        return float(np.mean(speeds)), float(np.std(speeds))
    
    def clear_history(self, track_id: Optional[int] = None) -> None:
        """清除速度历史
        
        Args:
            track_id: 特定轨迹ID，None则清除所有
        """
        if track_id is not None:
            if track_id in self._speed_history:
                del self._speed_history[track_id]
        else:
            self._speed_history.clear()
    
    def is_calibrated(self) -> bool:
        """检查是否已完成标定
        
        Returns:
            是否已标定
        """
        return self.homography_matrix is not None or self.pixels_per_meter is not None

#!/usr/bin/env python3
"""
摄像头处理器模块

整合检测、跟踪和速度估计的单摄像头处理流程。
支持RTSP/视频文件/摄像头输入和异步处理。

Example:
    >>> from traffic_energy.detection import CameraProcessor
    >>> processor = CameraProcessor('rtsp://camera/stream', 'cam_001')
    >>> processor.start()
    >>> while True:
    ...     result = processor.process_frame()
    ...     if result:
    ...         print(f"检测到 {len(result.tracks)} 辆车")
"""

import time
import threading
import queue
from typing import Optional, List, Callable, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import cv2

from shared.video_capture import VideoCapture
from shared.logger import setup_logger
from shared.performance import PerformanceMonitor

from .vehicle_detector import VehicleDetector, Detection
from .vehicle_tracker import VehicleTracker, Track, TrackerConfig
from .speed_estimator import SpeedEstimator, SpeedMeasurement

logger = setup_logger("camera_processor")


@dataclass
class ProcessingResult:
    """处理结果数据类
    
    Attributes:
        frame: 处理后的帧
        timestamp: 时间戳
        camera_id: 摄像头ID
        tracks: 跟踪结果列表
        detections: 检测结果列表
        speed_measurements: 速度测量结果
        fps: 处理帧率
        latency_ms: 处理延迟（毫秒）
    """
    frame: np.ndarray
    timestamp: float
    camera_id: str
    tracks: List[Track] = field(default_factory=list)
    detections: List[Detection] = field(default_factory=list)
    speed_measurements: Dict[int, SpeedMeasurement] = field(default_factory=dict)
    fps: float = 0.0
    latency_ms: float = 0.0


class CameraProcessor:
    """摄像头处理器
    
    整合检测、跟踪和速度估计的完整处理流程。
    支持多种输入源和异步处理。
    
    Attributes:
        source: 视频源
        camera_id: 摄像头ID
        detector: 车辆检测器
        tracker: 车辆跟踪器
        speed_estimator: 速度估计器
        
    Example:
        >>> processor = CameraProcessor(
        ...     source='video.mp4',
        ...     camera_id='cam_001',
        ...     model_path='yolo12n.pt'
        ... )
        >>> processor.start()
        >>> for result in processor:
        ...     cv2.imshow('result', result.frame)
        >>> processor.stop()
    """
    
    def __init__(
        self,
        source: Union[str, int],
        camera_id: str,
        model_path: str = "yolo12n.pt",
        conf_threshold: float = 0.5,
        tracker_config: Optional[TrackerConfig] = None,
        enable_speed: bool = False,
        async_mode: bool = False,
        queue_size: int = 10
    ) -> None:
        """初始化摄像头处理器
        
        Args:
            source: 视频源（RTSP URL、文件路径或摄像头索引）
            camera_id: 摄像头唯一标识
            model_path: YOLO模型路径
            conf_threshold: 检测置信度阈值
            tracker_config: 跟踪器配置
            enable_speed: 是否启用速度估计
            async_mode: 是否启用异步处理
            queue_size: 异步队列大小
        """
        self.source = source
        self.camera_id = camera_id
        self.async_mode = async_mode
        
        # 初始化视频捕获
        self._capture: Optional[VideoCapture] = None
        
        # 初始化检测器
        logger.info(f"[{camera_id}] 初始化检测器...")
        self.detector = VehicleDetector(
            model_path=model_path,
            conf_threshold=conf_threshold
        )
        
        # 初始化跟踪器
        logger.info(f"[{camera_id}] 初始化跟踪器...")
        self.tracker = VehicleTracker(tracker_config or TrackerConfig())
        
        # 初始化速度估计器
        self.speed_estimator: Optional[SpeedEstimator] = None
        if enable_speed:
            self.speed_estimator = SpeedEstimator()
        
        # 异步处理队列
        self._frame_queue: Optional[queue.Queue] = None
        self._result_queue: Optional[queue.Queue] = None
        self._processing_thread: Optional[threading.Thread] = None
        self._is_running = False
        
        if async_mode:
            self._frame_queue = queue.Queue(maxsize=queue_size)
            self._result_queue = queue.Queue(maxsize=queue_size)
        
        # 性能监控
        self._perf_monitor = PerformanceMonitor()
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._current_fps = 0.0
        
        # 回调函数
        self._callbacks: List[Callable[[ProcessingResult], None]] = []
        
        logger.info(f"[{camera_id}] 处理器初始化完成")
    
    def start(self) -> bool:
        """启动处理器
        
        Returns:
            是否成功启动
        """
        try:
            # 启动视频捕获
            self._capture = VideoCapture(self.source)
            if not self._capture.start():
                logger.error(f"[{self.camera_id}] 无法打开视频源: {self.source}")
                return False
            
            self._is_running = True
            
            # 启动异步处理线程
            if self.async_mode:
                self._processing_thread = threading.Thread(
                    target=self._async_processing_loop,
                    daemon=True
                )
                self._processing_thread.start()
            
            logger.info(f"[{self.camera_id}] 处理器已启动")
            return True
            
        except Exception as e:
            logger.error(f"[{self.camera_id}] 启动失败: {e}")
            return False
    
    def stop(self) -> None:
        """停止处理器"""
        self._is_running = False
        
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)
        
        if self._capture:
            self._capture.stop()
        
        logger.info(f"[{self.camera_id}] 处理器已停止")
    
    def process_frame(self) -> Optional[ProcessingResult]:
        """处理单帧
        
        Returns:
            ProcessingResult或None
        """
        if not self._capture:
            return None
        
        start_time = time.time()
        
        # 读取帧
        frame = self._capture.read()
        if frame is None:
            return None
        
        # 执行检测
        detections = self.detector.detect(frame)
        
        # 执行跟踪
        tracks = self.tracker.update(detections, frame)
        
        # 估计速度
        speed_measurements: Dict[int, SpeedMeasurement] = {}
        if self.speed_estimator and self.speed_estimator.is_calibrated():
            for track in tracks:
                if len(track.trajectory) >= 2:
                    measurement = self.speed_estimator.estimate_speed_from_trajectory(
                        track.trajectory,
                        track.track_id
                    )
                    if measurement:
                        speed_measurements[track.track_id] = measurement
        
        # 绘制结果
        result_frame = self._visualize(frame, tracks, speed_measurements)
        
        # 计算FPS
        self._frame_count += 1
        current_time = time.time()
        if current_time - self._last_fps_time >= 1.0:
            self._current_fps = self._frame_count / (current_time - self._last_fps_time)
            self._frame_count = 0
            self._last_fps_time = current_time
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = ProcessingResult(
            frame=result_frame,
            timestamp=time.time(),
            camera_id=self.camera_id,
            tracks=tracks,
            detections=detections,
            speed_measurements=speed_measurements,
            fps=self._current_fps,
            latency_ms=latency_ms
        )
        
        # 触发回调
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"回调执行失败: {e}")
        
        return result
    
    def _async_processing_loop(self) -> None:
        """异步处理循环"""
        while self._is_running:
            try:
                # 获取帧
                frame = self._frame_queue.get(timeout=0.1)
                
                # 处理
                detections = self.detector.detect(frame)
                tracks = self.tracker.update(detections, frame)
                
                # 放入结果队列
                self._result_queue.put((detections, tracks))
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"异步处理错误: {e}")
    
    def _visualize(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        speed_measurements: Dict[int, SpeedMeasurement]
    ) -> np.ndarray:
        """可视化结果
        
        Args:
            frame: 原始帧
            tracks: 跟踪结果
            speed_measurements: 速度测量
            
        Returns:
            绘制后的帧
        """
        result = frame.copy()
        
        # 绘制跟踪框
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            
            # 根据状态选择颜色
            if track.is_confirmed:
                color = (0, 255, 0)  # 绿色
            else:
                color = (0, 165, 255)  # 橙色
            
            # 绘制边界框
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # 准备标签
            label_parts = [f"ID:{track.track_id}", track.class_name]
            
            # 添加速度信息
            if track.track_id in speed_measurements:
                speed = speed_measurements[track.track_id]
                label_parts.append(f"{speed.speed_kmh:.1f}km/h")
            
            label = " | ".join(label_parts)
            
            # 绘制标签背景
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                result,
                (x1, y1 - text_h - 10),
                (x1 + text_w, y1),
                color,
                -1
            )
            
            # 绘制标签文字
            cv2.putText(
                result,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
            
            # 绘制轨迹
            if len(track.trajectory) > 1:
                points = [
                    (int(p.center[0]), int(p.center[1]))
                    for p in track.trajectory[-30:]  # 最近30个点
                ]
                for i in range(1, len(points)):
                    cv2.line(result, points[i-1], points[i], color, 1)
        
        # 绘制统计信息
        stats_text = [
            f"Camera: {self.camera_id}",
            f"Tracks: {len(tracks)}",
            f"FPS: {self._current_fps:.1f}",
            f"Detector FPS: {self.detector.fps:.1f}"
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(
                result,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            y_offset += 25
        
        return result
    
    def add_callback(self, callback: Callable[[ProcessingResult], None]) -> None:
        """添加结果回调函数
        
        Args:
            callback: 回调函数，接收ProcessingResult参数
        """
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[ProcessingResult], None]) -> None:
        """移除回调函数
        
        Args:
            callback: 要移除的回调函数
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def set_calibration_points(
        self,
        pixel_points: List[List[float]],
        world_points: List[List[float]]
    ) -> bool:
        """设置速度估计标定点
        
        Args:
            pixel_points: 像素坐标点
            world_points: 世界坐标点
            
        Returns:
            是否成功
        """
        if self.speed_estimator is None:
            logger.warning("速度估计器未启用")
            return False
        
        return self.speed_estimator.set_calibration_points(
            pixel_points, world_points
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'camera_id': self.camera_id,
            'fps': self._current_fps,
            'detector_fps': self.detector.fps,
            'total_tracks': len(self.tracker.tracks),
            'active_tracks': len(self.tracker.get_active_tracks()),
            'confirmed_tracks': len(self.tracker.get_confirmed_tracks())
        }
    
    def __iter__(self):
        """迭代器接口"""
        return self
    
    def __next__(self) -> ProcessingResult:
        """获取下一帧结果"""
        result = self.process_frame()
        if result is None:
            raise StopIteration
        return result
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()
        return False

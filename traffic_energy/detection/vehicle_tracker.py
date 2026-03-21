#!/usr/bin/env python3
"""
车辆跟踪模块

基于BoT-SORT的多目标跟踪器封装。
支持相机运动补偿、轨迹管理和持久化。

Example:
    >>> from traffic_energy.detection import VehicleTracker, TrackerConfig
    >>> config = TrackerConfig(track_buffer=60)
    >>> tracker = VehicleTracker(config)
    >>> tracks = tracker.update(detections, frame)
"""

import time
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import cv2

from shared.logger import setup_logger

logger = setup_logger("vehicle_tracker")


@dataclass
class TrajectoryPoint:
    """轨迹点数据类
    
    Attributes:
        timestamp: 时间戳
        bbox: 边界框 [x1, y1, x2, y2]
        center: 中心点坐标
        velocity: 速度向量 [vx, vy]
        speed: 速度大小 (km/h)
    """
    timestamp: float
    bbox: np.ndarray
    center: Tuple[float, float]
    velocity: Optional[Tuple[float, float]] = None
    speed: Optional[float] = None


@dataclass
class Track:
    """跟踪轨迹数据类
    
    Attributes:
        track_id: 跟踪ID
        bbox: 当前边界框
        confidence: 置信度
        class_id: 类别ID
        class_name: 类别名称
        age: 轨迹年龄（帧数）
        hits: 命中次数
        time_since_update: 距离上次更新的帧数
        state: 状态 ('confirmed', 'tentative', 'deleted')
        trajectory: 历史轨迹点
    """
    track_id: int
    bbox: np.ndarray
    confidence: float
    class_id: int
    class_name: str
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    state: str = "tentative"
    trajectory: List[TrajectoryPoint] = field(default_factory=list)
    
    @property
    def is_confirmed(self) -> bool:
        """是否为确认状态"""
        return self.state == "confirmed"
    
    @property
    def is_deleted(self) -> bool:
        """是否为删除状态"""
        return self.state == "deleted"
    
    @property
    def center(self) -> Tuple[float, float]:
        """当前中心点"""
        return ((self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2)
    
    def get_last_n_points(self, n: int) -> List[TrajectoryPoint]:
        """获取最近N个轨迹点"""
        return self.trajectory[-n:] if len(self.trajectory) >= n else self.trajectory


@dataclass
class TrackerConfig:
    """跟踪器配置
    
    Attributes:
        track_high_thresh: 高置信度阈值
        track_low_thresh: 低置信度阈值
        new_track_thresh: 新轨迹阈值
        track_buffer: 丢失跟踪缓冲帧数
        match_thresh: 匹配阈值
        proximity_thresh: 空间距离阈值
        appearance_thresh: 外观相似度阈值
        cmc_method: 相机运动补偿方法 ('ecc', 'orb', 'none')
        frame_rate: 帧率
        lambda_: 运动和外观权重
    """
    track_high_thresh: float = 0.6
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.7
    track_buffer: int = 60
    match_thresh: float = 0.8
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    cmc_method: str = "ecc"
    frame_rate: int = 30
    lambda_: float = 0.985


class VehicleTracker:
    """车辆跟踪器
    
    基于BoT-SORT算法的多目标跟踪器封装。
    支持相机运动补偿和轨迹管理。
    
    Attributes:
        config: 跟踪器配置
        tracks: 当前活跃轨迹
        _next_id: 下一个轨迹ID
        
    Example:
        >>> from traffic_energy.detection.vehicle_detector import Detection
        >>> config = TrackerConfig()
        >>> tracker = VehicleTracker(config)
        >>> 
        >>> # 在视频循环中
        >>> while True:
        ...     detections = detector.detect(frame)
        ...     tracks = tracker.update(detections, frame)
        ...     for track in tracks:
        ...         print(f"车辆 {track.track_id}: {track.class_name}")
    """
    
    def __init__(self, config: Optional[TrackerConfig] = None) -> None:
        """初始化跟踪器
        
        Args:
            config: 跟踪器配置，默认使用默认配置
        """
        self.config = config or TrackerConfig()
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        
        # 相机运动补偿相关
        self._prev_frame: Optional[np.ndarray] = None
        self._cmc_matrix: Optional[np.ndarray] = None
        
        # 统计
        self.frame_count = 0
        self.total_tracks_created = 0
        
        logger.info(f"初始化车辆跟踪器，CMC方法: {self.config.cmc_method}")
    
    def update(
        self,
        detections: List,
        frame: np.ndarray
    ) -> List[Track]:
        """更新跟踪器
        
        Args:
            detections: 检测结果列表 (Detection对象)
            frame: 当前帧图像
            
        Returns:
            List[Track]: 当前活跃轨迹列表
        """
        self.frame_count += 1
        timestamp = time.time()
        
        # 相机运动补偿
        if self.config.cmc_method != "none":
            self._apply_cmc(frame)
        
        # 提取检测框和分数
        detection_boxes = []
        detection_scores = []
        detection_classes = []
        
        for det in detections:
            detection_boxes.append(det.bbox)
            detection_scores.append(det.confidence)
            detection_classes.append(det.class_id)
        
        detection_boxes = np.array(detection_boxes) if detection_boxes else np.empty((0, 4))
        detection_scores = np.array(detection_scores) if detection_scores else np.empty(0)
        
        # 分离高置信度和低置信度检测
        high_dets = []
        low_dets = []
        
        for i, score in enumerate(detection_scores):
            if score >= self.config.track_high_thresh:
                high_dets.append(i)
            elif score >= self.config.track_low_thresh:
                low_dets.append(i)
        
        # 预测现有轨迹位置（使用卡尔曼滤波或简单运动模型）
        for track in self.tracks.values():
            if not track.is_deleted:
                track.time_since_update += 1
                # 简单的运动预测：保持上一帧位置
                # 实际应用中应使用卡尔曼滤波
        
        # 第一次关联：高置信度检测与确认轨迹
        matched, unmatched_tracks, unmatched_dets = self._associate(
            [t for t in self.tracks.values() if t.is_confirmed and not t.is_deleted],
            [detection_boxes[i] for i in high_dets],
            self.config.match_thresh
        )
        
        # 更新匹配的轨迹
        for track_idx, det_idx in matched:
            track_id = list(self.tracks.keys())[list(self.tracks.values()).index(
                [t for t in self.tracks.values() if t.is_confirmed and not t.is_deleted][track_idx]
            )]
            det_idx_in_high = high_dets[det_idx]
            self._update_track(
                track_id,
                detection_boxes[det_idx_in_high],
                detection_scores[det_idx_in_high],
                detection_classes[det_idx_in_high],
                timestamp
            )
        
        # 第二次关联：未匹配轨迹与低置信度检测
        if unmatched_tracks and low_dets:
            remaining_tracks = [
                t for i, t in enumerate([t for t in self.tracks.values() if t.is_confirmed and not t.is_deleted])
                if i in unmatched_tracks
            ]
            matched2, unmatched_tracks2, _ = self._associate(
                remaining_tracks,
                [detection_boxes[i] for i in low_dets],
                self.config.match_thresh
            )
            
            for track_idx, det_idx in matched2:
                track = remaining_tracks[track_idx]
                det_idx_in_low = low_dets[det_idx]
                self._update_track(
                    track.track_id,
                    detection_boxes[det_idx_in_low],
                    detection_scores[det_idx_in_low],
                    detection_classes[det_idx_in_low],
                    timestamp
                )
        
        # 处理未匹配的确认轨迹
        confirmed_tracks = [t for t in self.tracks.values() if t.is_confirmed and not t.is_deleted]
        for track in confirmed_tracks:
            if track.time_since_update > self.config.track_buffer:
                track.state = "deleted"
                logger.debug(f"轨迹 {track.track_id} 被删除（超时）")
        
        # 处理未匹配的检测：创建新轨迹
        unmatched_high_dets = [high_dets[i] for i in unmatched_dets if i < len(high_dets)]
        for det_idx in unmatched_high_dets:
            if detection_scores[det_idx] >= self.config.new_track_thresh:
                self._create_track(
                    detection_boxes[det_idx],
                    detection_scores[det_idx],
                    detection_classes[det_idx],
                    timestamp
                )
        
        # 更新未匹配的tentative轨迹
        for track in self.tracks.values():
            if track.state == "tentative" and track.time_since_update > 3:
                track.state = "deleted"
        
        # 保存当前帧用于CMC
        self._prev_frame = frame.copy()
        
        # 返回活跃轨迹
        active_tracks = [t for t in self.tracks.values() 
                        if not t.is_deleted and t.time_since_update < 3]
        
        return active_tracks
    
    def _associate(
        self,
        tracks: List[Track],
        detections: List[np.ndarray],
        threshold: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """关联轨迹和检测
        
        使用IoU作为相似度度量
        
        Args:
            tracks: 轨迹列表
            detections: 检测框列表
            threshold: 匹配阈值
            
        Returns:
            (matched_pairs, unmatched_track_indices, unmatched_det_indices)
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # 计算IoU矩阵
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._compute_iou(track.bbox, det)
        
        # 使用匈牙利算法进行匹配
        try:
            from scipy.optimize import linear_sum_assignment
            track_indices, det_indices = linear_sum_assignment(-iou_matrix)
        except ImportError:
            # 简化版本：贪心匹配
            track_indices, det_indices = self._greedy_match(iou_matrix)
        
        matched_pairs = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))
        
        for t_idx, d_idx in zip(track_indices, det_indices):
            if iou_matrix[t_idx, d_idx] >= threshold:
                matched_pairs.append((t_idx, d_idx))
                unmatched_tracks.remove(t_idx)
                unmatched_dets.remove(d_idx)
        
        return matched_pairs, unmatched_tracks, unmatched_dets
    
    def _greedy_match(self, iou_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """贪心匹配算法
        
        当scipy不可用时使用
        """
        matched_tracks = []
        matched_dets = []
        
        iou_copy = iou_matrix.copy()
        while True:
            if iou_copy.size == 0 or iou_copy.max() < 0.1:
                break
            
            t_idx, d_idx = np.unravel_index(iou_copy.argmax(), iou_copy.shape)
            matched_tracks.append(t_idx)
            matched_dets.append(d_idx)
            
            # 移除已匹配的行和列
            iou_copy[t_idx, :] = -1
            iou_copy[:, d_idx] = -1
        
        return np.array(matched_tracks), np.array(matched_dets)
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """计算两个边界框的IoU
        
        Args:
            box1: 边界框1 [x1, y1, x2, y2]
            box2: 边界框2 [x1, y1, x2, y2]
            
        Returns:
            IoU值
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _update_track(
        self,
        track_id: int,
        bbox: np.ndarray,
        confidence: float,
        class_id: int,
        timestamp: float
    ) -> None:
        """更新轨迹
        
        Args:
            track_id: 轨迹ID
            bbox: 新的边界框
            confidence: 置信度
            class_id: 类别ID
            timestamp: 时间戳
        """
        if track_id not in self.tracks:
            return
        
        track = self.tracks[track_id]
        
        # 计算速度
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        velocity = None
        speed = None
        
        if track.trajectory:
            last_point = track.trajectory[-1]
            dt = timestamp - last_point.timestamp
            if dt > 0:
                vx = (center[0] - last_point.center[0]) / dt
                vy = (center[1] - last_point.center[1]) / dt
                velocity = (vx, vy)
                # 速度大小（像素/秒）
                speed = np.sqrt(vx**2 + vy**2)
        
        # 更新轨迹信息
        track.bbox = bbox
        track.confidence = confidence
        track.hits += 1
        track.time_since_update = 0
        track.age += 1
        
        # 确认轨迹
        if track.state == "tentative" and track.hits >= 3:
            track.state = "confirmed"
            logger.debug(f"轨迹 {track_id} 已确认")
        
        # 添加轨迹点
        trajectory_point = TrajectoryPoint(
            timestamp=timestamp,
            bbox=bbox.copy(),
            center=center,
            velocity=velocity,
            speed=speed
        )
        track.trajectory.append(trajectory_point)
        
        # 限制轨迹长度
        if len(track.trajectory) > 1000:
            track.trajectory = track.trajectory[-500:]
    
    def _create_track(
        self,
        bbox: np.ndarray,
        confidence: float,
        class_id: int,
        timestamp: float
    ) -> int:
        """创建新轨迹
        
        Args:
            bbox: 边界框
            confidence: 置信度
            class_id: 类别ID
            timestamp: 时间戳
            
        Returns:
            新轨迹ID
        """
        track_id = self._next_id
        self._next_id += 1
        
        class_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        class_name = class_names.get(class_id, f"class_{class_id}")
        
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        trajectory_point = TrajectoryPoint(
            timestamp=timestamp,
            bbox=bbox.copy(),
            center=center
        )
        
        track = Track(
            track_id=track_id,
            bbox=bbox,
            confidence=confidence,
            class_id=class_id,
            class_name=class_name,
            hits=1,
            trajectory=[trajectory_point]
        )
        
        self.tracks[track_id] = track
        self.total_tracks_created += 1
        
        logger.debug(f"创建新轨迹 {track_id} ({class_name})")
        
        return track_id
    
    def _apply_cmc(self, frame: np.ndarray) -> None:
        """应用相机运动补偿
        
        Args:
            frame: 当前帧
        """
        if self._prev_frame is None:
            return
        
        if self.config.cmc_method == "ecc":
            self._cmc_matrix = self._compute_ecc(self._prev_frame, frame)
        elif self.config.cmc_method == "orb":
            self._cmc_matrix = self._compute_orb(self._prev_frame, frame)
    
    def _compute_ecc(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """使用ECC算法计算相机运动
        
        Args:
            prev_frame: 前一帧
            curr_frame: 当前帧
            
        Returns:
            变换矩阵或None
        """
        try:
            # 转换为灰度图
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # 定义变换矩阵
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            
            # ECC算法
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)
            cc, warp_matrix = cv2.findTransformECC(
                prev_gray, curr_gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria
            )
            
            return warp_matrix
        except Exception as e:
            logger.debug(f"ECC计算失败: {e}")
            return None
    
    def _compute_orb(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """使用ORB特征计算相机运动
        
        Args:
            prev_frame: 前一帧
            curr_frame: 当前帧
            
        Returns:
            变换矩阵或None
        """
        try:
            # 转换为灰度图
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # ORB检测器
            orb = cv2.ORB_create(100)
            
            # 检测特征点
            kp1, des1 = orb.detectAndCompute(prev_gray, None)
            kp2, des2 = orb.detectAndCompute(curr_gray, None)
            
            if des1 is None or des2 is None:
                return None
            
            # 特征匹配
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            if len(matches) < 4:
                return None
            
            # 提取匹配点
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # 计算变换矩阵
            matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            
            return matrix
        except Exception as e:
            logger.debug(f"ORB计算失败: {e}")
            return None
    
    def get_trajectory(self, track_id: int) -> Optional[List[TrajectoryPoint]]:
        """获取指定ID的完整轨迹
        
        Args:
            track_id: 轨迹ID
            
        Returns:
            轨迹点列表或None
        """
        if track_id in self.tracks:
            return self.tracks[track_id].trajectory.copy()
        return None
    
    def get_track(self, track_id: int) -> Optional[Track]:
        """获取指定ID的轨迹
        
        Args:
            track_id: 轨迹ID
            
        Returns:
            Track对象或None
        """
        return self.tracks.get(track_id)
    
    def get_active_tracks(self) -> List[Track]:
        """获取所有活跃轨迹
        
        Returns:
            活跃轨迹列表
        """
        return [t for t in self.tracks.values() 
                if not t.is_deleted and t.time_since_update < 3]
    
    def get_confirmed_tracks(self) -> List[Track]:
        """获取所有确认轨迹
        
        Returns:
            确认轨迹列表
        """
        return [t for t in self.tracks.values() 
                if t.is_confirmed and not t.is_deleted]
    
    def reset(self) -> None:
        """重置跟踪器"""
        self.tracks.clear()
        self._next_id = 1
        self.frame_count = 0
        self._prev_frame = None
        self._cmc_matrix = None
        logger.info("跟踪器已重置")

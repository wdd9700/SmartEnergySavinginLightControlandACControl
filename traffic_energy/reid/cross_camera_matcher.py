#!/usr/bin/env python3
"""
跨摄像头匹配模块

实现车辆跨摄像头重识别和轨迹关联。

Example:
    >>> from traffic_energy.reid import CrossCameraMatcher
    >>> matcher = CrossCameraMatcher(camera_topology)
    >>> match = matcher.match_vehicle(vehicle_features, camera_id, timestamp)
"""

from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass
import time

import numpy as np

from shared.logger import setup_logger
from .feature_database import FeatureDatabase

logger = setup_logger("cross_camera_matcher")


@dataclass
class Match:
    """匹配结果
    
    Attributes:
        vehicle_id: 匹配的车辆ID
        similarity: 相似度分数
        confidence: 置信度
        source_camera: 源摄像头
        target_camera: 目标摄像头
        travel_time: 通行时间（秒）
    """
    vehicle_id: str
    similarity: float
    confidence: float
    source_camera: str
    target_camera: str
    travel_time: float


@dataclass
class CameraNode:
    """摄像头节点"""
    camera_id: str
    location: Tuple[float, float]
    neighbors: List[str]
    min_travel_time: float
    max_travel_time: float


class CrossCameraMatcher:
    """跨摄像头匹配器
    
    结合外观特征和时空约束进行车辆匹配。
    
    Attributes:
        feature_db: 特征数据库
        camera_topology: 摄像头拓扑
        similarity_threshold: 相似度阈值
        
    Example:
        >>> matcher = CrossCameraMatcher(feature_db, topology)
        >>> match = matcher.match_vehicle(
        ...     features, 'cam_001', time.time()
        ... )
    """
    
    def __init__(
        self,
        feature_db: FeatureDatabase,
        camera_topology: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.7,
        temporal_constraint: float = 300.0,
        spatial_constraint: float = 1000.0
    ) -> None:
        """初始化匹配器
        
        Args:
            feature_db: 特征数据库
            camera_topology: 摄像头拓扑配置
            similarity_threshold: 相似度阈值
            temporal_constraint: 时间约束（秒）
            spatial_constraint: 空间约束（米）
        """
        self.feature_db = feature_db
        self.camera_topology = camera_topology or {}
        self.similarity_threshold = similarity_threshold
        self.temporal_constraint = temporal_constraint
        self.spatial_constraint = spatial_constraint
        
        # 最近观测缓存
        self._recent_observations: Dict[str, Dict[str, Any]] = {}
        
        logger.info("初始化跨摄像头匹配器")
    
    def match_vehicle(
        self,
        features: np.ndarray,
        camera_id: str,
        timestamp: float,
        top_k: int = 10
    ) -> Optional[Match]:
        """匹配车辆
        
        结合外观特征和时空约束进行匹配。
        
        Args:
            features: 特征向量
            camera_id: 当前摄像头ID
            timestamp: 当前时间戳
            top_k: 候选数量
            
        Returns:
            最佳匹配或None
        """
        # 1. 外观特征匹配
        candidates = self.feature_db.search(
            features,
            top_k=top_k,
            threshold=self.similarity_threshold
        )
        
        if not candidates:
            return None
        
        # 2. 时空约束验证
        valid_matches = []
        
        for vehicle_id, similarity, metadata in candidates:
            # 检查时空合理性
            if self._validate_temporal_spatial(
                vehicle_id, camera_id, timestamp, metadata
            ):
                confidence = self._calculate_confidence(
                    similarity, vehicle_id, camera_id, timestamp, metadata
                )
                
                valid_matches.append({
                    'vehicle_id': vehicle_id,
                    'similarity': similarity,
                    'confidence': confidence,
                    'metadata': metadata
                })
        
        if not valid_matches:
            return None
        
        # 3. 选择最佳匹配
        best_match = max(valid_matches, key=lambda x: x['confidence'])
        
        # 计算通行时间
        travel_time = 0.0
        if best_match['metadata'].get('timestamp'):
            travel_time = timestamp - best_match['metadata']['timestamp']
        
        return Match(
            vehicle_id=best_match['vehicle_id'],
            similarity=best_match['similarity'],
            confidence=best_match['confidence'],
            source_camera=best_match['metadata'].get('camera_id', 'unknown'),
            target_camera=camera_id,
            travel_time=travel_time
        )
    
    def _validate_temporal_spatial(
        self,
        vehicle_id: str,
        camera_id: str,
        timestamp: float,
        metadata: Dict[str, Any]
    ) -> bool:
        """验证时空约束
        
        Args:
            vehicle_id: 车辆ID
            camera_id: 当前摄像头ID
            timestamp: 当前时间戳
            metadata: 元数据
            
        Returns:
            是否通过验证
        """
        prev_camera = metadata.get('camera_id')
        prev_timestamp = metadata.get('timestamp')
        
        if not prev_camera or not prev_timestamp:
            return True  # 没有历史记录，无法验证
        
        # 时间约束
        time_diff = timestamp - prev_timestamp
        if time_diff < 0 or time_diff > self.temporal_constraint:
            return False
        
        # 空间约束（如果有拓扑信息）
        if self.camera_topology and prev_camera in self.camera_topology:
            camera_info = self.camera_topology[prev_camera]
            if camera_id in camera_info.get('neighbors', []):
                min_time = camera_info.get('min_travel_time', 0)
                max_time = camera_info.get('max_travel_time', float('inf'))
                
                if not (min_time <= time_diff <= max_time):
                    return False
        
        return True
    
    def _calculate_confidence(
        self,
        similarity: float,
        vehicle_id: str,
        camera_id: str,
        timestamp: float,
        metadata: Dict[str, Any]
    ) -> float:
        """计算匹配置信度
        
        综合外观相似度、时间合理性和空间合理性。
        
        Args:
            similarity: 外观相似度
            vehicle_id: 车辆ID
            camera_id: 当前摄像头ID
            timestamp: 当前时间戳
            metadata: 元数据
            
        Returns:
            置信度分数 (0-1)
        """
        # 外观权重
        appearance_weight = 0.6
        temporal_weight = 0.3
        spatial_weight = 0.1
        
        # 外观分数
        appearance_score = similarity
        
        # 时间分数
        temporal_score = 1.0
        prev_timestamp = metadata.get('timestamp')
        if prev_timestamp:
            time_diff = timestamp - prev_timestamp
            # 时间越接近预期越好
            expected_time = 60  # 假设预期60秒
            temporal_score = max(0, 1.0 - abs(time_diff - expected_time) / expected_time)
        
        # 空间分数
        spatial_score = 1.0
        prev_camera = metadata.get('camera_id')
        if self.camera_topology and prev_camera:
            if prev_camera in self.camera_topology:
                camera_info = self.camera_topology[prev_camera]
                if camera_id in camera_info.get('neighbors', []):
                    spatial_score = 1.0
                else:
                    spatial_score = 0.5
        
        # 加权综合
        confidence = (
            appearance_weight * appearance_score +
            temporal_weight * temporal_score +
            spatial_weight * spatial_score
        )
        
        return confidence
    
    def update_observation(
        self,
        vehicle_id: str,
        camera_id: str,
        timestamp: float,
        features: np.ndarray
    ) -> None:
        """更新车辆观测记录
        
        Args:
            vehicle_id: 车辆ID
            camera_id: 摄像头ID
            timestamp: 时间戳
            features: 特征向量
        """
        self._recent_observations[vehicle_id] = {
            'camera_id': camera_id,
            'timestamp': timestamp,
            'features': features
        }
    
    def register_new_vehicle(
        self,
        features: np.ndarray,
        camera_id: str,
        timestamp: float
    ) -> str:
        """注册新车辆
        
        Args:
            features: 特征向量
            camera_id: 摄像头ID
            timestamp: 时间戳
            
        Returns:
            新车辆ID
        """
        import uuid
        vehicle_id = f"vehicle_{uuid.uuid4().hex[:8]}"
        
        # 存入数据库
        self.feature_db.insert(vehicle_id, features, {
            'camera_id': camera_id,
            'timestamp': timestamp,
            'first_seen': timestamp
        })
        
        # 更新观测记录
        self.update_observation(vehicle_id, camera_id, timestamp, features)
        
        logger.info(f"注册新车辆: {vehicle_id}")
        
        return vehicle_id

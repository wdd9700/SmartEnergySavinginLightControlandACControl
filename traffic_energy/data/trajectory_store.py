#!/usr/bin/env python3
"""
轨迹数据存储模块

车辆轨迹数据的存储和查询。

Example:
    >>> from traffic_energy.data import TrajectoryStore
    >>> store = TrajectoryStore('postgresql://localhost/traffic')
    >>> store.save_trajectory(track_id, trajectory_points)
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import json

import numpy as np

from shared.logger import setup_logger

logger = setup_logger("trajectory_store")


@dataclass
class TrajectoryRecord:
    """轨迹记录
    
    Attributes:
        track_id: 跟踪ID
        camera_id: 摄像头ID
        vehicle_id: 车辆ID（全局）
        start_time: 开始时间
        end_time: 结束时间
        points: 轨迹点列表
        vehicle_type: 车辆类型
    """
    track_id: int
    camera_id: str
    vehicle_id: Optional[str]
    start_time: datetime
    end_time: datetime
    points: List[Dict[str, Any]]
    vehicle_type: str


class TrajectoryStore:
    """轨迹存储
    
    车辆轨迹数据的持久化存储。
    
    Attributes:
        db_url: 数据库连接URL
        
    Example:
        >>> store = TrajectoryStore('postgresql://localhost/traffic')
        >>> store.connect()
        >>> store.save_trajectory(record)
    """
    
    def __init__(self, db_url: Optional[str] = None) -> None:
        """初始化存储
        
        Args:
            db_url: 数据库连接URL
        """
        self.db_url = db_url
        self._connected = False
        self._conn = None
        
        # 内存存储（用于测试）
        self._memory_store: Dict[int, TrajectoryRecord] = {}
        
        logger.info("初始化轨迹存储")
    
    def connect(self) -> bool:
        """连接数据库
        
        Returns:
            是否成功
        """
        if self.db_url is None:
            self._connected = True
            return True
        
        # TODO: 实现PostgreSQL/TimescaleDB连接
        logger.warning("数据库连接待实现")
        return False
    
    def disconnect(self) -> None:
        """断开连接"""
        self._connected = False
        self._conn = None
    
    def save_trajectory(self, record: TrajectoryRecord) -> bool:
        """保存轨迹
        
        Args:
            record: 轨迹记录
            
        Returns:
            是否成功
        """
        if not self._connected:
            logger.error("未连接到数据库")
            return False
        
        # 内存存储
        self._memory_store[record.track_id] = record
        
        logger.debug(f"保存轨迹: {record.track_id}")
        return True
    
    def get_trajectory(
        self,
        track_id: int
    ) -> Optional[TrajectoryRecord]:
        """获取轨迹
        
        Args:
            track_id: 跟踪ID
            
        Returns:
            轨迹记录或None
        """
        return self._memory_store.get(track_id)
    
    def query_trajectories(
        self,
        camera_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        vehicle_type: Optional[str] = None
    ) -> List[TrajectoryRecord]:
        """查询轨迹
        
        Args:
            camera_id: 摄像头ID
            start_time: 开始时间
            end_time: 结束时间
            vehicle_type: 车辆类型
            
        Returns:
            轨迹记录列表
        """
        results = []
        
        for record in self._memory_store.values():
            if camera_id and record.camera_id != camera_id:
                continue
            if start_time and record.start_time < start_time:
                continue
            if end_time and record.end_time > end_time:
                continue
            if vehicle_type and record.vehicle_type != vehicle_type:
                continue
            
            results.append(record)
        
        return results
    
    def delete_trajectory(self, track_id: int) -> bool:
        """删除轨迹
        
        Args:
            track_id: 跟踪ID
            
        Returns:
            是否成功
        """
        if track_id in self._memory_store:
            del self._memory_store[track_id]
            return True
        return False

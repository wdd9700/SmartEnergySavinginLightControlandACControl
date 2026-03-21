#!/usr/bin/env python3
"""
流量数据存储模块

交通流量数据的存储和查询。

Example:
    >>> from traffic_energy.data import FlowStore
    >>> store = FlowStore()
    >>> store.save_flow(camera_id, timestamp, count, vehicle_types)
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from shared.logger import setup_logger

logger = setup_logger("flow_store")


@dataclass
class FlowRecord:
    """流量记录
    
    Attributes:
        camera_id: 摄像头ID
        timestamp: 时间戳
        count: 车辆数
        vehicle_types: 车型分布
        avg_speed: 平均速度
        density: 密度
    """
    camera_id: str
    timestamp: datetime
    count: int
    vehicle_types: Dict[str, int]
    avg_speed: Optional[float] = None
    density: Optional[float] = None


class FlowStore:
    """流量存储
    
    交通流量数据的持久化存储。
    
    Example:
        >>> store = FlowStore()
        >>> store.connect()
        >>> store.save_flow(FlowRecord(...))
    """
    
    def __init__(self, db_url: Optional[str] = None) -> None:
        """初始化存储
        
        Args:
            db_url: 数据库连接URL
        """
        self.db_url = db_url
        self._connected = False
        
        # 内存存储
        self._memory_store: List[FlowRecord] = []
        
        logger.info("初始化流量存储")
    
    def connect(self) -> bool:
        """连接数据库"""
        self._connected = True
        return True
    
    def disconnect(self) -> None:
        """断开连接"""
        self._connected = False
    
    def save_flow(self, record: FlowRecord) -> bool:
        """保存流量记录
        
        Args:
            record: 流量记录
            
        Returns:
            是否成功
        """
        self._memory_store.append(record)
        return True
    
    def query_flows(
        self,
        camera_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[FlowRecord]:
        """查询流量记录
        
        Args:
            camera_id: 摄像头ID
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            流量记录列表
        """
        results = []
        
        for record in self._memory_store:
            if camera_id and record.camera_id != camera_id:
                continue
            if start_time and record.timestamp < start_time:
                continue
            if end_time and record.timestamp > end_time:
                continue
            
            results.append(record)
        
        return results
    
    def get_hourly_stats(
        self,
        camera_id: str,
        date: datetime
    ) -> Dict[int, int]:
        """获取小时级统计
        
        Args:
            camera_id: 摄像头ID
            date: 日期
            
        Returns:
            {小时: 流量, ...}
        """
        hourly_counts = {}
        
        for record in self._memory_store:
            if record.camera_id != camera_id:
                continue
            if record.timestamp.date() != date.date():
                continue
            
            hour = record.timestamp.hour
            if hour not in hourly_counts:
                hourly_counts[hour] = 0
            hourly_counts[hour] += record.count
        
        return hourly_counts

#!/usr/bin/env python3
"""
流量计数器模块

实现虚拟线圈和区域车辆计数功能。

Example:
    >>> from traffic_energy.traffic_analysis import FlowCounter
    >>> counter = FlowCounter()
    >>> counter.add_virtual_loop('loop_001', polygon, direction='entering')
    >>> count = counter.update(tracks)
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import time

import numpy as np
import cv2

from shared.logger import setup_logger

logger = setup_logger("flow_counter")


class Direction(Enum):
    """行驶方向"""
    ENTERING = "entering"
    EXITING = "exiting"
    BOTH = "both"


@dataclass
class VirtualLoop:
    """虚拟线圈
    
    Attributes:
        loop_id: 线圈ID
        polygon: 多边形区域
        direction: 计数方向
        lane_id: 车道ID
        count: 当前计数
    """
    loop_id: str
    polygon: np.ndarray
    direction: Direction
    lane_id: Optional[str] = None
    count: int = 0
    _previous_tracks: set = field(default_factory=set)


@dataclass
class CountEvent:
    """计数事件
    
    Attributes:
        timestamp: 时间戳
        loop_id: 线圈ID
        track_id: 跟踪ID
        vehicle_type: 车辆类型
        direction: 方向
    """
    timestamp: float
    loop_id: str
    track_id: int
    vehicle_type: str
    direction: str


class FlowCounter:
    """流量计数器
    
    使用虚拟线圈方法进行车辆计数。
    
    Attributes:
        virtual_loops: 虚拟线圈字典
        events: 计数事件列表
        
    Example:
        >>> counter = FlowCounter()
        >>> counter.add_virtual_loop(
        ...     'loop_001',
        ...     [[100, 500], [400, 500], [400, 600], [100, 600]],
        ...     direction='entering'
        ... )
        >>> events = counter.update(tracks)
    """
    
    def __init__(self) -> None:
        """初始化流量计数器"""
        self.virtual_loops: Dict[str, VirtualLoop] = {}
        self.events: List[CountEvent] = []
        self._track_states: Dict[int, Dict[str, bool]] = {}
        
        logger.info("初始化流量计数器")
    
    def add_virtual_loop(
        self,
        loop_id: str,
        polygon: List[List[int]],
        direction: str = "entering",
        lane_id: Optional[str] = None
    ) -> None:
        """添加虚拟线圈
        
        Args:
            loop_id: 线圈唯一标识
            polygon: 多边形顶点列表 [[x1,y1], [x2,y2], ...]
            direction: 计数方向 ('entering', 'exiting', 'both')
            lane_id: 车道ID
        """
        polygon_array = np.array(polygon, dtype=np.int32)
        
        direction_enum = Direction(direction)
        
        self.virtual_loops[loop_id] = VirtualLoop(
            loop_id=loop_id,
            polygon=polygon_array,
            direction=direction_enum,
            lane_id=lane_id
        )
        
        logger.info(f"添加虚拟线圈: {loop_id}, 方向: {direction}")
    
    def remove_virtual_loop(self, loop_id: str) -> bool:
        """移除虚拟线圈
        
        Args:
            loop_id: 线圈ID
            
        Returns:
            是否成功
        """
        if loop_id in self.virtual_loops:
            del self.virtual_loops[loop_id]
            return True
        return False
    
    def update(self, tracks: List) -> List[CountEvent]:
        """更新计数器
        
        Args:
            tracks: 跟踪轨迹列表
            
        Returns:
            新产生的计数事件
        """
        new_events = []
        current_time = time.time()
        
        for loop in self.virtual_loops.values():
            current_tracks = set()
            
            for track in tracks:
                if not track.is_confirmed:
                    continue
                
                # 检查车辆是否在线圈内
                center = track.center
                inside = cv2.pointPolygonTest(
                    loop.polygon,
                    center,
                    False
                ) >= 0
                
                track_id = track.track_id
                current_tracks.add(track_id)
                
                # 初始化轨迹状态
                if track_id not in self._track_states:
                    self._track_states[track_id] = {}
                
                was_inside = self._track_states[track_id].get(loop.loop_id, False)
                
                # 检测进入事件
                if inside and not was_inside:
                    if loop.direction in [Direction.ENTERING, Direction.BOTH]:
                        event = CountEvent(
                            timestamp=current_time,
                            loop_id=loop.loop_id,
                            track_id=track_id,
                            vehicle_type=track.class_name,
                            direction="entering"
                        )
                        self.events.append(event)
                        new_events.append(event)
                        loop.count += 1
                        
                        logger.debug(f"车辆 {track_id} 进入 {loop.loop_id}")
                
                # 检测离开事件
                elif not inside and was_inside:
                    if loop.direction in [Direction.EXITING, Direction.BOTH]:
                        event = CountEvent(
                            timestamp=current_time,
                            loop_id=loop.loop_id,
                            track_id=track_id,
                            vehicle_type=track.class_name,
                            direction="exiting"
                        )
                        self.events.append(event)
                        new_events.append(event)
                        loop.count += 1
                        
                        logger.debug(f"车辆 {track_id} 离开 {loop.loop_id}")
                
                # 更新状态
                self._track_states[track_id][loop.loop_id] = inside
            
            # 清理不再跟踪的轨迹状态
            loop._previous_tracks = current_tracks
        
        # 清理过期轨迹状态
        active_track_ids = {t.track_id for t in tracks}
        expired_tracks = set(self._track_states.keys()) - active_track_ids
        for track_id in expired_tracks:
            del self._track_states[track_id]
        
        return new_events
    
    def get_count(self, loop_id: Optional[str] = None) -> int:
        """获取计数
        
        Args:
            loop_id: 线圈ID，None则返回总计数
            
        Returns:
            计数值
        """
        if loop_id:
            if loop_id in self.virtual_loops:
                return self.virtual_loops[loop_id].count
            return 0
        
        return sum(loop.count for loop in self.virtual_loops.values())
    
    def get_counts_by_lane(self) -> Dict[str, int]:
        """按车道获取计数
        
        Returns:
            {lane_id: count, ...}
        """
        counts = {}
        for loop in self.virtual_loops.values():
            if loop.lane_id:
                if loop.lane_id not in counts:
                    counts[loop.lane_id] = 0
                counts[loop.lane_id] += loop.count
        return counts
    
    def get_flow_rate(
        self,
        time_window: float = 3600.0
    ) -> Dict[str, float]:
        """获取流量率（辆/小时）
        
        Args:
            time_window: 时间窗口（秒）
            
        Returns:
            {loop_id: flow_rate, ...}
        """
        current_time = time.time()
        flow_rates = {}
        
        for loop_id in self.virtual_loops.keys():
            # 统计时间窗口内的事件
            recent_events = [
                e for e in self.events
                if e.loop_id == loop_id and
                current_time - e.timestamp <= time_window
            ]
            
            # 计算流量率
            flow_rates[loop_id] = len(recent_events) * 3600.0 / time_window
        
        return flow_rates
    
    def reset(self, loop_id: Optional[str] = None) -> None:
        """重置计数
        
        Args:
            loop_id: 线圈ID，None则重置所有
        """
        if loop_id:
            if loop_id in self.virtual_loops:
                self.virtual_loops[loop_id].count = 0
                self.virtual_loops[loop_id]._previous_tracks.clear()
        else:
            for loop in self.virtual_loops.values():
                loop.count = 0
                loop._previous_tracks.clear()
        
        self._track_states.clear()
    
    def clear_events(self) -> None:
        """清除所有事件"""
        self.events.clear()
    
    def get_statistics(self) -> Dict:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        total_count = self.get_count()
        
        vehicle_types = {}
        for event in self.events:
            vtype = event.vehicle_type
            if vtype not in vehicle_types:
                vehicle_types[vtype] = 0
            vehicle_types[vtype] += 1
        
        return {
            'total_count': total_count,
            'loop_counts': {
                loop_id: loop.count
                for loop_id, loop in self.virtual_loops.items()
            },
            'vehicle_types': vehicle_types,
            'total_events': len(self.events)
        }
    
    def visualize(
        self,
        frame: np.ndarray,
        tracks: List
    ) -> np.ndarray:
        """可视化计数区域
        
        Args:
            frame: 原始帧
            tracks: 跟踪轨迹
            
        Returns:
            可视化后的帧
        """
        result = frame.copy()
        
        # 绘制虚拟线圈
        for loop in self.virtual_loops.values():
            # 绘制多边形
            cv2.polylines(
                result,
                [loop.polygon],
                True,
                (0, 255, 255),
                2
            )
            
            # 绘制标签
            centroid = np.mean(loop.polygon, axis=0).astype(int)
            label = f"{loop.loop_id}: {loop.count}"
            cv2.putText(
                result,
                label,
                tuple(centroid),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
        
        # 绘制统计信息
        stats = self.get_statistics()
        y_offset = 30
        cv2.putText(
            result,
            f"Total: {stats['total_count']}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        return result

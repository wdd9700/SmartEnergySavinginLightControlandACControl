#!/usr/bin/env python3
"""
路径分析模块

分析车辆行驶路径和转向比例。

Example:
    >>> from traffic_energy.traffic_analysis import PathAnalyzer
    >>> analyzer = PathAnalyzer()
    >>> analyzer.add_trajectory(track_id, trajectory_points)
    >>> turns = analyzer.get_turn_ratio()
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import math

import numpy as np

from shared.logger import setup_logger

logger = setup_logger("path_analyzer")


@dataclass
class PathSegment:
    """路径段
    
    Attributes:
        start_point: 起点
        end_point: 终点
        direction: 方向向量
        length: 长度
    """
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    direction: Tuple[float, float]
    length: float


@dataclass
class TurnEvent:
    """转向事件
    
    Attributes:
        track_id: 跟踪ID
        turn_type: 转向类型 ('left', 'right', 'straight', 'u_turn')
        angle: 转向角度
        location: 位置
    """
    track_id: int
    turn_type: str
    angle: float
    location: Tuple[float, float]


class PathAnalyzer:
    """路径分析器
    
    分析车辆行驶路径，统计转向比例。
    
    Example:
        >>> analyzer = PathAnalyzer()
        >>> for track in tracks:
        ...     analyzer.add_trajectory(track.track_id, track.trajectory)
        >>> stats = analyzer.get_statistics()
    """
    
    def __init__(
        self,
        min_segment_length: float = 50.0,
        turn_angle_threshold: float = 30.0
    ) -> None:
        """初始化分析器
        
        Args:
            min_segment_length: 最小路径段长度
            turn_angle_threshold: 转向角度阈值
        """
        self.min_segment_length = min_segment_length
        self.turn_angle_threshold = turn_angle_threshold
        
        self._trajectories: Dict[int, List] = {}
        self._turns: List[TurnEvent] = []
        self._entry_zones: Dict[str, int] = defaultdict(int)
        self._exit_zones: Dict[str, int] = defaultdict(int)
        
        logger.info("初始化路径分析器")
    
    def add_trajectory(
        self,
        track_id: int,
        trajectory: List
    ) -> None:
        """添加轨迹
        
        Args:
            track_id: 跟踪ID
            trajectory: 轨迹点列表
        """
        if len(trajectory) < 3:
            return
        
        self._trajectories[track_id] = trajectory
        
        # 分析转向
        self._analyze_turns(track_id, trajectory)
    
    def _analyze_turns(
        self,
        track_id: int,
        trajectory: List
    ) -> None:
        """分析转向
        
        Args:
            track_id: 跟踪ID
            trajectory: 轨迹点
        """
        points = [(p.center[0], p.center[1]) for p in trajectory if hasattr(p, 'center')]
        
        if len(points) < 3:
            return
        
        # 计算每点的方向
        for i in range(1, len(points) - 1):
            # 入方向
            dx1 = points[i][0] - points[i-1][0]
            dy1 = points[i][1] - points[i-1][1]
            
            # 出方向
            dx2 = points[i+1][0] - points[i][0]
            dy2 = points[i+1][1] - points[i][1]
            
            # 计算转向角度
            angle1 = math.atan2(dy1, dx1)
            angle2 = math.atan2(dy2, dx2)
            
            turn_angle = math.degrees(angle2 - angle1)
            
            # 归一化到 -180 ~ 180
            while turn_angle > 180:
                turn_angle -= 360
            while turn_angle < -180:
                turn_angle += 360
            
            # 判断转向类型
            if abs(turn_angle) < self.turn_angle_threshold:
                turn_type = 'straight'
            elif turn_angle > 0:
                if turn_angle > 150:
                    turn_type = 'u_turn'
                else:
                    turn_type = 'left'
            else:
                if turn_angle < -150:
                    turn_type = 'u_turn'
                else:
                    turn_type = 'right'
            
            if abs(turn_angle) >= self.turn_angle_threshold:
                turn_event = TurnEvent(
                    track_id=track_id,
                    turn_type=turn_type,
                    angle=turn_angle,
                    location=points[i]
                )
                self._turns.append(turn_event)
    
    def get_turn_ratio(self) -> Dict[str, float]:
        """获取转向比例
        
        Returns:
            {转向类型: 比例, ...}
        """
        if not self._turns:
            return {}
        
        turn_counts = defaultdict(int)
        for turn in self._turns:
            turn_counts[turn.turn_type] += 1
        
        total = len(self._turns)
        return {
            turn_type: count / total
            for turn_type, count in turn_counts.items()
        }
    
    def get_origin_destination_matrix(
        self,
        zones: Dict[str, List[Tuple[float, float]]]
    ) -> Dict[Tuple[str, str], int]:
        """获取OD矩阵
        
        Args:
            zones: 区域定义 {zone_id: [polygon_points]}
            
        Returns:
            {(origin, dest): count, ...}
        """
        od_matrix = defaultdict(int)
        
        for track_id, trajectory in self._trajectories.items():
            if len(trajectory) < 2:
                continue
            
            start_point = trajectory[0].center if hasattr(trajectory[0], 'center') else trajectory[0]
            end_point = trajectory[-1].center if hasattr(trajectory[-1], 'center') else trajectory[-1]
            
            origin = self._get_zone(start_point, zones)
            dest = self._get_zone(end_point, zones)
            
            if origin and dest:
                od_matrix[(origin, dest)] += 1
        
        return dict(od_matrix)
    
    def _get_zone(
        self,
        point: Tuple[float, float],
        zones: Dict[str, List[Tuple[float, float]]]
    ) -> Optional[str]:
        """获取点所属区域
        
        Args:
            point: 点坐标
            zones: 区域定义
            
        Returns:
            区域ID或None
        """
        import cv2
        
        for zone_id, polygon in zones.items():
            poly = np.array(polygon, dtype=np.int32)
            if cv2.pointPolygonTest(poly, point, False) >= 0:
                return zone_id
        
        return None
    
    def get_average_trajectory(self) -> Optional[List[Tuple[float, float]]]:
        """获取平均轨迹
        
        Returns:
            平均轨迹点列表或None
        """
        if not self._trajectories:
            return None
        
        # 简化：返回最长轨迹的中心线
        longest_trajectory = max(
            self._trajectories.values(),
            key=lambda t: len(t)
        )
        
        return [
            p.center if hasattr(p, 'center') else p
            for p in longest_trajectory
        ]
    
    def get_statistics(self) -> Dict:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'total_trajectories': len(self._trajectories),
            'total_turns': len(self._turns),
            'turn_ratio': self.get_turn_ratio(),
            'avg_trajectory_length': np.mean([
                len(t) for t in self._trajectories.values()
            ]) if self._trajectories else 0
        }
    
    def clear(self) -> None:
        """清除所有数据"""
        self._trajectories.clear()
        self._turns.clear()
        self._entry_zones.clear()
        self._exit_zones.clear()

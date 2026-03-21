#!/usr/bin/env python3
"""
充电桩调度模块

基于OR-Tools的智能充电调度优化。

Example:
    >>> from traffic_energy.charging import ChargingScheduler
    >>> scheduler = ChargingScheduler()
    >>> schedule = scheduler.optimize(charging_requests, available_piles)
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

import numpy as np

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    cp_model = None

from shared.logger import setup_logger

logger = setup_logger("charging_scheduler")


class ChargingStatus(Enum):
    """充电状态"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    CHARGING = "charging"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ChargingRequest:
    """充电请求
    
    Attributes:
        request_id: 请求ID
        vehicle_id: 车辆ID
        arrival_time: 到达时间
        requested_energy: 请求电量 (kWh)
        deadline: 截止时间
        priority: 优先级 (1-10)
        max_power: 最大功率 (kW)
    """
    request_id: str
    vehicle_id: str
    arrival_time: float
    requested_energy: float
    deadline: float
    priority: int = 5
    max_power: float = 50.0


@dataclass
class ChargingPile:
    """充电桩
    
    Attributes:
        pile_id: 桩ID
        max_power: 最大功率 (kW)
        current_power: 当前功率
        status: 状态
    """
    pile_id: str
    max_power: float = 50.0
    current_power: float = 0.0
    status: str = "available"


@dataclass
class ChargingSchedule:
    """充电调度方案
    
    Attributes:
        request_id: 请求ID
        pile_id: 分配的桩ID
        start_time: 开始时间
        end_time: 结束时间
        power: 充电功率
        energy: 充电电量
    """
    request_id: str
    pile_id: str
    start_time: float
    end_time: float
    power: float
    energy: float


class ChargingScheduler:
    """充电调度器
    
    基于OR-Tools CP-SAT求解器的充电调度优化。
    
    Attributes:
        cost_weight: 成本权重
        wait_time_weight: 等待时间权重
        solver: OR-Tools求解器
        
    Example:
        >>> scheduler = ChargingScheduler(cost_weight=0.6, wait_time_weight=0.4)
        >>> requests = [ChargingRequest(...), ...]
        >>> piles = [ChargingPile(...), ...]
        >>> schedule = scheduler.optimize(requests, piles)
    """
    
    def __init__(
        self,
        cost_weight: float = 0.6,
        wait_time_weight: float = 0.4,
        max_wait_time: float = 1800.0  # 30分钟
    ) -> None:
        """初始化调度器
        
        Args:
            cost_weight: 成本权重
            wait_time_weight: 等待时间权重
            max_wait_time: 最大等待时间（秒）
        """
        if not ORTOOLS_AVAILABLE:
            raise ImportError("ortools未安装")
        
        self.cost_weight = cost_weight
        self.wait_time_weight = wait_time_weight
        self.max_wait_time = max_wait_time
        
        self._schedules: Dict[str, ChargingSchedule] = {}
        
        logger.info("初始化充电调度器")
    
    def optimize(
        self,
        requests: List[ChargingRequest],
        piles: List[ChargingPile],
        time_resolution: int = 60,  # 时间分辨率（秒）
        horizon: int = 86400  # 调度范围（秒）
    ) -> List[ChargingSchedule]:
        """优化充电调度
        
        Args:
            requests: 充电请求列表
            piles: 充电桩列表
            time_resolution: 时间分辨率
            horizon: 调度范围
            
        Returns:
            调度方案列表
        """
        if not requests or not piles:
            return []
        
        model = cp_model.CpModel()
        
        n_requests = len(requests)
        n_piles = len(piles)
        n_slots = horizon // time_resolution
        
        # 决策变量
        # x[i][j][t] = 1 如果请求i在桩j的时间槽t开始充电
        x = {}
        for i in range(n_requests):
            for j in range(n_piles):
                for t in range(n_slots):
                    x[i, j, t] = model.NewBoolVar(f'x_{i}_{j}_{t}')
        
        # 约束1: 每个请求最多分配一个桩和一个时间槽
        for i in range(n_requests):
            model.Add(sum(x[i, j, t] 
                         for j in range(n_piles) 
                         for t in range(n_slots)) <= 1)
        
        # 约束2: 每个桩同一时间只能服务一个请求
        for j in range(n_piles):
            for t in range(n_slots):
                model.Add(sum(x[i, j, t] 
                             for i in range(n_requests)) <= 1)
        
        # 约束3: 时间窗口约束
        for i, req in enumerate(requests):
            arrival_slot = int(req.arrival_time // time_resolution)
            deadline_slot = int(req.deadline // time_resolution)
            
            for j in range(n_piles):
                for t in range(n_slots):
                    if t < arrival_slot or t > deadline_slot:
                        model.Add(x[i, j, t] == 0)
        
        # 目标函数: 最小化成本和等待时间
        # 简化的目标：最大化优先级总和
        objective = []
        for i, req in enumerate(requests):
            for j in range(n_piles):
                for t in range(n_slots):
                    # 优先级越高越好，等待时间越短越好
                    wait_time = t * time_resolution - req.arrival_time
                    score = req.priority * 10 - wait_time / 60
                    objective.append(score * x[i, j, t])
        
        model.Maximize(sum(objective))
        
        # 求解
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0
        
        status = solver.Solve(model)
        
        schedules = []
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            for i, req in enumerate(requests):
                for j, pile in enumerate(piles):
                    for t in range(n_slots):
                        if solver.Value(x[i, j, t]) == 1:
                            start_time = t * time_resolution
                            
                            # 计算充电时长
                            energy_needed = req.requested_energy
                            power = min(req.max_power, pile.max_power)
                            duration = energy_needed / power * 3600  # 转换为秒
                            
                            schedule = ChargingSchedule(
                                request_id=req.request_id,
                                pile_id=pile.pile_id,
                                start_time=start_time,
                                end_time=start_time + duration,
                                power=power,
                                energy=energy_needed
                            )
                            schedules.append(schedule)
                            break
        
        logger.info(f"调度完成: {len(schedules)}/{len(requests)} 请求已安排")
        
        return schedules
    
    def reschedule(
        self,
        active_schedules: List[ChargingSchedule],
        new_requests: List[ChargingRequest],
        piles: List[ChargingPile]
    ) -> List[ChargingSchedule]:
        """重新调度
        
        Args:
            active_schedules: 当前活跃的调度
            new_requests: 新请求
            piles: 充电桩
            
        Returns:
            更新后的调度方案
        """
        # 合并现有调度和新请求
        all_requests = []
        
        # TODO: 实现动态重调度
        
        return self.optimize(all_requests, piles)
    
    def get_schedule(self, request_id: str) -> Optional[ChargingSchedule]:
        """获取调度方案
        
        Args:
            request_id: 请求ID
            
        Returns:
            调度方案或None
        """
        return self._schedules.get(request_id)
    
    def cancel_schedule(self, request_id: str) -> bool:
        """取消调度
        
        Args:
            request_id: 请求ID
            
        Returns:
            是否成功
        """
        if request_id in self._schedules:
            del self._schedules[request_id]
            return True
        return False

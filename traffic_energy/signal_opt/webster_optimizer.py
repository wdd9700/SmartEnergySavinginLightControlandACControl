#!/usr/bin/env python3
"""
Webster信号优化算法

基于Webster方法的传统信号配时优化。

Example:
    >>> from traffic_energy.signal_opt import WebsterOptimizer
    >>> optimizer = WebsterOptimizer()
    >>> timing = optimizer.optimize(flow_rates, saturation_flows)
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass

import numpy as np

from shared.logger import setup_logger

logger = setup_logger("webster_optimizer")


@dataclass
class SignalTiming:
    """信号配时方案
    
    Attributes:
        cycle_length: 周期时长（秒）
        green_times: 各相位绿灯时间
        yellow_times: 各相位黄灯时间
        all_red_time: 全红时间
    """
    cycle_length: float
    green_times: List[float]
    yellow_times: List[float]
    all_red_time: float


class WebsterOptimizer:
    """Webster优化器
    
    基于Webster方法计算最优信号配时。
    
    Attributes:
        min_green: 最小绿灯时间
        max_green: 最大绿灯时间
        yellow_time: 黄灯时间
        all_red_time: 全红时间
        
    Example:
        >>> optimizer = WebsterOptimizer()
        >>> flow_rates = [800, 600]  # 辆/小时
        >>> saturation_flows = [1800, 1800]  # 辆/小时
        >>> timing = optimizer.optimize(flow_rates, saturation_flows)
    """
    
    def __init__(
        self,
        min_green: float = 10.0,
        max_green: float = 60.0,
        yellow_time: float = 3.0,
        all_red_time: float = 2.0,
        target_degree_saturation: float = 0.9
    ) -> None:
        """初始化优化器
        
        Args:
            min_green: 最小绿灯时间
            max_green: 最大绿灯时间
            yellow_time: 黄灯时间
            all_red_time: 全红时间
            target_degree_saturation: 目标饱和度
        """
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.all_red_time = all_red_time
        self.target_degree_saturation = target_degree_saturation
        
        logger.info("初始化Webster优化器")
    
    def optimize(
        self,
        flow_rates: List[float],
        saturation_flows: List[float],
        lost_time: float = 4.0
    ) -> SignalTiming:
        """优化信号配时
        
        Args:
            flow_rates: 各相位流量（辆/小时）
            saturation_flows: 各相位饱和流量（辆/小时）
            lost_time: 每相位损失时间（秒）
            
        Returns:
            信号配时方案
        """
        n_phases = len(flow_rates)
        
        # 计算流量比
        y = [q / s for q, s in zip(flow_rates, saturation_flows)]
        Y = sum(y)
        
        if Y >= 1.0:
            logger.warning("总流量比>=1，需要改善几何条件或相位设计")
            Y = 0.99
        
        # 计算最优周期（Webster公式）
        total_lost_time = n_phases * lost_time
        
        # C0 = (1.5L + 5) / (1 - Y)
        optimal_cycle = (1.5 * total_lost_time + 5) / (1 - Y)
        
        # 限制周期范围
        min_cycle = n_phases * (self.min_green + self.yellow_time + self.all_red_time)
        max_cycle = 180  # 最大180秒
        
        cycle_length = np.clip(optimal_cycle, min_cycle, max_cycle)
        
        # 计算有效绿灯时间
        effective_green = cycle_length - total_lost_time
        
        # 分配绿灯时间
        green_times = []
        for yi in y:
            # gi = (yi / Y) * effective_green
            green_time = (yi / Y) * effective_green
            green_time = np.clip(green_time, self.min_green, self.max_green)
            green_times.append(green_time)
        
        # 调整使总和等于周期
        total_allocated = sum(green_times) + n_phases * (self.yellow_time + self.all_red_time)
        if abs(total_allocated - cycle_length) > 1.0:
            # 按比例调整
            scale = (cycle_length - n_phases * (self.yellow_time + self.all_red_time)) / sum(green_times)
            green_times = [g * scale for g in green_times]
        
        yellow_times = [self.yellow_time] * n_phases
        
        timing = SignalTiming(
            cycle_length=cycle_length,
            green_times=green_times,
            yellow_times=yellow_times,
            all_red_time=self.all_red_time
        )
        
        logger.info(f"优化完成: 周期={cycle_length:.1f}s")
        
        return timing
    
    def calculate_delay(
        self,
        flow_rate: float,
        saturation_flow: float,
        cycle_length: float,
        green_time: float
    ) -> float:
        """计算平均延误（Webster延误公式）
        
        Args:
            flow_rate: 流量（辆/小时）
            saturation_flow: 饱和流量（辆/小时）
            cycle_length: 周期时长（秒）
            green_time: 绿灯时间（秒）
            
        Returns:
            平均延误（秒/辆）
        """
        # 转换为辆/秒
        q = flow_rate / 3600
        s = saturation_flow / 3600
        
        # 绿信比
        lambda_ = green_time / cycle_length
        
        # 饱和度
        x = q / (s * lambda_)
        
        if x >= 1.0:
            return float('inf')
        
        # Webster延误公式
        # d = (c(1-λ)^2) / (2(1-λx)) + (x^2) / (2q(1-x)) - 0.65(c/q^2)^(1/3) * x^(2+5λ)
        
        term1 = (cycle_length * (1 - lambda_)**2) / (2 * (1 - lambda_ * x))
        term2 = (x**2) / (2 * q * (1 - x))
        
        delay = term1 + term2
        
        return delay

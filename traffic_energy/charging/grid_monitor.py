#!/usr/bin/env python3
"""
电网监测模块

监测电网状态并触发需求响应。

Example:
    >>> from traffic_energy.charging import GridMonitor
    >>> monitor = GridMonitor()
    >>> if monitor.check_voltage(voltage):
    ...     monitor.trigger_demand_response()
"""

from typing import Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import time

from shared.logger import setup_logger

logger = setup_logger("grid_monitor")


class GridStatus(Enum):
    """电网状态"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class GridMeasurement:
    """电网测量值
    
    Attributes:
        timestamp: 时间戳
        voltage: 电压 (V)
        frequency: 频率 (Hz)
        load: 负载率 (0-1)
        status: 状态
    """
    timestamp: float
    voltage: float
    frequency: float
    load: float
    status: GridStatus


class GridMonitor:
    """电网监测器
    
    监测电网电压、频率和负载，触发需求响应。
    
    Attributes:
        voltage_min: 最小电压
        voltage_max: 最大电压
        frequency_min: 最小频率
        frequency_max: 最大频率
        
    Example:
        >>> monitor = GridMonitor(voltage_min=210, voltage_max=230)
        >>> monitor.add_listener(on_grid_status_change)
        >>> monitor.update(voltage=220, frequency=50.0, load=0.7)
    """
    
    def __init__(
        self,
        voltage_min: float = 210.0,
        voltage_max: float = 230.0,
        frequency_min: float = 49.5,
        frequency_max: float = 50.5,
        demand_response_threshold: float = 0.8
    ) -> None:
        """初始化监测器
        
        Args:
            voltage_min: 最小电压 (V)
            voltage_max: 最大电压 (V)
            frequency_min: 最小频率 (Hz)
            frequency_max: 最大频率 (Hz)
            demand_response_threshold: 需求响应阈值
        """
        self.voltage_min = voltage_min
        self.voltage_max = voltage_max
        self.frequency_min = frequency_min
        self.frequency_max = frequency_max
        self.demand_response_threshold = demand_response_threshold
        
        self._current_status = GridStatus.NORMAL
        self._listeners: List[Callable[[GridStatus, GridStatus], None]] = []
        self._history: List[GridMeasurement] = []
        
        logger.info("初始化电网监测器")
    
    def update(
        self,
        voltage: float,
        frequency: float,
        load: float
    ) -> GridStatus:
        """更新测量值
        
        Args:
            voltage: 电压
            frequency: 频率
            load: 负载率
            
        Returns:
            当前状态
        """
        # 判断状态
        status = self._determine_status(voltage, frequency, load)
        
        measurement = GridMeasurement(
            timestamp=time.time(),
            voltage=voltage,
            frequency=frequency,
            load=load,
            status=status
        )
        
        self._history.append(measurement)
        
        # 限制历史长度
        if len(self._history) > 1000:
            self._history = self._history[-500:]
        
        # 状态变化时触发回调
        if status != self._current_status:
            old_status = self._current_status
            self._current_status = status
            
            for listener in self._listeners:
                try:
                    listener(old_status, status)
                except Exception as e:
                    logger.error(f"监听器执行失败: {e}")
        
        return status
    
    def _determine_status(
        self,
        voltage: float,
        frequency: float,
        load: float
    ) -> GridStatus:
        """确定电网状态
        
        Args:
            voltage: 电压
            frequency: 频率
            load: 负载率
            
        Returns:
            状态
        """
        # 检查关键条件
        if (voltage < self.voltage_min * 0.95 or
            voltage > self.voltage_max * 1.05 or
            frequency < self.frequency_min - 0.5 or
            frequency > self.frequency_max + 0.5 or
            load > 0.95):
            return GridStatus.CRITICAL
        
        # 检查警告条件
        if (voltage < self.voltage_min or
            voltage > self.voltage_max or
            frequency < self.frequency_min or
            frequency > self.frequency_max or
            load > self.demand_response_threshold):
            return GridStatus.WARNING
        
        return GridStatus.NORMAL
    
    def check_voltage(self, voltage: float) -> bool:
        """检查电压是否正常
        
        Args:
            voltage: 电压
            
        Returns:
            是否正常
        """
        return self.voltage_min <= voltage <= self.voltage_max
    
    def check_frequency(self, frequency: float) -> bool:
        """检查频率是否正常
        
        Args:
            frequency: 频率
            
        Returns:
            是否正常
        """
        return self.frequency_min <= frequency <= self.frequency_max
    
    def should_trigger_demand_response(self) -> bool:
        """检查是否应该触发需求响应
        
        Returns:
            是否触发
        """
        return self._current_status in [GridStatus.WARNING, GridStatus.CRITICAL]
    
    def add_listener(
        self,
        callback: Callable[[GridStatus, GridStatus], None]
    ) -> None:
        """添加状态变化监听器
        
        Args:
            callback: 回调函数 (old_status, new_status)
        """
        self._listeners.append(callback)
    
    def remove_listener(
        self,
        callback: Callable[[GridStatus, GridStatus], None]
    ) -> None:
        """移除监听器
        
        Args:
            callback: 回调函数
        """
        if callback in self._listeners:
            self._listeners.remove(callback)
    
    def get_current_status(self) -> GridStatus:
        """获取当前状态
        
        Returns:
            当前状态
        """
        return self._current_status
    
    def get_statistics(self) -> dict:
        """获取统计信息
        
        Returns:
            统计信息
        """
        if not self._history:
            return {}
        
        voltages = [m.voltage for m in self._history]
        frequencies = [m.frequency for m in self._history]
        loads = [m.load for m in self._history]
        
        return {
            'avg_voltage': sum(voltages) / len(voltages),
            'avg_frequency': sum(frequencies) / len(frequencies),
            'avg_load': sum(loads) / len(loads),
            'max_load': max(loads),
            'status_changes': len([m for m in self._history if m.status != GridStatus.NORMAL])
        }

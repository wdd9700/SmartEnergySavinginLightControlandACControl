#!/usr/bin/env python3
"""
智能交通能源管理系统
基于YOLO12和强化学习的交通能源管理解决方案

主要模块:
    - detection: 车辆检测与跟踪
    - reid: 车辆重识别
    - traffic_analysis: 交通流量分析
    - signal_opt: 信号优化
    - charging: 充电桩调度
    - data: 数据层
    - api: API接口

版本: 1.0.0
作者: Smart Energy Team
"""

__version__ = "1.0.0"
__author__ = "Smart Energy Team"

from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
TRAFFIC_ENERGY_ROOT = Path(__file__).parent

#!/usr/bin/env python3
"""
充电需求预测模块

基于Prophet的充电需求时间序列预测。

Example:
    >>> from traffic_energy.charging import DemandPredictor
    >>> predictor = DemandPredictor()
    >>> predictor.fit(historical_data)
    >>> forecast = predictor.predict(horizon=24)
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import time

import numpy as np
import pandas as pd

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None

from shared.logger import setup_logger

logger = setup_logger("demand_predictor")


@dataclass
class DemandForecast:
    """需求预测结果
    
    Attributes:
        timestamp: 时间戳
        predicted_demand: 预测需求
        lower_bound: 置信区间下限
        upper_bound: 置信区间上限
    """
    timestamp: datetime
    predicted_demand: float
    lower_bound: float
    upper_bound: float


class DemandPredictor:
    """需求预测器
    
    基于Prophet的充电需求预测。
    
    Attributes:
        model: Prophet模型
        forecast_horizon: 预测范围
        
    Example:
        >>> predictor = DemandPredictor()
        >>> df = pd.DataFrame({'ds': dates, 'y': demands})
        >>> predictor.fit(df)
        >>> forecast = predictor.predict(horizon=24)
    """
    
    def __init__(
        self,
        forecast_horizon: int = 24,
        retrain_interval: int = 86400
    ) -> None:
        """初始化预测器
        
        Args:
            forecast_horizon: 预测范围（小时）
            retrain_interval: 重新训练间隔（秒）
        """
        if not PROPHET_AVAILABLE:
            logger.warning("prophet未安装，使用简化预测")
        
        self.forecast_horizon = forecast_horizon
        self.retrain_interval = retrain_interval
        
        self.model = None
        self._last_train_time = 0
        self._historical_data = []
        
        logger.info("初始化需求预测器")
    
    def fit(self, data: pd.DataFrame) -> None:
        """训练模型
        
        Args:
            data: 历史数据 (DataFrame with 'ds' and 'y' columns)
        """
        if not PROPHET_AVAILABLE:
            # 简化版本：保存历史数据
            self._historical_data = data['y'].tolist() if 'y' in data.columns else []
            self._last_train_time = time.time()
            return
        
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True
        )
        
        self.model.fit(data)
        self._last_train_time = time.time()
        
        logger.info("模型训练完成")
    
    def predict(
        self,
        horizon: Optional[int] = None
    ) -> List[DemandForecast]:
        """预测需求
        
        Args:
            horizon: 预测范围（小时），默认使用初始化值
            
        Returns:
            预测结果列表
        """
        hours = horizon or self.forecast_horizon
        
        if not PROPHET_AVAILABLE or self.model is None:
            # 简化预测：使用历史平均值
            return self._simple_predict(hours)
        
        # 创建未来日期
        future = self.model.make_future_dataframe(periods=hours, freq='H')
        
        # 预测
        forecast = self.model.predict(future)
        
        # 提取结果
        results = []
        for i in range(-hours, 0):
            row = forecast.iloc[i]
            results.append(DemandForecast(
                timestamp=row['ds'],
                predicted_demand=max(0, row['yhat']),
                lower_bound=max(0, row['yhat_lower']),
                upper_bound=max(0, row['yhat_upper'])
            ))
        
        return results
    
    def _simple_predict(self, hours: int) -> List[DemandForecast]:
        """简化预测（当Prophet不可用时）"""
        results = []
        
        # 使用历史平均值
        avg_demand = np.mean(self._historical_data) if self._historical_data else 10.0
        
        now = datetime.now()
        for i in range(hours):
            future_time = now + timedelta(hours=i)
            results.append(DemandForecast(
                timestamp=future_time,
                predicted_demand=avg_demand,
                lower_bound=avg_demand * 0.8,
                upper_bound=avg_demand * 1.2
            ))
        
        return results
    
    def should_retrain(self) -> bool:
        """检查是否需要重新训练
        
        Returns:
            是否需要重新训练
        """
        return time.time() - self._last_train_time > self.retrain_interval
    
    def update(self, new_data: pd.DataFrame) -> None:
        """增量更新
        
        Args:
            new_data: 新数据
        """
        if self.should_retrain():
            self.fit(new_data)
        else:
            # 追加到历史数据
            if 'y' in new_data.columns:
                self._historical_data.extend(new_data['y'].tolist())

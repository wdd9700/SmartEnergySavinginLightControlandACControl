#!/usr/bin/env python3
"""
强化学习控制器

基于Stable-Baselines3的交通信号RL控制器。

Example:
    >>> from traffic_energy.signal_opt import RLController
    >>> controller = RLController('PPO')
    >>> controller.train(env, total_timesteps=100000)
    >>> controller.save('model.zip')
"""

from typing import Optional, Dict, Any
from pathlib import Path

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    PPO = SAC = BaseCallback = None

from shared.logger import setup_logger

logger = setup_logger("rl_controller")


class RLController:
    """强化学习控制器
    
    封装Stable-Baselines3 RL算法。
    
    Attributes:
        algorithm: 算法名称 ('PPO', 'SAC')
        model: RL模型
        
    Example:
        >>> controller = RLController('PPO', learning_rate=0.0003)
        >>> controller.train(env, total_timesteps=100000)
        >>> action = controller.predict(obs)
    """
    
    def __init__(
        self,
        algorithm: str = "PPO",
        policy: str = "MlpPolicy",
        learning_rate: float = 0.0003,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        verbose: int = 1,
        tensorboard_log: Optional[str] = None
    ) -> None:
        """初始化控制器
        
        Args:
            algorithm: 算法名称
            policy: 策略网络
            learning_rate: 学习率
            n_steps: 每轮步数
            batch_size: 批次大小
            n_epochs: 训练轮数
            gamma: 折扣因子
            verbose: 日志级别
            tensorboard_log: TensorBoard日志路径
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3未安装")
        
        self.algorithm = algorithm
        self.policy = policy
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.verbose = verbose
        self.tensorboard_log = tensorboard_log
        
        self.model = None
        self.env = None
        
        logger.info(f"初始化RL控制器: {algorithm}")
    
    def create_model(self, env) -> None:
        """创建RL模型
        
        Args:
            env: 环境
        """
        self.env = env
        
        if self.algorithm == "PPO":
            self.model = PPO(
                self.policy,
                env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                verbose=self.verbose,
                tensorboard_log=self.tensorboard_log
            )
        elif self.algorithm == "SAC":
            self.model = SAC(
                self.policy,
                env,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                gamma=self.gamma,
                verbose=self.verbose,
                tensorboard_log=self.tensorboard_log
            )
        else:
            raise ValueError(f"不支持的算法: {self.algorithm}")
        
        logger.info(f"创建{self.algorithm}模型")
    
    def train(
        self,
        env,
        total_timesteps: int = 100000,
        callback: Optional[Any] = None
    ) -> None:
        """训练模型
        
        Args:
            env: 环境
            total_timesteps: 总训练步数
            callback: 回调函数
        """
        if self.model is None:
            self.create_model(env)
        
        logger.info(f"开始训练，总步数: {total_timesteps}")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback
        )
        
        logger.info("训练完成")
    
    def predict(self, observation, deterministic: bool = True):
        """预测动作
        
        Args:
            observation: 观测值
            deterministic: 是否确定性预测
            
        Returns:
            (action, state)
        """
        if self.model is None:
            raise RuntimeError("模型未创建")
        
        return self.model.predict(observation, deterministic=deterministic)
    
    def save(self, path: str) -> None:
        """保存模型
        
        Args:
            path: 保存路径
        """
        if self.model is None:
            raise RuntimeError("模型未创建")
        
        self.model.save(path)
        logger.info(f"模型已保存: {path}")
    
    def load(self, path: str, env=None) -> None:
        """加载模型
        
        Args:
            path: 模型路径
            env: 环境（可选）
        """
        if self.algorithm == "PPO":
            self.model = PPO.load(path, env=env)
        elif self.algorithm == "SAC":
            self.model = SAC.load(path, env=env)
        
        logger.info(f"模型已加载: {path}")

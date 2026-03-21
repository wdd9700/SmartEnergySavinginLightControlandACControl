#!/usr/bin/env python3
"""
特征数据库模块

向量数据库存储和检索车辆特征。

Example:
    >>> from traffic_energy.reid import FeatureDatabase
    >>> db = FeatureDatabase()
    >>> db.connect()
    >>> db.insert(vehicle_id, feature_vector)
    >>> results = db.search(query_vector, top_k=10)
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from shared.logger import setup_logger

logger = setup_logger("feature_database")


class FeatureDatabase:
    """特征数据库
    
    封装向量数据库（Milvus/PGVector）接口。
    
    Attributes:
        db_type: 数据库类型
        host: 主机地址
        port: 端口
        collection_name: 集合名称
        
    Example:
        >>> db = FeatureDatabase(db_type='milvus')
        >>> db.connect()
        >>> db.insert('vehicle_001', feature_vector, {'camera_id': 'cam_001'})
    """
    
    def __init__(
        self,
        db_type: str = "milvus",
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "vehicle_features",
        metric_type: str = "COSINE",
        feature_dim: int = 2048
    ) -> None:
        """初始化特征数据库
        
        Args:
            db_type: 数据库类型 ('milvus', 'pgvector', 'memory')
            host: 主机地址
            port: 端口
            collection_name: 集合名称
            metric_type: 距离度量类型
            feature_dim: 特征维度
        """
        self.db_type = db_type
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.metric_type = metric_type
        self.feature_dim = feature_dim
        
        self._connected = False
        self._client = None
        
        # 内存存储（用于测试）
        self._memory_store: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"初始化特征数据库: {db_type}@{host}:{port}")
    
    def connect(self) -> bool:
        """连接数据库
        
        Returns:
            是否成功
        """
        if self.db_type == "memory":
            self._connected = True
            return True
        
        # TODO: 实现Milvus/PGVector连接
        logger.warning(f"{self.db_type}连接待实现")
        return False
    
    def disconnect(self) -> None:
        """断开连接"""
        self._connected = False
        self._client = None
    
    def insert(
        self,
        vehicle_id: str,
        feature: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """插入特征向量
        
        Args:
            vehicle_id: 车辆ID
            feature: 特征向量
            metadata: 元数据
            
        Returns:
            是否成功
        """
        if not self._connected:
            logger.error("数据库未连接")
            return False
        
        if self.db_type == "memory":
            self._memory_store[vehicle_id] = {
                'feature': feature.copy(),
                'metadata': metadata or {}
            }
            return True
        
        # TODO: 实现向量插入
        return False
    
    def search(
        self,
        query_feature: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.7
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """搜索相似特征
        
        Args:
            query_feature: 查询特征
            top_k: 返回结果数量
            threshold: 相似度阈值
            
        Returns:
            [(vehicle_id, similarity, metadata), ...]
        """
        if not self._connected:
            logger.error("数据库未连接")
            return []
        
        if self.db_type == "memory":
            return self._memory_search(query_feature, top_k, threshold)
        
        # TODO: 实现向量搜索
        return []
    
    def _memory_search(
        self,
        query_feature: np.ndarray,
        top_k: int,
        threshold: float
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """内存搜索实现"""
        results = []
        
        for vehicle_id, data in self._memory_store.items():
            feature = data['feature']
            
            # 计算余弦相似度
            similarity = self._cosine_similarity(query_feature, feature)
            
            if similarity >= threshold:
                results.append((vehicle_id, float(similarity), data['metadata']))
        
        # 按相似度排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def _cosine_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray
    ) -> float:
        """计算余弦相似度"""
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a_norm, b_norm))
    
    def delete(self, vehicle_id: str) -> bool:
        """删除特征
        
        Args:
            vehicle_id: 车辆ID
            
        Returns:
            是否成功
        """
        if self.db_type == "memory" and vehicle_id in self._memory_store:
            del self._memory_store[vehicle_id]
            return True
        
        return False
    
    def clear(self) -> None:
        """清空数据库"""
        self._memory_store.clear()

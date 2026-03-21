#!/usr/bin/env python3
"""
摄像头注册表模块

摄像头管理和拓扑配置。

Example:
    >>> from traffic_energy.data import CameraRegistry
    >>> registry = CameraRegistry()
    >>> registry.register(camera_config)
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from shared.logger import setup_logger

logger = setup_logger("camera_registry")


@dataclass
class CameraConfig:
    """摄像头配置
    
    Attributes:
        camera_id: 摄像头ID
        name: 名称
        location: 位置 [lat, lon]
        stream_url: 视频流URL
        direction: 朝向角度
        roi: 感兴趣区域
        next_cameras: 相邻摄像头
    """
    camera_id: str
    name: str
    location: List[float]
    stream_url: str
    direction: int = 0
    roi: Optional[List[List[int]]] = None
    next_cameras: Optional[List[str]] = None


class CameraRegistry:
    """摄像头注册表
    
    管理摄像头配置和拓扑关系。
    
    Example:
        >>> registry = CameraRegistry()
        >>> registry.register(CameraConfig(...))
        >>> camera = registry.get('cam_001')
    """
    
    def __init__(self) -> None:
        """初始化注册表"""
        self._cameras: Dict[str, CameraConfig] = {}
        self._topology: Dict[str, List[str]] = {}
        
        logger.info("初始化摄像头注册表")
    
    def register(self, config: CameraConfig) -> bool:
        """注册摄像头
        
        Args:
            config: 摄像头配置
            
        Returns:
            是否成功
        """
        self._cameras[config.camera_id] = config
        
        if config.next_cameras:
            self._topology[config.camera_id] = config.next_cameras
        
        logger.info(f"注册摄像头: {config.camera_id}")
        return True
    
    def unregister(self, camera_id: str) -> bool:
        """注销摄像头
        
        Args:
            camera_id: 摄像头ID
            
        Returns:
            是否成功
        """
        if camera_id in self._cameras:
            del self._cameras[camera_id]
            if camera_id in self._topology:
                del self._topology[camera_id]
            return True
        return False
    
    def get(self, camera_id: str) -> Optional[CameraConfig]:
        """获取摄像头配置
        
        Args:
            camera_id: 摄像头ID
            
        Returns:
            配置或None
        """
        return self._cameras.get(camera_id)
    
    def get_all(self) -> List[CameraConfig]:
        """获取所有摄像头
        
        Returns:
            配置列表
        """
        return list(self._cameras.values())
    
    def get_neighbors(self, camera_id: str) -> List[str]:
        """获取相邻摄像头
        
        Args:
            camera_id: 摄像头ID
            
        Returns:
            相邻摄像头ID列表
        """
        return self._topology.get(camera_id, [])
    
    def get_travel_time(
        self,
        from_camera: str,
        to_camera: str
    ) -> Optional[Dict[str, float]]:
        """获取通行时间
        
        Args:
            from_camera: 起始摄像头
            to_camera: 目标摄像头
            
        Returns:
            {min_time, max_time} 或 None
        """
        if to_camera not in self._topology.get(from_camera, []):
            return None
        
        # 默认通行时间
        return {
            'min_time': 10.0,
            'max_time': 120.0
        }
    
    def load_from_config(self, config_path: str) -> bool:
        """从配置文件加载
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            是否成功
        """
        import yaml
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if 'camera_network' in config and 'nodes' in config['camera_network']:
                for node in config['camera_network']['nodes']:
                    camera_config = CameraConfig(
                        camera_id=node['id'],
                        name=node['name'],
                        location=[node['location']['lat'], node['location']['lon']],
                        stream_url=node['stream_url'],
                        direction=node.get('direction', 0),
                        roi=node.get('roi'),
                        next_cameras=node.get('next_cameras', [])
                    )
                    self.register(camera_config)
            
            return True
            
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            return False

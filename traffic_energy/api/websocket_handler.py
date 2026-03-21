#!/usr/bin/env python3
"""
WebSocket处理器模块

实时数据推送的WebSocket接口。

Example:
    >>> from traffic_energy.api import WebSocketHandler
    >>> handler = WebSocketHandler()
    >>> await handler.broadcast({"type": "vehicle_update", "data": {...}})
"""

from typing import Set, Dict, Any, Optional
import json
import asyncio

from shared.logger import setup_logger

logger = setup_logger("websocket_handler")

# 尝试导入WebSocket
try:
    from fastapi import WebSocket, WebSocketDisconnect
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    WebSocket = WebSocketDisconnect = None


class WebSocketHandler:
    """WebSocket处理器
    
    管理WebSocket连接和实时数据推送。
    
    Example:
        >>> handler = WebSocketHandler()
        >>> # 在FastAPI路由中
        >>> @app.websocket("/ws")
        >>> async def websocket_endpoint(websocket: WebSocket):
        ...     await handler.connect(websocket)
    """
    
    def __init__(self) -> None:
        """初始化处理器"""
        if not WEBSOCKET_AVAILABLE:
            logger.warning("fastapi未安装，WebSocket功能不可用")
        
        self.active_connections: Set[WebSocket] = set()
        self._message_queue: asyncio.Queue = asyncio.Queue()
        
        logger.info("初始化WebSocket处理器")
    
    async def connect(self, websocket: WebSocket) -> None:
        """接受新连接
        
        Args:
            websocket: WebSocket连接
        """
        if not WEBSOCKET_AVAILABLE:
            return
        
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"新WebSocket连接，当前连接数: {len(self.active_connections)}")
        
        try:
            while True:
                # 接收消息
                data = await websocket.receive_text()
                await self._handle_message(websocket, data)
                
        except WebSocketDisconnect:
            self.disconnect(websocket)
    
    def disconnect(self, websocket: WebSocket) -> None:
        """断开连接
        
        Args:
            websocket: WebSocket连接
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket断开，当前连接数: {len(self.active_connections)}")
    
    async def _handle_message(self, websocket: WebSocket, data: str) -> None:
        """处理接收到的消息
        
        Args:
            websocket: WebSocket连接
            data: 消息数据
        """
        try:
            message = json.loads(data)
            msg_type = message.get('type')
            
            if msg_type == 'subscribe':
                # 处理订阅请求
                channel = message.get('channel')
                await self._subscribe(websocket, channel)
                
            elif msg_type == 'ping':
                # 心跳响应
                await websocket.send_json({'type': 'pong'})
                
        except json.JSONDecodeError:
            logger.warning(f"收到无效的JSON消息: {data}")
    
    async def _subscribe(self, websocket: WebSocket, channel: str) -> None:
        """处理订阅
        
        Args:
            websocket: WebSocket连接
            channel: 频道名称
        """
        await websocket.send_json({
            'type': 'subscribed',
            'channel': channel
        })
    
    async def broadcast(self, message: Dict[str, Any]) -> None:
        """广播消息到所有连接
        
        Args:
            message: 消息字典
        """
        if not self.active_connections:
            return
        
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"发送消息失败: {e}")
                disconnected.append(connection)
        
        # 清理断开的连接
        for conn in disconnected:
            self.disconnect(conn)
    
    async def send_to_camera(
        self,
        camera_id: str,
        message: Dict[str, Any]
    ) -> None:
        """发送消息到特定摄像头的订阅者
        
        Args:
            camera_id: 摄像头ID
            message: 消息字典
        """
        # TODO: 实现按摄像头过滤发送
        await self.broadcast(message)
    
    async def send_vehicle_update(
        self,
        camera_id: str,
        vehicle_data: Dict[str, Any]
    ) -> None:
        """发送车辆更新
        
        Args:
            camera_id: 摄像头ID
            vehicle_data: 车辆数据
        """
        await self.send_to_camera(camera_id, {
            'type': 'vehicle_update',
            'camera_id': camera_id,
            'data': vehicle_data,
            'timestamp': asyncio.get_event_loop().time()
        })
    
    async def send_congestion_alert(
        self,
        camera_id: str,
        level: str,
        details: Dict[str, Any]
    ) -> None:
        """发送拥堵预警
        
        Args:
            camera_id: 摄像头ID
            level: 拥堵等级
            details: 详细信息
        """
        await self.broadcast({
            'type': 'congestion_alert',
            'camera_id': camera_id,
            'level': level,
            'details': details,
            'timestamp': asyncio.get_event_loop().time()
        })
    
    def get_connection_count(self) -> int:
        """获取连接数
        
        Returns:
            当前连接数
        """
        return len(self.active_connections)

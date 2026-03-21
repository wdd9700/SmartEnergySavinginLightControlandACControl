#!/usr/bin/env python3
"""
智能交通能源管理系统 - 主入口

基于YOLO12和强化学习的交通能源管理解决方案。

Usage:
    python -m traffic_energy.main --config config/default_config.yaml
    python -m traffic_energy.main --source video.mp4 --camera-id cam_001
"""

import argparse
import sys
import signal
from pathlib import Path
from typing import Optional, List

import cv2

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared.logger import setup_logger
from traffic_energy.config import ConfigManager
from traffic_energy.detection import CameraProcessor

logger = setup_logger("traffic_energy_main")


class TrafficEnergySystem:
    """交通能源管理系统主类
    
    协调各模块工作，提供统一的系统管理接口。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化系统
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        self.config = None
        self.processors: List[CameraProcessor] = []
        self._is_running = False
        
        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def initialize(self) -> bool:
        """初始化系统
        
        Returns:
            是否成功
        """
        try:
            # 加载配置
            if self.config_path:
                self.config = self.config_manager.load(self.config_path)
            else:
                self.config = self.config_manager.config
            
            logger.info(f"系统初始化完成: {self.config.system_name} v{self.config.version}")
            return True
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            return False
    
    def add_camera(
        self,
        source: str,
        camera_id: str,
        **kwargs
    ) -> bool:
        """添加摄像头处理器
        
        Args:
            source: 视频源
            camera_id: 摄像头ID
            **kwargs: 其他参数
            
        Returns:
            是否成功
        """
        try:
            processor = CameraProcessor(
                source=source,
                camera_id=camera_id,
                model_path=kwargs.get('model_path', 'yolo12n.pt'),
                conf_threshold=kwargs.get('conf_threshold', 0.5),
                enable_speed=kwargs.get('enable_speed', False)
            )
            self.processors.append(processor)
            logger.info(f"添加摄像头: {camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"添加摄像头失败 {camera_id}: {e}")
            return False
    
    def run_single_camera(
        self,
        source: str,
        camera_id: str,
        display: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """运行单摄像头处理
        
        Args:
            source: 视频源
            camera_id: 摄像头ID
            display: 是否显示结果
            save_path: 保存路径
        """
        logger.info(f"启动单摄像头处理: {camera_id}")
        
        with CameraProcessor(
            source=source,
            camera_id=camera_id,
            model_path='yolo12n.pt',
            conf_threshold=0.5,
            enable_speed=False
        ) as processor:
            
            # 设置视频写入器
            writer = None
            
            try:
                for result in processor:
                    # 显示结果
                    if display:
                        cv2.imshow(f"Traffic Energy - {camera_id}", result.frame)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    # 保存视频
                    if save_path and writer is None:
                        h, w = result.frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(save_path, fourcc, 30, (w, h))
                    
                    if writer:
                        writer.write(result.frame)
                    
                    # 打印统计
                    if processor._frame_count % 30 == 0:
                        stats = processor.get_stats()
                        logger.info(
                            f"[{camera_id}] FPS: {stats['fps']:.1f}, "
                            f"Tracks: {stats['active_tracks']}"
                        )
                        
            finally:
                if writer:
                    writer.release()
                cv2.destroyAllWindows()
    
    def run(self) -> None:
        """运行系统主循环"""
        if not self.processors:
            logger.warning("没有配置摄像头处理器")
            return
        
        self._is_running = True
        
        # 启动所有处理器
        for processor in self.processors:
            processor.start()
        
        logger.info("系统运行中，按Ctrl+C停止")
        
        try:
            while self._is_running:
                # 处理各摄像头结果
                for processor in self.processors:
                    result = processor.process_frame()
                    if result:
                        # 这里可以添加结果处理逻辑
                        pass
                        
        except KeyboardInterrupt:
            logger.info("收到停止信号")
        finally:
            self.shutdown()
    
    def shutdown(self) -> None:
        """关闭系统"""
        logger.info("正在关闭系统...")
        self._is_running = False
        
        for processor in self.processors:
            processor.stop()
        
        logger.info("系统已关闭")
    
    def _signal_handler(self, signum, frame):
        """信号处理"""
        logger.info(f"收到信号 {signum}")
        self.shutdown()
        sys.exit(0)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="智能交通能源管理系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用配置文件运行
  python -m traffic_energy.main --config traffic_energy/config/default_config.yaml
  
  # 处理单个视频文件
  python -m traffic_energy.main --source video.mp4 --camera-id cam_001
  
  # 处理RTSP流
  python -m traffic_energy.main --source rtsp://192.168.1.101/stream --camera-id cam_001
  
  # 使用摄像头
  python -m traffic_energy.main --source 0 --camera-id cam_001
  
  # 保存结果
  python -m traffic_energy.main --source video.mp4 --camera-id cam_001 --save output.mp4
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='配置文件路径'
    )
    parser.add_argument(
        '--source', '-s',
        type=str,
        help='视频源（文件路径、RTSP URL或摄像头索引）'
    )
    parser.add_argument(
        '--camera-id',
        type=str,
        default='cam_001',
        help='摄像头ID（默认: cam_001）'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolo12n.pt',
        help='YOLO模型路径（默认: yolo12n.pt）'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='检测置信度阈值（默认: 0.5）'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='不显示视频窗口'
    )
    parser.add_argument(
        '--save',
        type=str,
        help='保存结果视频的路径'
    )
    
    args = parser.parse_args()
    
    # 创建系统实例
    system = TrafficEnergySystem(args.config)
    
    if not system.initialize():
        logger.error("系统初始化失败")
        sys.exit(1)
    
    # 根据参数运行
    if args.source:
        # 尝试转换摄像头索引
        source = args.source
        try:
            source = int(source)
        except ValueError:
            pass
        
        system.run_single_camera(
            source=source,
            camera_id=args.camera_id,
            display=not args.no_display,
            save_path=args.save
        )
    else:
        # 从配置文件运行
        logger.info("从配置文件加载摄像头...")
        # TODO: 从配置加载摄像头
        system.run()


if __name__ == "__main__":
    main()

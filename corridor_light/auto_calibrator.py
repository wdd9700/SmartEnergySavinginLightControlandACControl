#!/usr/bin/env python3
"""
灯光位置自动校准工具 v4.0
基于亮度分析自动确定:
- 摄像头与灯的相对位置
- 多个摄像头之间的相对位置
- 照明区域边界

使用方法:
1. 单灯开关法: 逐一开关每个灯，分析亮度变化
2. 多摄像头法: 多个摄像头同时观察，通过共享灯光确定相对位置
"""
import sys
import time
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import json

from shared.video_capture import VideoCapture
from corridor_light.brightness_analyzer import BrightnessExtractor, LightBrightnessComparator
from corridor_light.multi_camera_calibrator import MultiCameraCalibrator
from corridor_light.light_zones import LightConfig, LightZone


class LightAutoCalibrator:
    """灯光自动校准器"""
    
    def __init__(self):
        self.extractor = BrightnessExtractor(grid_size=(12, 8))
        self.calibrator = MultiCameraCalibrator()
        self.results = {}
    
    def calibrate_single_camera_single_light(self, 
                                              video_source,
                                              light_controller,
                                              light_id: str = "light_0",
                                              delay: float = 2.0) -> dict:
        """
        单摄像头单灯校准
        
        Args:
            video_source: 视频源
            light_controller: 灯光控制函数 (lambda on/off)
            light_id: 灯光标识
            delay: 开关灯后的等待延迟
        
        Returns:
            校准结果
        """
        print(f"\n{'='*60}")
        print(f"单灯自动校准: {light_id}")
        print(f"{'='*60}")
        
        # 打开视频源
        cap = VideoCapture(video_source)
        if not cap.start():
            print("错误: 无法打开视频源")
            return None
        
        # 等待画面稳定
        print("等待画面稳定...")
        time.sleep(1)
        
        # 1. 关灯拍摄
        print("1. 关闭灯光，记录环境亮度...")
        light_controller(False)
        time.sleep(delay)
        
        frame_off = None
        for _ in range(5):  # 取几帧平均
            frame = cap.read()
            if frame is not None:
                frame_off = frame
        
        if frame_off is None:
            print("错误: 无法获取关灯帧")
            cap.stop()
            return None
        
        # 2. 开灯拍摄
        print("2. 开启灯光，记录照明亮度...")
        light_controller(True)
        time.sleep(delay)
        
        frame_on = None
        for _ in range(5):
            frame = cap.read()
            if frame is not None:
                frame_on = frame
        
        if frame_on is None:
            print("错误: 无法获取开灯帧")
            cap.stop()
            return None
        
        # 3. 分析亮度差异
        print("3. 分析亮度差异...")
        comparator = LightBrightnessComparator(self.extractor)
        comparator.capture_light_off(frame_off)
        comparator.capture_light_on(frame_on)
        
        # 计算灯光贡献
        regions = comparator.compute_light_contribution()
        
        # 估算灯位置
        light_pos = comparator.estimate_light_source_position()
        radius = comparator.estimate_illumination_radius()
        
        if not light_pos:
            print("警告: 无法确定灯光位置")
            cap.stop()
            return None
        
        h, w = frame_on.shape[:2]
        
        result = {
            'light_id': light_id,
            'pixel_position': light_pos,
            'normalized_position': (light_pos[0] / w, light_pos[1] / h),
            'estimated_radius': radius,
            'frame_shape': (h, w),
            'brightness_contribution': comparator.compute_light_contribution()
        }
        
        print(f"\n校准结果:")
        print(f"  画面位置: ({light_pos[0]}, {light_pos[1]})")
        print(f"  归一化位置: ({light_pos[0]/w:.3f}, {light_pos[1]/h:.3f})")
        print(f"  估算照明半径: {radius:.1f} 像素")
        print(f"  画面尺寸: {w}x{h}")
        
        # 4. 可视化并保存
        vis_off = self.extractor.visualize_brightness(frame_off, show_grid=True, show_isophotes=True)
        vis_on = self.extractor.visualize_brightness(frame_on, show_grid=True, show_isophotes=True)
        vis_diff = comparator.visualize_contribution(frame_on)
        
        # 保存结果图
        cv2.imwrite(f"calib_{light_id}_off.jpg", vis_off)
        cv2.imwrite(f"calib_{light_id}_on.jpg", vis_on)
        cv2.imwrite(f"calib_{light_id}_contribution.jpg", vis_diff)
        
        print(f"  结果图已保存: calib_{light_id}_*.jpg")
        
        cap.stop()
        return result
    
    def calibrate_single_camera_multiple_lights(self,
                                                 video_source,
                                                 light_controllers: dict,
                                                 delay: float = 2.0) -> LightConfig:
        """
        单摄像头多灯校准
        
        Args:
            video_source: 视频源
            light_controllers: {light_id: controller_func}
            delay: 开关灯延迟
        
        Returns:
            灯光配置
        """
        print(f"\n{'='*60}")
        print(f"多灯自动校准: {len(light_controllers)} 个灯")
        print(f"{'='*60}")
        
        # 先关闭所有灯
        print("\n关闭所有灯光...")
        for light_id, controller in light_controllers.items():
            controller(False)
        time.sleep(delay)
        
        # 获取环境基准
        cap = VideoCapture(video_source)
        if not cap.start():
            print("错误: 无法打开视频源")
            return None
        
        time.sleep(1)
        frame_baseline = cap.read()
        cap.stop()
        
        if frame_baseline is None:
            print("错误: 无法获取基准帧")
            return None
        
        h, w = frame_baseline.shape[:2]
        
        # 逐一校准每个灯
        light_zones = []
        light_connections = {}  # 记录灯光间的空间关系
        
        for light_id, controller in light_controllers.items():
            print(f"\n校准灯光: {light_id}")
            
            # 关闭其他灯，只开当前灯
            for other_id, other_ctrl in light_controllers.items():
                other_ctrl(other_id == light_id)
            
            time.sleep(delay)
            
            cap = VideoCapture(video_source)
            cap.start()
            time.sleep(0.5)
            frame_on = cap.read()
            cap.stop()
            
            if frame_on is None:
                print(f"  警告: 无法获取 {light_id} 的帧")
                continue
            
            # 分析
            comparator = LightBrightnessComparator(self.extractor)
            comparator.capture_light_off(frame_baseline)
            comparator.capture_light_on(frame_on)
            
            light_pos = comparator.estimate_light_source_position()
            radius = comparator.estimate_illumination_radius()
            
            if light_pos:
                # 创建灯光区域
                zone = LightZone(
                    id=light_id,
                    name=f"灯{len(light_zones)+1}",
                    x=light_pos[0],
                    y=light_pos[1],
                    radius=int(radius),
                    forward_zones=[],
                    backward_zones=[]
                )
                light_zones.append(zone)
                light_connections[light_id] = {
                    'position': light_pos,
                    'radius': radius
                }
                
                print(f"  位置: ({light_pos[0]}, {light_pos[1]}), 半径: {radius:.1f}px")
        
        # 关闭所有灯
        for controller in light_controllers.values():
            controller(False)
        
        # 根据位置关系建立前向/后向连接
        if len(light_zones) > 1:
            # 按x坐标排序（假设走廊是水平布局）
            sorted_zones = sorted(light_zones, key=lambda z: z.x)
            
            for i, zone in enumerate(sorted_zones):
                if i < len(sorted_zones) - 1:
                    zone.forward_zones = [sorted_zones[i+1].id]
                if i > 0:
                    zone.backward_zones = [sorted_zones[i-1].id]
        
        # 创建配置
        config = LightConfig(light_zones)
        
        print(f"\n校准完成! 共 {len(light_zones)} 个灯光区域")
        
        return config
    
    def calibrate_multi_camera(self,
                                camera_configs: list,
                                shared_light_id: str = "light_0",
                                delay: float = 2.0) -> MultiCameraCalibrator:
        """
        多摄像头校准
        
        Args:
            camera_configs: [
                {'id': 'cam1', 'source': 0, 'light_controller': func},
                ...
            ]
            shared_light_id: 用于校准的共享灯光ID
            delay: 开关灯延迟
        
        Returns:
            多摄像头校准器
        """
        print(f"\n{'='*60}")
        print(f"多摄像头校准: {len(camera_configs)} 个摄像头")
        print(f"{'='*60}")
        
        # 1. 关闭所有灯，获取基准
        print("\n1. 关闭所有灯光，记录基准...")
        for cam_config in camera_configs:
            if 'light_controller' in cam_config:
                cam_config['light_controller'](False)
        time.sleep(delay)
        
        frames_off = {}
        for cam_config in camera_configs:
            cap = VideoCapture(cam_config['source'])
            if cap.start():
                time.sleep(0.5)
                frame = cap.read()
                if frame is not None:
                    frames_off[cam_config['id']] = frame
                    self.calibrator.register_camera(cam_config['id'], frame.shape[:2])
                cap.stop()
        
        # 2. 逐一开关每个摄像头的灯
        print("\n2. 逐一分析每个摄像头的灯光...")
        for cam_config in camera_configs:
            cam_id = cam_config['id']
            controller = cam_config.get('light_controller')
            
            if not controller:
                continue
            
            print(f"\n  分析 {cam_id}...")
            
            # 开灯
            controller(True)
            time.sleep(delay)
            
            cap = VideoCapture(cam_config['source'])
            if cap.start():
                time.sleep(0.5)
                frame_on = cap.read()
                cap.stop()
                
                if frame_on is not None and cam_id in frames_off:
                    # 分析该摄像头
                    comparator = LightBrightnessComparator(self.extractor)
                    comparator.capture_light_off(frames_off[cam_id])
                    comparator.capture_light_on(frame_on)
                    
                    light_pos = comparator.estimate_light_source_position()
                    radius = comparator.estimate_illumination_radius()
                    
                    if light_pos:
                        self.calibrator.cameras[cam_id].detected_lights[shared_light_id] = light_pos
                        self.calibrator.cameras[cam_id].light_radii[shared_light_id] = radius
                        self.calibrator.cameras[cam_id].normalized_lights[shared_light_id] = (
                            light_pos[0] / frames_off[cam_id].shape[1],
                            light_pos[1] / frames_off[cam_id].shape[0]
                        )
                        print(f"    检测到灯光: ({light_pos[0]}, {light_pos[1]})")
            
            controller(False)
        
        # 3. 计算相对位置
        print("\n3. 计算摄像头相对位置...")
        self.calibrator.calibrate_relative_positions()
        
        # 4. 创建共享坐标系
        print("\n4. 创建共享坐标系...")
        global_lights = self.calibrator.create_shared_coordinate_system()
        print(f"全局灯光位置:")
        for light_id, pos in global_lights.items():
            print(f"  {light_id}: ({pos[0]:.3f}, {pos[1]:.3f})")
        
        return self.calibrator
    
    def save_results(self, filepath: str):
        """保存校准结果"""
        self.calibrator.save_calibration(filepath)
        print(f"校准结果已保存: {filepath}")


def demo_mode():
    """演示模式 - 使用测试视频模拟"""
    print("=" * 60)
    print("灯光自动校准 - 演示模式")
    print("=" * 60)
    print("\n使用测试视频模拟灯光开关效果...")
    
    # 加载测试视频
    video_path = "tests/test_corridor.mp4"
    if not Path(video_path).exists():
        print(f"错误: 测试视频不存在: {video_path}")
        return
    
    cap = VideoCapture(video_path)
    if not cap.start():
        print("错误: 无法打开测试视频")
        return
    
    # 获取一帧作为基准
    frame_baseline = cap.read()
    if frame_baseline is None:
        print("错误: 无法读取视频帧")
        cap.stop()
        return
    
    h, w = frame_baseline.shape[:2]
    
    # 模拟灯光效果 (添加一个亮度渐变区域)
    def simulate_light(frame, center, radius, intensity=80):
        result = frame.copy()
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, center, radius, 1, -1)
        
        # 高斯模糊模拟光照衰减
        mask = cv2.GaussianBlur(mask, (radius//2*2+1, radius//2*2+1), radius/3)
        
        # 转换为3通道
        mask = np.stack([mask] * 3, axis=2)
        
        # 增加亮度
        light = np.ones_like(result, dtype=np.float32) * intensity
        result = result.astype(np.float32) + light * mask
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    # 模拟3个灯的位置
    light_positions = [
        (w // 4, h // 2),
        (w // 2, h // 2),
        (w * 3 // 4, h // 2)
    ]
    
    calibrator = LightAutoCalibrator()
    
    print(f"\n模拟 {len(light_positions)} 个灯光...")
    print("实际部署时应逐一开关真实灯光")
    
    # 分析每个"灯"
    for i, pos in enumerate(light_positions):
        light_id = f"light_{i}"
        print(f"\n分析灯光 {light_id} (模拟位置: {pos})...")
        
        # 模拟开灯帧
        frame_on = simulate_light(frame_baseline, pos, 120)
        
        # 分析
        comparator = LightBrightnessComparator(calibrator.extractor)
        comparator.capture_light_off(frame_baseline)
        comparator.capture_light_on(frame_on)
        
        estimated_pos = comparator.estimate_light_source_position()
        radius = comparator.estimate_illumination_radius()
        
        if estimated_pos:
            error = np.sqrt((estimated_pos[0] - pos[0])**2 + (estimated_pos[1] - pos[1])**2)
            print(f"  估算位置: {estimated_pos}")
            print(f"  实际位置: {pos}")
            print(f"  误差: {error:.1f} 像素")
            print(f"  照明半径: {radius:.1f} 像素")
            
            # 保存可视化
            vis = comparator.visualize_contribution(frame_on)
            cv2.imwrite(f"demo_calib_{light_id}.jpg", vis)
    
    cap.stop()
    print("\n演示完成! 查看 demo_calib_*.jpg 了解结果")


def main():
    parser = argparse.ArgumentParser(description='灯光位置自动校准工具 v4.0')
    parser.add_argument('--demo', action='store_true',
                       help='运行演示模式')
    parser.add_argument('--source', type=str, default='0',
                       help='视频源 (0=摄像头, 路径=视频文件)')
    parser.add_argument('--delay', type=float, default=2.0,
                       help='开关灯后等待延迟(秒)')
    parser.add_argument('--output', type=str, default='light_calibration.json',
                       help='校准结果输出文件')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_mode()
        return
    
    print("=" * 60)
    print("灯光位置自动校准工具 v4.0")
    print("=" * 60)
    print("\n请按照提示逐一开关灯光进行校准")
    print("或者使用 --demo 查看模拟效果")
    print("\n实际使用示例:")
    print("  1. 连接摄像头和可控制灯光")
    print("  2. 逐一开关每个灯，记录亮度变化")
    print("  3. 系统自动计算灯光位置和照明范围")


if __name__ == "__main__":
    main()

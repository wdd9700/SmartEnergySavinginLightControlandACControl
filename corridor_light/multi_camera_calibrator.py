#!/usr/bin/env python3
"""
多摄像头位置校准模块
基于灯光亮度分析确定:
1. 摄像头与灯的相对位置
2. 多个摄像头之间的相对位置
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json

from corridor_light.brightness_analyzer import BrightnessExtractor, LightBrightnessComparator


@dataclass
class CameraPosition:
    """摄像头位置信息"""
    camera_id: str                    # 摄像头标识
    frame_shape: Tuple[int, int]      # 画面尺寸 (h, w)
    
    # 检测到的灯光位置 (在画面坐标系中)
    detected_lights: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    
    # 灯光照明半径 (像素)
    light_radii: Dict[str, float] = field(default_factory=dict)
    
    # 归一化坐标 (0-1范围，用于跨摄像头比较)
    normalized_lights: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # 相对于参考摄像头的偏移 (用于多摄像头校准)
    relative_offset: Tuple[float, float] = (0.0, 0.0)
    scale_factor: float = 1.0


@dataclass
class LightSource:
    """灯光源信息 (全局坐标系)"""
    light_id: str                     # 灯标识
    global_position: Tuple[float, float]  # 全局坐标 (可基于某个参考)
    estimated_power: float            # 估算功率/亮度
    coverage_radius: float            # 覆盖半径
    detected_by_cameras: List[str] = field(default_factory=list)  # 检测到该灯的摄像头


class MultiCameraCalibrator:
    """多摄像头校准器"""
    
    def __init__(self):
        self.cameras: Dict[str, CameraPosition] = {}
        self.lights: Dict[str, LightSource] = {}
        self.reference_camera: Optional[str] = None
        
        # 共享灯光匹配数据
        self.shared_lights: Dict[str, Dict[str, Tuple[int, int]]] = defaultdict(dict)
    
    def register_camera(self, camera_id: str, frame_shape: Tuple[int, int]):
        """注册摄像头"""
        self.cameras[camera_id] = CameraPosition(
            camera_id=camera_id,
            frame_shape=frame_shape
        )
        print(f"注册摄像头: {camera_id} ({frame_shape[1]}x{frame_shape[0]})")
    
    def analyze_single_camera(self, camera_id: str,
                               light_on_frame: np.ndarray,
                               light_off_frame: np.ndarray,
                               light_id: str = "light_0") -> bool:
        """
        分析单个摄像头的灯光位置
        
        Args:
            camera_id: 摄像头ID
            light_on_frame: 灯开启帧
            light_off_frame: 灯关闭帧
            light_id: 灯光标识
        
        Returns:
            是否成功
        """
        if camera_id not in self.cameras:
            print(f"错误: 摄像头 {camera_id} 未注册")
            return False
        
        # 使用亮度对比器分析
        comparator = LightBrightnessComparator()
        comparator.capture_light_on(light_on_frame)
        comparator.capture_light_off(light_off_frame)
        
        # 估算灯光位置
        light_pos = comparator.estimate_light_source_position()
        if not light_pos:
            print(f"警告: 无法估算 {camera_id} 的灯光位置")
            return False
        
        # 估算照明半径
        radius = comparator.estimate_illumination_radius()
        
        # 保存结果
        camera = self.cameras[camera_id]
        camera.detected_lights[light_id] = light_pos
        camera.light_radii[light_id] = radius
        
        # 计算归一化坐标
        h, w = camera.frame_shape
        normalized_x = light_pos[0] / w
        normalized_y = light_pos[1] / h
        camera.normalized_lights[light_id] = (normalized_x, normalized_y)
        
        # 记录到共享数据
        self.shared_lights[light_id][camera_id] = light_pos
        
        print(f"  {camera_id} 检测到 {light_id}: "
              f"画面位置({light_pos[0]}, {light_pos[1]}), "
              f"归一化({normalized_x:.3f}, {normalized_y:.3f}), "
              f"半径{radius:.1f}px")
        
        return True
    
    def calibrate_relative_positions(self, reference_camera_id: str = None):
        """
        校准摄像头相对位置
        
        基于共同检测到的灯光，计算摄像头间的相对偏移和缩放
        """
        if not self.cameras:
            print("错误: 没有注册的摄像头")
            return False
        
        # 选择参考摄像头
        if reference_camera_id:
            self.reference_camera = reference_camera_id
        else:
            # 选择检测到最多灯光的摄像头作为参考
            self.reference_camera = max(self.cameras.keys(),
                                       key=lambda cid: len(self.cameras[cid].detected_lights))
        
        ref_camera = self.cameras[self.reference_camera]
        print(f"\n参考摄像头: {self.reference_camera}")
        
        # 对每个非参考摄像头计算相对位置
        for camera_id, camera in self.cameras.items():
            if camera_id == self.reference_camera:
                continue
            
            # 找到共同检测到的灯光
            common_lights = set(ref_camera.detected_lights.keys()) & \
                          set(camera.detected_lights.keys())
            
            if len(common_lights) < 1:
                print(f"  警告: {camera_id} 与参考摄像头没有共同检测到的灯光")
                continue
            
            if len(common_lights) >= 2:
                # 使用两个灯光计算偏移和缩放
                self._compute_transform_with_two_lights(
                    ref_camera, camera, list(common_lights)[:2]
                )
            else:
                # 只使用一个灯光，仅计算偏移（假设缩放相同）
                self._compute_transform_with_one_light(
                    ref_camera, camera, list(common_lights)[0]
                )
        
        return True
    
    def _compute_transform_with_one_light(self, ref_cam: CameraPosition, 
                                          target_cam: CameraPosition,
                                          light_id: str):
        """使用单个灯光计算相对偏移"""
        # 归一化坐标差作为偏移
        ref_pos = ref_cam.normalized_lights[light_id]
        target_pos = target_cam.normalized_lights[light_id]
        
        offset_x = target_pos[0] - ref_pos[0]
        offset_y = target_pos[1] - ref_pos[1]
        
        target_cam.relative_offset = (offset_x, offset_y)
        target_cam.scale_factor = 1.0  # 无法确定缩放
        
        print(f"  {target_cam.camera_id}: 偏移 ({offset_x:+.3f}, {offset_y:+.3f}), "
              f"缩放 1.0 (基于1个共同灯光)")
    
    def _compute_transform_with_two_lights(self, ref_cam: CameraPosition,
                                           target_cam: CameraPosition,
                                           light_ids: List[str]):
        """使用两个灯光计算相对偏移和缩放"""
        # 获取两个灯光在各自坐标系中的位置
        ref_pos1 = ref_cam.normalized_lights[light_ids[0]]
        ref_pos2 = ref_cam.normalized_lights[light_ids[1]]
        target_pos1 = target_cam.normalized_lights[light_ids[0]]
        target_pos2 = target_cam.normalized_lights[light_ids[1]]
        
        # 计算参考系中两灯之间的距离
        ref_dist = np.sqrt((ref_pos1[0] - ref_pos2[0])**2 + 
                          (ref_pos1[1] - ref_pos2[1])**2)
        
        # 计算目标系中两灯之间的距离
        target_dist = np.sqrt((target_pos1[0] - target_pos2[0])**2 + 
                             (target_pos1[1] - target_pos2[1])**2)
        
        if ref_dist > 0 and target_dist > 0:
            # 计算缩放因子
            scale = target_dist / ref_dist
            target_cam.scale_factor = scale
        else:
            scale = 1.0
            target_cam.scale_factor = 1.0
        
        # 计算偏移 (使用第一个灯光作为参考点)
        offset_x = target_pos1[0] - ref_pos1[0] * scale
        offset_y = target_pos1[1] - ref_pos1[1] * scale
        target_cam.relative_offset = (offset_x, offset_y)
        
        print(f"  {target_cam.camera_id}: 偏移 ({offset_x:+.3f}, {offset_y:+.3f}), "
              f"缩放 {scale:.3f} (基于2个共同灯光)")
    
    def map_position_between_cameras(self, position: Tuple[int, int],
                                      from_camera: str,
                                      to_camera: str) -> Optional[Tuple[int, int]]:
        """
        将位置从一个摄像头坐标系映射到另一个
        
        Args:
            position: (x, y) 在from_camera中的位置
            from_camera: 源摄像头ID
            to_camera: 目标摄像头ID
        
        Returns:
            在to_camera中的位置
        """
        if from_camera not in self.cameras or to_camera not in self.cameras:
            return None
        
        from_cam = self.cameras[from_camera]
        to_cam = self.cameras[to_camera]
        
        # 归一化
        h, w = from_cam.frame_shape
        norm_x = position[0] / w
        norm_y = position[1] / h
        
        # 如果目标不是参考摄像头，需要反向变换
        if to_camera != self.reference_camera:
            # 先转换到参考坐标系
            if from_camera != self.reference_camera:
                # from -> ref
                ref_x = (norm_x - from_cam.relative_offset[0]) / from_cam.scale_factor
                ref_y = (norm_y - from_cam.relative_offset[1]) / from_cam.scale_factor
            else:
                ref_x, ref_y = norm_x, norm_y
            
            # ref -> to
            to_x = ref_x * to_cam.scale_factor + to_cam.relative_offset[0]
            to_y = ref_y * to_cam.scale_factor + to_cam.relative_offset[1]
        else:
            # 直接转换到参考系
            if from_camera != self.reference_camera:
                to_x = (norm_x - from_cam.relative_offset[0]) / from_cam.scale_factor
                to_y = (norm_y - from_cam.relative_offset[1]) / from_cam.scale_factor
            else:
                to_x, to_y = norm_x, norm_y
        
        # 反归一化到目标摄像头像素坐标
        to_h, to_w = to_cam.frame_shape
        pixel_x = int(to_x * to_w)
        pixel_y = int(to_y * to_h)
        
        return (pixel_x, pixel_y)
    
    def create_shared_coordinate_system(self) -> Dict[str, Tuple[float, float]]:
        """
        创建共享坐标系
        
        将所有检测到的灯光位置转换到统一的全局坐标系
        
        Returns:
            {light_id: (global_x, global_y)} 全局坐标在0-1范围内
        """
        global_lights = {}
        
        for light_id, camera_detections in self.shared_lights.items():
            # 收集所有摄像头对该灯的检测（归一化坐标）
            positions = []
            for camera_id, pixel_pos in camera_detections.items():
                camera = self.cameras[camera_id]
                h, w = camera.frame_shape
                
                # 转换到参考坐标系
                norm_x = pixel_pos[0] / w
                norm_y = pixel_pos[1] / h
                
                if camera_id != self.reference_camera:
                    ref_x = (norm_x - camera.relative_offset[0]) / camera.scale_factor
                    ref_y = (norm_y - camera.relative_offset[1]) / camera.scale_factor
                else:
                    ref_x, ref_y = norm_x, norm_y
                
                positions.append((ref_x, ref_y))
            
            # 平均所有摄像头的检测
            if positions:
                avg_x = sum(p[0] for p in positions) / len(positions)
                avg_y = sum(p[1] for p in positions) / len(positions)
                global_lights[light_id] = (avg_x, avg_y)
        
        return global_lights
    
    def save_calibration(self, filepath: str):
        """保存校准结果"""
        data = {
            'reference_camera': self.reference_camera,
            'cameras': {},
            'shared_lights': dict(self.shared_lights),
            'global_light_positions': self.create_shared_coordinate_system()
        }
        
        for camera_id, camera in self.cameras.items():
            data['cameras'][camera_id] = {
                'frame_shape': camera.frame_shape,
                'detected_lights': camera.detected_lights,
                'light_radii': camera.light_radii,
                'normalized_lights': camera.normalized_lights,
                'relative_offset': camera.relative_offset,
                'scale_factor': camera.scale_factor
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n校准结果已保存: {filepath}")
    
    def load_calibration(self, filepath: str):
        """加载校准结果"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.reference_camera = data.get('reference_camera')
        self.shared_lights = defaultdict(dict, data.get('shared_lights', {}))
        
        for camera_id, cam_data in data['cameras'].items():
            self.cameras[camera_id] = CameraPosition(
                camera_id=camera_id,
                frame_shape=tuple(cam_data['frame_shape']),
                detected_lights=cam_data['detected_lights'],
                light_radii=cam_data['light_radii'],
                normalized_lights={k: tuple(v) for k, v in cam_data['normalized_lights'].items()},
                relative_offset=tuple(cam_data['relative_offset']),
                scale_factor=cam_data['scale_factor']
            )
        
        print(f"已加载校准结果: {filepath}")
        print(f"参考摄像头: {self.reference_camera}")
        print(f"摄像头数量: {len(self.cameras)}")
    
    def visualize_calibration(self, frame_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        可视化校准结果
        
        Args:
            frame_dict: {camera_id: frame}
        
        Returns:
            可视化后的帧字典
        """
        result = {}
        
        # 获取全局灯光位置
        global_lights = self.create_shared_coordinate_system()
        
        for camera_id, frame in frame_dict.items():
            if camera_id not in self.cameras:
                result[camera_id] = frame
                continue
            
            display = frame.copy()
            camera = self.cameras[camera_id]
            h, w = camera.frame_shape
            
            # 绘制检测到的灯光位置
            for light_id, pos in camera.detected_lights.items():
                cv2.circle(display, pos, 10, (0, 255, 255), -1)
                cv2.circle(display, pos, int(camera.light_radii.get(light_id, 50)), 
                          (0, 255, 255), 2)
                cv2.putText(display, light_id, (pos[0] - 30, pos[1] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 绘制全局坐标系中的灯光位置（映射回当前摄像头）
            for light_id, global_pos in global_lights.items():
                # 从参考系映射到当前摄像头
                if camera_id != self.reference_camera:
                    local_x = global_pos[0] * camera.scale_factor + camera.relative_offset[0]
                    local_y = global_pos[1] * camera.scale_factor + camera.relative_offset[1]
                else:
                    local_x, local_y = global_pos
                
                pixel_x = int(local_x * w)
                pixel_y = int(local_y * h)
                
                # 绘制为红色叉
                size = 15
                cv2.line(display, (pixel_x - size, pixel_y - size),
                        (pixel_x + size, pixel_y + size), (0, 0, 255), 2)
                cv2.line(display, (pixel_x - size, pixel_y + size),
                        (pixel_x + size, pixel_y - size), (0, 0, 255), 2)
            
            # 显示相对位置信息
            if camera_id != self.reference_camera:
                info = f"Offset: ({camera.relative_offset[0]:+.3f}, {camera.relative_offset[1]:+.3f})"
                info2 = f"Scale: {camera.scale_factor:.3f}"
                cv2.putText(display, info, (10, h - 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(display, info2, (10, h - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(display, "REFERENCE", (10, h - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            result[camera_id] = display
        
        return result

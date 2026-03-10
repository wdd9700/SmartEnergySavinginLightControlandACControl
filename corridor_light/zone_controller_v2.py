#!/usr/bin/env python3
"""
基于人形位置的智能灯光控制器 v3.5
集成多目标跟踪优化

改进:
- 人员ID保持，避免频繁切换导致的灯光闪烁
- 轨迹平滑，减少误判
- 多人场景下的智能灯光策略
- 拥堵检测与动态调整
"""
import time
import threading
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from corridor_light.light_zones import LightConfig, LightZone
from corridor_light.multi_object_tracker import (
    MultiObjectTracker, 
    MultiPersonLightStrategy,
    CongestionDetector,
    TrackedPerson
)


class ZoneLightControllerV2:
    """
    基于区域的智能灯光控制器 v2
    集成多目标跟踪和拥堵检测
    """
    
    def __init__(self, 
                 light_config: LightConfig,
                 light_off_delay: float = 0.5,
                 facing_direction: str = 'forward',
                 demo_mode: bool = True):
        """
        Args:
            light_config: 灯光区域配置
            light_off_delay: 人离开后延迟关灯时间(秒)
            facing_direction: 默认朝向
            demo_mode: True=仅打印状态，False=控制实际硬件
        """
        self.config = light_config
        self.light_off_delay = light_off_delay
        self.facing_direction = facing_direction
        self.demo_mode = demo_mode
        
        # 多目标跟踪器
        self.tracker = MultiObjectTracker(
            iou_threshold=0.3,
            max_miss_frames=10,
            min_track_frames=3
        )
        
        # 多人灯光策略
        self.light_strategy = MultiPersonLightStrategy(light_config)
        
        # 拥堵检测器
        self.congestion_detector = CongestionDetector(
            density_threshold=3,  # 3人/平方米视为拥堵
            area_threshold=20.0
        )
        
        # 灯状态管理
        self.light_states: Dict[str, Dict] = {}
        for zone_id in self.config.zones.keys():
            self.light_states[zone_id] = {
                'state': False,
                'last_person_time': 0,
                'last_zone': None,
                'reason': ''
            }
        
        self._lock = threading.Lock()
        self._gpio_available = False
        self._gpio_pins: Dict[str, int] = {}
        
        # 统计
        self.stats = {
            'zone_entries': {z: 0 for z in self.config.zones.keys()},
            'light_on_count': {z: 0 for z in self.config.zones.keys()},
            'light_on_time': {z: 0.0 for z in self.config.zones.keys()},
            'last_on_time': {z: None for z in self.config.zones.keys()},
            'total_people_tracked': 0,
            'congestion_events': 0
        }
    
    def init(self, gpio_mapping: Dict[str, int] = None) -> bool:
        """初始化控制器"""
        print("=" * 60)
        print("基于多目标跟踪的智能灯光控制器 v2")
        print("=" * 60)
        
        print(f"\n跟踪参数:")
        print(f"  IOU阈值: {self.tracker.iou_threshold}")
        print(f"  最大丢失帧数: {self.tracker.max_miss_frames}")
        print(f"  最小跟踪帧数: {self.tracker.min_track_frames}")
        
        print(f"\n灯光区域配置 ({len(self.config.zones)} 个区域):")
        for zone in self.config.get_all_zones():
            print(f"  [{zone.id}] {zone.name}: 位置({zone.x}, {zone.y}), "
                  f"半径{zone.radius}px")
        
        print(f"\n控制参数:")
        print(f"  关灯延迟: {self.light_off_delay}s")
        print(f"  默认朝向: {self.facing_direction}")
        print(f"  模式: {'Demo' if self.demo_mode else 'Deploy'}")
        
        # GPIO初始化
        if not self.demo_mode and gpio_mapping:
            self._gpio_pins = gpio_mapping
            try:
                import RPi.GPIO as GPIO
                GPIO.setmode(GPIO.BCM)
                for light_id, pin in gpio_mapping.items():
                    GPIO.setup(pin, GPIO.OUT)
                    GPIO.output(pin, GPIO.LOW)
                self._gpio_available = True
                print(f"\nGPIO初始化成功: {len(gpio_mapping)} 个灯")
            except ImportError:
                print("\n警告: RPi.GPIO未安装，切换到Demo模式")
                self.demo_mode = True
            except Exception as e:
                print(f"\nGPIO初始化失败: {e}，切换到Demo模式")
                self.demo_mode = True
        
        if self.demo_mode:
            print("\nDemo模式: 仅显示灯光状态，不控制硬件")
        
        return True
    
    def _find_zone_by_position(self, position: Tuple[int, int]) -> Optional[LightZone]:
        """根据位置查找所在区域"""
        for zone in self.config.get_all_zones():
            if zone.contains_point(position):
                return zone
        return None
    
    def update(self, detections: List[Dict]) -> Dict[str, bool]:
        """
        更新检测状态并控制灯光
        
        Args:
            detections: 检测结果列表
        
        Returns:
            各灯的当前状态
        """
        now = time.time()
        
        # 1. 多目标跟踪更新
        # 先确定每个人所在的区域
        current_zones = {}
        for det in detections:
            if det.get('class') == 'person':
                foot_point = det.get('foot_point')
                if foot_point:
                    zone = self._find_zone_by_position(foot_point)
                    if zone:
                        # 临时使用bbox作为track_id的占位
                        current_zones[det.get('id', id(det))] = zone.id
        
        # 更新跟踪器
        tracked_persons = self.tracker.update(detections, current_zones)
        
        # 2. 拥堵检测
        frame_shape = (480, 640)  # 假设标准分辨率
        congestion_info = self.congestion_detector.analyze(
            tracked_persons, 
            frame_shape[1], 
            frame_shape[0]
        )
        
        if congestion_info['is_congested']:
            self.stats['congestion_events'] += 1
        
        # 3. 使用多人策略决定灯光
        lights_decision = self.light_strategy.decide_lights(
            tracked_persons, 
            self
        )
        
        # 4. 应用灯光状态
        for light_id, should_on in lights_decision.items():
            self._set_light(light_id, should_on, now)
        
        # 5. 更新统计
        self.stats['total_people_tracked'] = self.tracker.total_tracked
        
        return lights_decision
    
    def _set_light(self, light_id: str, state: bool, now: float):
        """设置单个灯的状态"""
        with self._lock:
            if light_id not in self.light_states:
                return
            
            old_state = self.light_states[light_id]['state']
            
            # 检查延迟关灯逻辑
            if not state and old_state:
                time_since_last = now - self.light_states[light_id]['last_person_time']
                if time_since_last < self.light_off_delay:
                    return  # 保持开启
            
            if old_state == state:
                return
            
            self.light_states[light_id]['state'] = state
            
            # GPIO控制
            if self._gpio_available and light_id in self._gpio_pins:
                try:
                    import RPi.GPIO as GPIO
                    pin = self._gpio_pins[light_id]
                    GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)
                except Exception as e:
                    print(f"GPIO控制失败 [{light_id}]: {e}")
            
            # 打印状态
            zone = self.config.get_zone(light_id)
            name = zone.name if zone else light_id
            action = "开启" if state else "关闭"
            
            # 获取原因
            reason = ""
            if state:
                # 查找原因
                active_tracks = self.tracker.get_active_tracks()
                for track in active_tracks:
                    if track.is_in_zone(light_id):
                        reason = f"{track.track_id}在区域内"
                        break
            
            print(f"[灯光] {name} ({light_id}): {action} {reason}")
            
            # 统计
            if state:
                self.stats['light_on_count'][light_id] += 1
                self.stats['last_on_time'][light_id] = time.time()
            else:
                if self.stats['last_on_time'][light_id]:
                    duration = time.time() - self.stats['last_on_time'][light_id]
                    self.stats['light_on_time'][light_id] += duration
                    self.stats['last_on_time'][light_id] = None
    
    def get_tracking_visualization(self, frame) -> object:
        """获取带跟踪信息的可视化图像"""
        import cv2
        
        display = frame.copy()
        
        # 绘制所有跟踪对象
        for track in self.tracker.get_active_tracks():
            x1, y1, x2, y2 = track.bbox
            
            # 边界框颜色根据跟踪时长变化
            if track.frame_count < 5:
                color = (0, 165, 255)  # 橙色 - 新跟踪
            else:
                color = (0, 255, 0)    # 绿色 - 稳定跟踪
            
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            # 显示ID和方向
            label = f"ID:{track.track_id[-4:]} {track.direction}"
            cv2.putText(display, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 绘制轨迹
            if len(track.trajectory) > 1:
                points = list(track.trajectory)
                for i in range(1, len(points)):
                    cv2.line(display, points[i-1], points[i], (255, 255, 0), 1)
            
            # 绘制脚底点
            cv2.circle(display, track.foot_point, 5, (0, 0, 255), -1)
            
            # 绘制预测位置
            predicted = track.get_predicted_position()
            cv2.circle(display, predicted, 5, (255, 0, 255), 2)
            cv2.line(display, track.foot_point, predicted, (255, 0, 255), 1)
        
        # 显示统计信息
        stats = self.tracker.get_statistics()
        y_offset = 30
        cv2.putText(display, f"Active: {stats['currently_active']}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25
        cv2.putText(display, f"Total: {stats['total_tracked_all_time']}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return display
    
    def get_active_lights(self) -> List[str]:
        """获取当前开启的灯列表"""
        return [lid for lid, info in self.light_states.items() if info['state']]
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = self.stats.copy()
        stats.update(self.tracker.get_statistics())
        stats['congestion'] = {
            'current': self.congestion_detector.congestion_history[-1] 
                      if self.congestion_detector.congestion_history else None,
            'flow_rate': self.congestion_detector.get_flow_rate()
        }
        return stats
    
    def cleanup(self):
        """清理资源"""
        print("\n关闭所有灯光...")
        for light_id in self.light_states.keys():
            self._set_light(light_id, False, time.time())
        
        if self._gpio_available:
            try:
                import RPi.GPIO as GPIO
                GPIO.cleanup()
                print("GPIO清理完成")
            except:
                pass

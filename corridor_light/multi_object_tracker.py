#!/usr/bin/env python3
"""
多目标跟踪与优化模块 (Multi-Object Tracking & Optimization)

针对楼道多目标共存场景的优化:
1. 人员跟踪 (MOT) - 保持ID一致性
2. 轨迹平滑 - 减少抖动导致的误判
3. 去重策略 - 避免同一人在多个区域被重复计算
4. 多人场景灯光控制策略
5. 拥堵检测与疏散引导
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import uuid


@dataclass
class TrackedPerson:
    """被跟踪的人员"""
    track_id: str                    # 唯一跟踪ID
    bbox: List[int]                  # 当前边界框 [x1, y1, x2, y2]
    foot_point: Tuple[int, int]      # 脚底位置
    confidence: float                # 置信度
    
    # 跟踪状态
    first_seen: datetime             # 首次出现时间
    last_seen: datetime              # 最后出现时间
    frame_count: int = 0             # 跟踪帧数
    
    # 轨迹历史
    trajectory: deque = field(default_factory=lambda: deque(maxlen=30))  # 最近30帧轨迹
    zone_history: deque = field(default_factory=lambda: deque(maxlen=10))  # 区域历史
    
    # 状态标记
    is_active: bool = True           # 是否活跃
    miss_count: int = 0              # 连续丢失帧数
    
    # 行为分析
    velocity: Tuple[float, float] = (0.0, 0.0)  # 速度向量 (vx, vy)
    direction: str = "unknown"       # 移动方向: left/right/up/down/stationary
    activity: str = "walking"        # 活动状态: walking/running/standing
    
    def update(self, bbox: List[int], foot_point: Tuple[int, int], 
               confidence: float, current_zone: str = None):
        """更新跟踪信息"""
        # 计算速度
        if len(self.trajectory) > 0:
            last_pos = self.trajectory[-1]
            dt = 1.0  # 假设1帧
            vx = (foot_point[0] - last_pos[0]) / dt
            vy = (foot_point[1] - last_pos[1]) / dt
            self.velocity = (vx, vy)
            
            # 判断方向
            if abs(vx) > abs(vy):
                self.direction = "right" if vx > 0 else "left"
            else:
                self.direction = "down" if vy > 0 else "up"
            
            # 判断活动状态
            speed = (vx ** 2 + vy ** 2) ** 0.5
            if speed < 2:
                self.activity = "standing"
            elif speed < 10:
                self.activity = "walking"
            else:
                self.activity = "running"
        
        self.bbox = bbox
        self.foot_point = foot_point
        self.confidence = confidence
        self.last_seen = datetime.now()
        self.frame_count += 1
        self.miss_count = 0
        self.is_active = True
        
        # 记录轨迹
        self.trajectory.append(foot_point)
        if current_zone:
            self.zone_history.append(current_zone)
    
    def mark_missing(self):
        """标记为丢失"""
        self.miss_count += 1
        if self.miss_count > 10:  # 连续丢失10帧视为离开
            self.is_active = False
    
    def get_predicted_position(self) -> Tuple[int, int]:
        """预测下一帧位置 (基于速度)"""
        if len(self.trajectory) > 0:
            last_pos = self.trajectory[-1]
            predicted_x = int(last_pos[0] + self.velocity[0])
            predicted_y = int(last_pos[1] + self.velocity[1])
            return (predicted_x, predicted_y)
        return self.foot_point
    
    def is_in_zone(self, zone_id: str) -> bool:
        """检查是否在当前或曾经经过某区域"""
        if len(self.zone_history) > 0:
            return self.zone_history[-1] == zone_id
        return False


class MultiObjectTracker:
    """多目标跟踪器 (基于IOU的简易跟踪)"""
    
    def __init__(self, 
                 iou_threshold: float = 0.3,
                 max_miss_frames: int = 10,
                 min_track_frames: int = 3):
        """
        Args:
            iou_threshold: IOU阈值，用于匹配
            max_miss_frames: 最大允许丢失帧数
            min_track_frames: 最小跟踪帧数 (过滤短暂误检)
        """
        self.iou_threshold = iou_threshold
        self.max_miss_frames = max_miss_frames
        self.min_track_frames = min_track_frames
        
        # 跟踪对象存储
        self.tracks: Dict[str, TrackedPerson] = {}
        self.next_id = 0
        
        # 统计
        self.total_tracked = 0
        self.current_active = 0
        
        # 轨迹平滑滤波器
        self.use_kalman = False  # 可启用卡尔曼滤波
    
    def calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """计算两个边界框的IOU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算交集
        x1 = max(x1_1, x1_2)
        y1 = max(y1_1, y1_2)
        x2 = min(x2_1, x2_2)
        y2 = min(y2_1, y2_2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """计算两点距离"""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    
    def update(self, detections: List[Dict], current_zones: Dict[str, str] = None) -> List[TrackedPerson]:
        """
        更新跟踪器
        
        Args:
            detections: 当前帧检测结果列表
            current_zones: {track_id: zone_id} 当前所在区域
        
        Returns:
            跟踪对象列表
        """
        current_time = datetime.now()
        current_zones = current_zones or {}
        
        # 1. 预测现有跟踪对象的位置
        predicted_positions = {}
        for track_id, track in self.tracks.items():
            if track.is_active:
                predicted_positions[track_id] = track.get_predicted_position()
        
        # 2. 匹配检测与跟踪对象
        matched_tracks = set()
        matched_detections = set()
        
        # 计算IOU矩阵
        for det_idx, det in enumerate(detections):
            best_iou = self.iou_threshold
            best_track_id = None
            
            det_bbox = det['bbox']
            det_foot = det.get('foot_point', 
                             ((det_bbox[0] + det_bbox[2]) // 2, det_bbox[3]))
            
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks or not track.is_active:
                    continue
                
                # 计算IOU
                iou = self.calculate_iou(det_bbox, track.bbox)
                
                # 如果IOU低，检查预测位置距离
                if iou < self.iou_threshold:
                    predicted_pos = predicted_positions.get(track_id)
                    if predicted_pos:
                        distance = self.calculate_distance(det_foot, predicted_pos)
                        # 距离近也认为是同一人 (考虑快速移动)
                        if distance < 50:  # 50像素阈值
                            iou = max(iou, 0.2)
                
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id:
                # 更新匹配的跟踪对象
                zone_id = current_zones.get(best_track_id)
                self.tracks[best_track_id].update(
                    det_bbox, det_foot, det['confidence'], zone_id
                )
                matched_tracks.add(best_track_id)
                matched_detections.add(det_idx)
        
        # 3. 为未匹配的检测创建新跟踪对象
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_detections:
                det_bbox = det['bbox']
                det_foot = det.get('foot_point',
                                 ((det_bbox[0] + det_bbox[2]) // 2, det_bbox[3]))
                
                track_id = f"person_{self.next_id}"
                self.next_id += 1
                
                new_track = TrackedPerson(
                    track_id=track_id,
                    bbox=det_bbox,
                    foot_point=det_foot,
                    confidence=det['confidence'],
                    first_seen=current_time,
                    last_seen=current_time
                )
                
                zone_id = current_zones.get(track_id)
                if zone_id:
                    new_track.zone_history.append(zone_id)
                
                self.tracks[track_id] = new_track
                self.total_tracked += 1
        
        # 4. 标记未匹配的跟踪对象为丢失
        for track_id, track in self.tracks.items():
            if track_id not in matched_tracks and track.is_active:
                track.mark_missing()
        
        # 5. 清理长期不活跃的跟踪对象
        self._cleanup_tracks()
        
        # 6. 更新统计
        self.current_active = sum(1 for t in self.tracks.values() if t.is_active)
        
        # 返回活跃且满足最小帧数要求的跟踪对象
        return [t for t in self.tracks.values() 
                if t.is_active and t.frame_count >= self.min_track_frames]
    
    def _cleanup_tracks(self):
        """清理不活跃的跟踪对象"""
        to_remove = []
        for track_id, track in self.tracks.items():
            if not track.is_active and track.miss_count > self.max_miss_frames * 2:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
    
    def get_active_tracks(self) -> List[TrackedPerson]:
        """获取所有活跃跟踪对象"""
        return [t for t in self.tracks.values() 
                if t.is_active and t.frame_count >= self.min_track_frames]
    
    def get_tracks_by_zone(self, zone_id: str) -> List[TrackedPerson]:
        """获取特定区域内的跟踪对象"""
        return [t for t in self.get_active_tracks() if t.is_in_zone(zone_id)]
    
    def get_moving_towards(self, direction: str) -> List[TrackedPerson]:
        """获取向特定方向移动的人员"""
        return [t for t in self.get_active_tracks() if t.direction == direction]
    
    def get_statistics(self) -> Dict:
        """获取跟踪统计信息"""
        active = self.get_active_tracks()
        
        directions = defaultdict(int)
        activities = defaultdict(int)
        zones = defaultdict(int)
        
        for t in active:
            directions[t.direction] += 1
            activities[t.activity] += 1
            if len(t.zone_history) > 0:
                zones[t.zone_history[-1]] += 1
        
        return {
            'total_tracked_all_time': self.total_tracked,
            'currently_active': len(active),
            'directions': dict(directions),
            'activities': dict(activities),
            'zones': dict(zones),
            'avg_track_duration': np.mean([t.frame_count for t in active]) if active else 0
        }


class MultiPersonLightStrategy:
    """多人场景灯光控制策略"""
    
    def __init__(self, light_config):
        self.light_config = light_config
        self.zone_entry_time = {}  # track_id -> {zone_id: entry_time}
        self.zone_occupancy = defaultdict(set)  # zone_id -> set of track_ids
    
    def decide_lights(self, tracks: List[TrackedPerson], 
                     zone_controller) -> Dict[str, bool]:
        """
        多人场景下的灯光决策
        
        策略:
        1. 有人在区域 -> 开灯
        2. 人刚离开 (3秒内) -> 保持开灯 (避免频繁开关)
        3. 多人在同一区域 -> 正常开灯
        4. 人员流动方向预测 -> 提前开启前方灯
        """
        current_time = datetime.now()
        lights_decision = {}
        
        # 获取所有灯ID
        all_zones = zone_controller.config.get_all_zones()
        
        for zone in all_zones:
            should_on = False
            reason = ""
            
            # 检查当前是否有人在区域内
            people_in_zone = [t for t in tracks if t.is_in_zone(zone.id)]
            
            if people_in_zone:
                should_on = True
                reason = f"{len(people_in_zone)}人在区域内"
                
                # 记录进入时间
                for person in people_in_zone:
                    if person.track_id not in self.zone_entry_time:
                        self.zone_entry_time[person.track_id] = {}
                    if zone.id not in self.zone_entry_time[person.track_id]:
                        self.zone_entry_time[person.track_id][zone.id] = current_time
                
                self.zone_occupancy[zone.id].update(p.track_id for p in people_in_zone)
            
            # 检查是否有人刚离开 (保持3秒)
            elif zone.id in self.zone_occupancy:
                left_recently = False
                for track_id in list(self.zone_occupancy[zone.id]):
                    entry_time = self.zone_entry_time.get(track_id, {}).get(zone.id)
                    if entry_time:
                        time_since_entry = (current_time - entry_time).total_seconds()
                        # 如果这个人还在活跃跟踪中，但不在本区域，计算离开时间
                        track = next((t for t in tracks if t.track_id == track_id), None)
                        if track and track.is_active and not track.is_in_zone(zone.id):
                            # 人在活跃但不在这个区域，检查离开时间
                            last_in_zone = time_since_entry - track.frame_count * 0.1  # 估算
                            if last_in_zone < 3:  # 3秒内离开
                                left_recently = True
                                reason = "人刚离开，保持照明"
                                break
                
                if left_recently:
                    should_on = True
                else:
                    # 清理不在这个区域的人
                    self.zone_occupancy[zone.id].clear()
            
            # 检查是否有人正向这个区域移动 (预测)
            if not should_on:
                approaching = [t for t in tracks 
                              if t.is_active and 
                              self._is_approaching_zone(t, zone)]
                if approaching:
                    should_on = True
                    reason = f"{len(approaching)}人接近中"
            
            lights_decision[zone.id] = should_on
            
            # 添加前方灯
            if should_on and people_in_zone:
                # 根据人员移动方向开启前方灯
                for person in people_in_zone:
                    if person.direction in ['right', 'down']:
                        for forward_zone in zone.forward_zones:
                            lights_decision[forward_zone] = True
                    elif person.direction in ['left', 'up']:
                        for backward_zone in zone.backward_zones:
                            lights_decision[backward_zone] = True
        
        return lights_decision
    
    def _is_approaching_zone(self, track: TrackedPerson, zone, threshold_px: int = 100) -> bool:
        """检查人员是否正在接近某区域"""
        if len(track.trajectory) < 2:
            return False
        
        # 计算到区域中心的距离
        distance = ((track.foot_point[0] - zone.x) ** 2 + 
                   (track.foot_point[1] - zone.y) ** 2) ** 0.5
        
        # 如果在区域半径+阈值范围内，且向中心移动
        if distance < zone.radius + threshold_px:
            # 检查速度方向是否指向区域中心
            dx = zone.x - track.foot_point[0]
            dy = zone.y - track.foot_point[1]
            
            # 速度向量与指向区域中心的向量点积
            dot_product = track.velocity[0] * dx + track.velocity[1] * dy
            
            # 点积为正表示方向大致相同
            return dot_product > 0
        
        return False


class CongestionDetector:
    """拥堵检测器"""
    
    def __init__(self, density_threshold: int = 5, 
                 area_threshold: float = 10.0):
        """
        Args:
            density_threshold: 拥堵密度阈值 (人/平方米)
            area_threshold: 检测区域面积阈值 (平方米)
        """
        self.density_threshold = density_threshold
        self.area_threshold = area_threshold
        self.congestion_history = deque(maxlen=100)
    
    def analyze(self, tracks: List[TrackedPerson], 
                zone_width_px: int, zone_height_px: int,
                pixels_per_meter: float = 50) -> Dict:
        """
        分析拥堵情况
        
        Returns:
            {
                'is_congested': 是否拥堵,
                'density': 当前密度 (人/m²),
                'people_count': 人数,
                'recommendation': 建议操作
            }
        """
        active_tracks = [t for t in tracks if t.is_active]
        people_count = len(active_tracks)
        
        # 计算区域面积 (平方米)
        width_m = zone_width_px / pixels_per_meter
        height_m = zone_height_px / pixels_per_meter
        area = width_m * height_m
        
        # 计算密度
        density = people_count / area if area > 0 else 0
        
        # 判断是否拥堵
        is_congested = density > self.density_threshold
        
        # 生成建议
        recommendation = "正常"
        if is_congested:
            if density > self.density_threshold * 2:
                recommendation = "严重拥堵，建议分流"
            else:
                recommendation = "轻度拥堵，注意疏导"
        elif people_count == 0:
            recommendation = "无人，可关闭所有灯光"
        
        result = {
            'is_congested': is_congested,
            'density': round(density, 2),
            'people_count': people_count,
            'area_m2': round(area, 2),
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat()
        }
        
        self.congestion_history.append(result)
        return result
    
    def get_flow_rate(self, minutes: int = 5) -> Dict:
        """获取人流速率统计"""
        if len(self.congestion_history) < 2:
            return {'inflow': 0, 'outflow': 0, 'net_flow': 0}
        
        # 简化计算：基于历史密度变化估算
        recent = list(self.congestion_history)[-10:]
        people_changes = [r['people_count'] for r in recent]
        
        if len(people_changes) >= 2:
            net_change = people_changes[-1] - people_changes[0]
            return {
                'net_flow': net_change,
                'avg_people': sum(people_changes) / len(people_changes),
                'peak_people': max(people_changes)
            }
        
        return {'inflow': 0, 'outflow': 0, 'net_flow': 0}

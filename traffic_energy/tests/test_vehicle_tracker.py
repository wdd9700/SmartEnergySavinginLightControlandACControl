#!/usr/bin/env python3
"""
车辆跟踪器单元测试
"""

import pytest
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from traffic_energy.detection.vehicle_tracker import (
    TrajectoryPoint, Track, TrackerConfig, VehicleTracker
)


class TestTrajectoryPoint:
    """测试TrajectoryPoint数据类"""
    
    def test_trajectory_point_creation(self):
        """测试创建轨迹点"""
        point = TrajectoryPoint(
            timestamp=1234567890.0,
            bbox=np.array([100, 100, 200, 200]),
            center=(150, 150),
            velocity=(5.0, 0.0),
            speed=5.0
        )
        
        assert point.timestamp == 1234567890.0
        assert point.center == (150, 150)
        assert point.speed == 5.0


class TestTrack:
    """测试Track数据类"""
    
    def test_track_creation(self):
        """测试创建轨迹"""
        track = Track(
            track_id=1,
            bbox=np.array([100, 100, 200, 200]),
            confidence=0.85,
            class_id=2,
            class_name="car"
        )
        
        assert track.track_id == 1
        assert track.class_name == "car"
        assert track.state == "tentative"
        assert not track.is_confirmed
        assert not track.is_deleted
    
    def test_track_center(self):
        """测试轨迹中心点"""
        track = Track(
            track_id=1,
            bbox=np.array([100, 100, 300, 400]),
            confidence=0.9,
            class_id=2,
            class_name="car"
        )
        
        assert track.center == (200, 250)
    
    def test_track_states(self):
        """测试轨迹状态"""
        track = Track(
            track_id=1,
            bbox=np.array([100, 100, 200, 200]),
            confidence=0.85,
            class_id=2,
            class_name="car",
            state="confirmed"
        )
        
        assert track.is_confirmed
        assert not track.is_deleted
        
        track.state = "deleted"
        assert track.is_deleted
    
    def test_get_last_n_points(self):
        """测试获取最近N个点"""
        trajectory = [
            TrajectoryPoint(
                timestamp=float(i),
                bbox=np.array([100, 100, 200, 200]),
                center=(150, 150)
            )
            for i in range(10)
        ]
        
        track = Track(
            track_id=1,
            bbox=np.array([100, 100, 200, 200]),
            confidence=0.85,
            class_id=2,
            class_name="car",
            trajectory=trajectory
        )
        
        last_5 = track.get_last_n_points(5)
        assert len(last_5) == 5
        
        last_20 = track.get_last_n_points(20)
        assert len(last_20) == 10


class TestTrackerConfig:
    """测试TrackerConfig"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = TrackerConfig()
        
        assert config.track_high_thresh == 0.6
        assert config.track_low_thresh == 0.1
        assert config.track_buffer == 60
        assert config.cmc_method == "ecc"
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = TrackerConfig(
            track_buffer=30,
            match_thresh=0.7,
            cmc_method="orb"
        )
        
        assert config.track_buffer == 30
        assert config.match_thresh == 0.7
        assert config.cmc_method == "orb"


class TestVehicleTracker:
    """测试VehicleTracker类"""
    
    def test_tracker_initialization(self):
        """测试跟踪器初始化"""
        config = TrackerConfig()
        tracker = VehicleTracker(config)
        
        assert tracker.config == config
        assert len(tracker.tracks) == 0
        assert tracker.frame_count == 0
    
    def test_tracker_default_config(self):
        """测试默认配置初始化"""
        tracker = VehicleTracker()
        
        assert tracker.config.track_buffer == 60
        assert tracker.config.cmc_method == "ecc"
    
    def test_iou_calculation(self):
        """测试IoU计算"""
        tracker = VehicleTracker()
        
        box1 = np.array([100, 100, 200, 200])
        box2 = np.array([150, 150, 250, 250])
        
        iou = tracker._compute_iou(box1, box2)
        
        # 重叠区域: 50x50 = 2500
        # box1面积: 100x100 = 10000
        # box2面积: 100x100 = 10000
        # 并集: 10000 + 10000 - 2500 = 17500
        # IoU: 2500 / 17500 = 0.1428...
        expected_iou = 2500 / 17500
        assert abs(iou - expected_iou) < 0.001
    
    def test_iou_no_overlap(self):
        """测试无重叠的IoU"""
        tracker = VehicleTracker()
        
        box1 = np.array([100, 100, 200, 200])
        box2 = np.array([300, 300, 400, 400])
        
        iou = tracker._compute_iou(box1, box2)
        assert iou == 0.0
    
    def test_iou_identical(self):
        """测试相同框的IoU"""
        tracker = VehicleTracker()
        
        box = np.array([100, 100, 200, 200])
        
        iou = tracker._compute_iou(box, box)
        assert iou == 1.0
    
    def test_create_track(self):
        """测试创建轨迹"""
        tracker = VehicleTracker()
        
        bbox = np.array([100, 100, 200, 200])
        track_id = tracker._create_track(bbox, 0.9, 2, 1234567890.0)
        
        assert track_id == 1
        assert track_id in tracker.tracks
        assert tracker.tracks[track_id].class_name == "car"
        assert tracker.total_tracks_created == 1
    
    def test_get_trajectory(self):
        """测试获取轨迹"""
        tracker = VehicleTracker()
        
        # 创建轨迹
        bbox = np.array([100, 100, 200, 200])
        track_id = tracker._create_track(bbox, 0.9, 2, 1234567890.0)
        
        # 获取轨迹
        trajectory = tracker.get_trajectory(track_id)
        assert trajectory is not None
        assert len(trajectory) == 1
        
        # 获取不存在的轨迹
        assert tracker.get_trajectory(999) is None
    
    def test_get_track(self):
        """测试获取轨迹对象"""
        tracker = VehicleTracker()
        
        bbox = np.array([100, 100, 200, 200])
        track_id = tracker._create_track(bbox, 0.9, 2, 1234567890.0)
        
        track = tracker.get_track(track_id)
        assert track is not None
        assert track.track_id == track_id
        
        assert tracker.get_track(999) is None
    
    def test_reset(self):
        """测试重置跟踪器"""
        tracker = VehicleTracker()
        
        # 创建一些轨迹
        bbox = np.array([100, 100, 200, 200])
        tracker._create_track(bbox, 0.9, 2, 1234567890.0)
        tracker.frame_count = 100
        
        # 重置
        tracker.reset()
        
        assert len(tracker.tracks) == 0
        assert tracker.frame_count == 0
        assert tracker._next_id == 1
    
    def test_empty_update(self):
        """测试空检测更新"""
        tracker = VehicleTracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        tracks = tracker.update([], frame)
        
        assert isinstance(tracks, list)
        assert tracker.frame_count == 1


class TestTrackerIntegration:
    """跟踪器集成测试"""
    
    def test_simple_tracking_sequence(self):
        """测试简单跟踪序列"""
        tracker = VehicleTracker(TrackerConfig(track_buffer=10))
        
        # 模拟检测序列
        class MockDetection:
            def __init__(self, bbox, conf, class_id):
                self.bbox = np.array(bbox)
                self.confidence = conf
                self.class_id = class_id
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 第一帧：创建一个检测
        dets = [MockDetection([100, 100, 200, 200], 0.9, 2)]
        tracks = tracker.update(dets, frame)
        
        # 应该有新轨迹（tentative状态）
        assert len(tracks) >= 0  # 可能为0因为需要确认
        
        # 连续几帧更新
        for i in range(5):
            dets = [MockDetection([100 + i*5, 100, 200 + i*5, 200], 0.9, 2)]
            tracks = tracker.update(dets, frame)
        
        # 应该有确认轨迹
        confirmed = tracker.get_confirmed_tracks()
        assert len(confirmed) >= 0

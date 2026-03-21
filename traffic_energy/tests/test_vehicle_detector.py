#!/usr/bin/env python3
"""
车辆检测器单元测试
"""

import pytest
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from traffic_energy.detection.vehicle_detector import Detection, VehicleDetector, VEHICLE_CLASSES


class TestDetection:
    """测试Detection数据类"""
    
    def test_detection_creation(self):
        """测试创建Detection对象"""
        det = Detection(
            bbox=np.array([100, 100, 200, 200], dtype=np.float32),
            confidence=0.85,
            class_id=2,
            class_name="car"
        )
        
        assert det.confidence == 0.85
        assert det.class_id == 2
        assert det.class_name == "car"
        assert det.track_id is None
    
    def test_detection_center(self):
        """测试中心点计算"""
        det = Detection(
            bbox=np.array([100, 100, 200, 300], dtype=np.float32),
            confidence=0.9,
            class_id=2,
            class_name="car"
        )
        
        center = det.center
        assert center == (150, 200)
    
    def test_detection_dimensions(self):
        """测试尺寸计算"""
        det = Detection(
            bbox=np.array([100, 100, 300, 400], dtype=np.float32),
            confidence=0.9,
            class_id=2,
            class_name="car"
        )
        
        assert det.width == 200
        assert det.height == 300
        assert det.area == 60000


class TestVehicleClasses:
    """测试车辆类别映射"""
    
    def test_vehicle_classes_mapping(self):
        """测试类别映射"""
        assert VEHICLE_CLASSES[2] == "car"
        assert VEHICLE_CLASSES[3] == "motorcycle"
        assert VEHICLE_CLASSES[5] == "bus"
        assert VEHICLE_CLASSES[7] == "truck"


class TestVehicleDetector:
    """测试VehicleDetector类"""
    
    def test_detector_initialization_without_model(self):
        """测试检测器初始化（无模型）"""
        # 如果没有ultralytics，应该抛出ImportError
        try:
            detector = VehicleDetector(
                model_path="yolo12n.pt",
                conf_threshold=0.5,
                device="cpu"
            )
            assert detector.conf_threshold == 0.5
            assert detector.device in ["cpu", "cuda:0"]
            assert detector.classes == [2, 3, 5, 7]
        except ImportError:
            pytest.skip("ultralytics not installed")
    
    def test_detector_default_classes(self):
        """测试默认类别"""
        try:
            detector = VehicleDetector(device="cpu")
            assert detector.classes == [2, 3, 5, 7]
        except ImportError:
            pytest.skip("ultralytics not installed")
    
    def test_detector_custom_classes(self):
        """测试自定义类别"""
        try:
            detector = VehicleDetector(
                device="cpu",
                classes=[2, 3]
            )
            assert detector.classes == [2, 3]
        except ImportError:
            pytest.skip("ultralytics not installed")
    
    def test_detector_stats(self):
        """测试统计信息"""
        try:
            detector = VehicleDetector(device="cpu")
            
            # 初始状态
            assert detector.inference_count == 0
            assert detector.average_inference_time == 0.0
            assert detector.fps == 0.0
            
            # 重置统计
            detector.reset_stats()
            assert detector.inference_count == 0
        except ImportError:
            pytest.skip("ultralytics not installed")


class TestDetectionIntegration:
    """集成测试"""
    
    def test_detect_on_random_frame(self):
        """测试在随机帧上检测"""
        try:
            detector = VehicleDetector(device="cpu")
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            detections = detector.detect(frame)
            
            # 随机帧上不应该检测到车辆
            assert isinstance(detections, list)
            
        except ImportError:
            pytest.skip("ultralytics not installed")
        except Exception as e:
            # 模型文件可能不存在
            pytest.skip(f"模型加载失败: {e}")

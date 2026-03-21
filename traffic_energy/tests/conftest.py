#!/usr/bin/env python3
"""
测试配置和 fixtures
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_frame():
    """生成测试用帧"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_detection():
    """生成测试用检测框"""
    from traffic_energy.detection.vehicle_detector import Detection
    
    return Detection(
        bbox=np.array([100, 100, 200, 200], dtype=np.float32),
        confidence=0.85,
        class_id=2,
        class_name="car"
    )


@pytest.fixture
def tracker_config():
    """跟踪器配置"""
    from traffic_energy.detection.vehicle_tracker import TrackerConfig
    
    return TrackerConfig(
        track_buffer=30,
        match_thresh=0.7
    )


@pytest.fixture
def temp_config_file(tmp_path):
    """临时配置文件"""
    config_content = """
system:
  name: "Test System"
  version: "1.0.0"
  log_level: "DEBUG"

detection:
  model:
    name: "yolo12n.pt"
    conf_threshold: 0.5
    device: "cpu"
  tracker:
    type: "botsort"
    track_buffer: 60
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(config_content, encoding='utf-8')
    return str(config_path)

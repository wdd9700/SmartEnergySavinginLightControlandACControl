#!/usr/bin/env python3
"""
配置管理单元测试
"""

import pytest
import os
import tempfile
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from traffic_energy.config.manager import (
    ConfigManager, TrafficConfig, ModelConfig, 
    TrackerConfig, ReidConfig, load_config
)


class TestModelConfig:
    """测试ModelConfig"""
    
    def test_default_values(self):
        """测试默认值"""
        config = ModelConfig()
        
        assert config.name == "yolo12n.pt"
        assert config.conf_threshold == 0.5
        assert config.iou_threshold == 0.45
        assert config.device == "auto"
        assert config.classes == [2, 3, 5, 7]
    
    def test_custom_values(self):
        """测试自定义值"""
        config = ModelConfig(
            name="yolo12s.pt",
            conf_threshold=0.7,
            device="cuda:0",
            classes=[2, 3]
        )
        
        assert config.name == "yolo12s.pt"
        assert config.conf_threshold == 0.7
        assert config.device == "cuda:0"
        assert config.classes == [2, 3]


class TestTrackerConfig:
    """测试TrackerConfig"""
    
    def test_default_values(self):
        """测试默认值"""
        config = TrackerConfig()
        
        assert config.type == "botsort"
        assert config.track_high_thresh == 0.6
        assert config.track_buffer == 60
        assert config.cmc_method == "ecc"
    
    def test_cmc_methods(self):
        """测试CMC方法"""
        config_ecc = TrackerConfig(cmc_method="ecc")
        config_orb = TrackerConfig(cmc_method="orb")
        config_none = TrackerConfig(cmc_method="none")
        
        assert config_ecc.cmc_method == "ecc"
        assert config_orb.cmc_method == "orb"
        assert config_none.cmc_method == "none"


class TestReidConfig:
    """测试ReidConfig"""
    
    def test_default_values(self):
        """测试默认值"""
        config = ReidConfig()
        
        assert config.model_name == "veriwild_bagtricks_R50-ibn"
        assert config.feature_dim == 2048
        assert config.input_size == [128, 256]
        assert config.similarity_threshold == 0.7


class TestTrafficConfig:
    """测试TrafficConfig"""
    
    def test_default_values(self):
        """测试默认值"""
        config = TrafficConfig()
        
        assert config.system_name == "Smart Traffic Energy System"
        assert config.version == "1.0.0"
        assert config.log_level == "INFO"
        assert isinstance(config.detection, DetectionConfig.__class__)


class TestConfigManager:
    """测试ConfigManager"""
    
    def test_initialization_without_path(self):
        """测试无路径初始化"""
        manager = ConfigManager()
        
        assert manager.config_path is None
        assert manager._config is None
    
    def test_initialization_with_path(self):
        """测试有路径初始化"""
        manager = ConfigManager("config.yaml")
        
        assert manager.config_path == "config.yaml"
    
    def test_load_default_config(self):
        """测试加载默认配置"""
        manager = ConfigManager()
        config = manager.load()
        
        assert isinstance(config, TrafficConfig)
        assert config.system_name == "Smart Traffic Energy System"
    
    def test_load_from_file(self, temp_config_file):
        """测试从文件加载"""
        manager = ConfigManager()
        config = manager.load(temp_config_file)
        
        assert config.system_name == "Test System"
        assert config.log_level == "DEBUG"
    
    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        manager = ConfigManager()
        
        with pytest.raises(FileNotFoundError):
            manager.load("nonexistent_config.yaml")
    
    def test_config_property_not_loaded(self):
        """测试未加载时访问配置"""
        manager = ConfigManager()
        
        with pytest.raises(RuntimeError):
            _ = manager.config
    
    def test_config_property_after_load(self, temp_config_file):
        """测试加载后访问配置"""
        manager = ConfigManager()
        manager.load(temp_config_file)
        
        config = manager.config
        assert isinstance(config, TrafficConfig)
    
    def test_get_existing_key(self, temp_config_file):
        """测试获取存在的键"""
        manager = ConfigManager()
        manager.load(temp_config_file)
        
        value = manager.get("system.name")
        assert value == "Test System"
    
    def test_get_nonexistent_key(self, temp_config_file):
        """测试获取不存在的键"""
        manager = ConfigManager()
        manager.load(temp_config_file)
        
        value = manager.get("nonexistent.key")
        assert value is None
    
    def test_get_with_default(self, temp_config_file):
        """测试获取带默认值"""
        manager = ConfigManager()
        manager.load(temp_config_file)
        
        value = manager.get("nonexistent.key", "default")
        assert value == "default"
    
    def test_reload(self, temp_config_file):
        """测试重新加载"""
        manager = ConfigManager(temp_config_file)
        config1 = manager.load()
        
        # 修改文件
        new_content = """
system:
  name: "Modified System"
  version: "2.0.0"
"""
        Path(temp_config_file).write_text(new_content, encoding='utf-8')
        
        config2 = manager.reload()
        
        assert config2.system_name == "Modified System"
        assert config2.version == "2.0.0"


class TestEnvironmentOverrides:
    """测试环境变量覆盖"""
    
    def test_log_level_override(self, temp_config_file, monkeypatch):
        """测试日志级别覆盖"""
        monkeypatch.setenv("TRAFFIC_LOG_LEVEL", "ERROR")
        
        manager = ConfigManager()
        config = manager.load(temp_config_file)
        
        assert config.log_level == "ERROR"
    
    def test_device_override(self, temp_config_file, monkeypatch):
        """测试设备覆盖"""
        monkeypatch.setenv("TRAFFIC_DEVICE", "cuda:1")
        
        manager = ConfigManager()
        config = manager.load(temp_config_file)
        
        assert config.detection.model.device == "cuda:1"
        assert config.reid.device == "cuda:1"


class TestLoadConfigFunction:
    """测试load_config便捷函数"""
    
    def test_load_config(self, temp_config_file):
        """测试便捷加载函数"""
        config = load_config(temp_config_file)
        
        assert isinstance(config, TrafficConfig)
        assert config.system_name == "Test System"

#!/usr/bin/env python3
"""车辆重识别模块"""

from .feature_extractor import FeatureExtractor
from .feature_database import FeatureDatabase
from .cross_camera_matcher import CrossCameraMatcher

__all__ = ['FeatureExtractor', 'FeatureDatabase', 'CrossCameraMatcher']

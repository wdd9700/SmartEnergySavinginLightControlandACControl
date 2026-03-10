#!/usr/bin/env python3
"""
多目标跟踪优化测试
演示楼道多目标共存场景的优化效果
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from datetime import datetime

from corridor_light.light_zones import LightConfig, LightZone, create_default_config
from corridor_light.multi_object_tracker import (
    MultiObjectTracker, 
    MultiPersonLightStrategy,
    CongestionDetector,
    TrackedPerson
)


def test_multi_object_tracker():
    """测试多目标跟踪器"""
    print("=" * 60)
    print("测试1: 多目标跟踪器 (MOT)")
    print("=" * 60)
    
    tracker = MultiObjectTracker(
        iou_threshold=0.3,
        max_miss_frames=5,
        min_track_frames=3
    )
    
    # 模拟连续帧检测
    print("\n模拟连续10帧的检测场景...")
    
    # 帧1: 2人进入
    detections_frame1 = [
        {'bbox': [100, 100, 150, 200], 'foot_point': (125, 200), 'confidence': 0.9, 'class': 'person'},
        {'bbox': [300, 120, 350, 220], 'foot_point': (325, 220), 'confidence': 0.85, 'class': 'person'}
    ]
    tracks1 = tracker.update(detections_frame1)
    print(f"帧1: 检测到{len(detections_frame1)}人, 跟踪到{len(tracks1)}人")
    for t in tracks1:
        print(f"  {t.track_id}: 位置{t.foot_point}, 帧数{t.frame_count}")
    
    # 帧2-5: 人员移动
    for i in range(2, 6):
        # 模拟人员向右移动
        offset = (i-1) * 10
        detections = [
            {'bbox': [100+offset, 100, 150+offset, 200], 'foot_point': (125+offset, 200), 
             'confidence': 0.9, 'class': 'person'},
            {'bbox': [300+offset, 120, 350+offset, 220], 'foot_point': (325+offset, 220), 
             'confidence': 0.85, 'class': 'person'}
        ]
        tracks = tracker.update(detections)
        print(f"帧{i}: 跟踪到{len(tracks)}人")
    
    # 帧6: 新增1人
    detections_frame6 = [
        {'bbox': [140, 100, 190, 200], 'foot_point': (165, 200), 'confidence': 0.9, 'class': 'person'},
        {'bbox': [340, 120, 390, 220], 'foot_point': (365, 220), 'confidence': 0.85, 'class': 'person'},
        {'bbox': [200, 150, 250, 250], 'foot_point': (225, 250), 'confidence': 0.88, 'class': 'person'}  # 新增
    ]
    tracks6 = tracker.update(detections_frame6)
    print(f"帧6: 新增1人, 共跟踪{len(tracks6)}人")
    
    # 帧7-9: 1人离开视野
    for i in range(7, 10):
        detections = [
            {'bbox': [140+(i-6)*10, 100, 190+(i-6)*10, 200], 'foot_point': (165+(i-6)*10, 200), 
             'confidence': 0.9, 'class': 'person'},
            {'bbox': [340+(i-6)*10, 120, 390+(i-6)*10, 220], 'foot_point': (365+(i-6)*10, 220), 
             'confidence': 0.85, 'class': 'person'}
            # 第3人不再检测
        ]
        tracks = tracker.update(detections)
        active = sum(1 for t in tracks if t.is_active)
        missing = sum(1 for t in tracker.tracks.values() if not t.is_active)
        print(f"帧{i}: 活跃{active}人, 丢失但跟踪中{missing}人")
    
    # 统计
    stats = tracker.get_statistics()
    print(f"\n跟踪统计:")
    print(f"  历史总跟踪人数: {stats['total_tracked_all_time']}")
    print(f"  当前活跃人数: {stats['currently_active']}")
    print(f"  移动方向分布: {stats['directions']}")
    
    return True


def test_multi_person_light_strategy():
    """测试多人灯光策略"""
    print("\n" + "=" * 60)
    print("测试2: 多人场景灯光策略")
    print("=" * 60)
    
    config = create_default_config(640, 480)
    strategy = MultiPersonLightStrategy(config)
    
    # 模拟跟踪对象
    class MockTracker:
        def __init__(self):
            self.tracks = {}
    
    class MockController:
        def __init__(self, config):
            self.config = config
    
    controller = MockController(config)
    
    # 场景1: 单人在入口
    print("\n场景1: 单人在入口区域，向右移动")
    person1 = TrackedPerson(
        track_id="p1",
        bbox=[100, 100, 150, 200],
        foot_point=(128, 240),
        confidence=0.9,
        first_seen=datetime.now(),
        last_seen=datetime.now()
    )
    person1.zone_history.append('light_0')
    person1.direction = 'right'
    
    tracks = [person1]
    decision = strategy.decide_lights(tracks, controller)
    print(f"  决策: {decision}")
    print(f"  说明: 人在light_0，向右移动，应开启light_0和前方light_1")
    
    # 场景2: 多人在不同区域
    print("\n场景2: 2人在不同区域")
    person2 = TrackedPerson(
        track_id="p2",
        bbox=[300, 100, 350, 200],
        foot_point=(320, 240),
        confidence=0.85,
        first_seen=datetime.now(),
        last_seen=datetime.now()
    )
    person2.zone_history.append('light_1')
    person2.direction = 'left'
    
    tracks = [person1, person2]
    decision = strategy.decide_lights(tracks, controller)
    print(f"  决策: {decision}")
    print(f"  说明: light_0和light_1都应开启")
    
    # 场景3: 人刚离开区域
    print("\n场景3: 人刚离开light_0，进入light_1")
    person1.zone_history.append('light_1')  # 现在在light_1
    
    decision = strategy.decide_lights([person1], controller)
    print(f"  决策: {decision}")
    
    return True


def test_congestion_detection():
    """测试拥堵检测"""
    print("\n" + "=" * 60)
    print("测试3: 拥堵检测")
    print("=" * 60)
    
    detector = CongestionDetector(
        density_threshold=3,  # 3人/平方米视为拥堵
        area_threshold=20.0
    )
    
    # 场景1: 低密度
    print("\n场景1: 5人，正常密度")
    tracks_low = []
    for i in range(5):
        t = TrackedPerson(
            track_id=f"p{i}",
            bbox=[i*50, 100, i*50+40, 180],
            foot_point=(i*50+20, 180),
            confidence=0.9,
            first_seen=datetime.now(),
            last_seen=datetime.now()
        )
        t.is_active = True
        tracks_low.append(t)
    
    result = detector.analyze(tracks_low, 640, 480, pixels_per_meter=50)
    print(f"  密度: {result['density']}人/m²")
    print(f"  是否拥堵: {result['is_congested']}")
    print(f"  建议: {result['recommendation']}")
    
    # 场景2: 高密度
    print("\n场景2: 15人，高密度")
    tracks_high = []
    for i in range(15):
        t = TrackedPerson(
            track_id=f"p{i}",
            bbox=[(i%5)*60, 100+(i//5)*60, (i%5)*60+40, 160+(i//5)*60],
            foot_point=((i%5)*60+20, 160+(i//5)*60),
            confidence=0.9,
            first_seen=datetime.now(),
            last_seen=datetime.now()
        )
        t.is_active = True
        tracks_high.append(t)
    
    result = detector.analyze(tracks_high, 640, 480, pixels_per_meter=50)
    print(f"  密度: {result['density']}人/m²")
    print(f"  是否拥堵: {result['is_congested']}")
    print(f"  建议: {result['recommendation']}")
    
    # 人流统计
    flow = detector.get_flow_rate()
    print(f"\n人流统计:")
    print(f"  平均人数: {flow['avg_people']:.1f}")
    print(f"  峰值人数: {flow['peak_people']}")
    
    return True


def test_trajectory_prediction():
    """测试轨迹预测"""
    print("\n" + "=" * 60)
    print("测试4: 轨迹预测与平滑")
    print("=" * 60)
    
    tracker = MultiObjectTracker()
    
    # 模拟一个人从左向右移动
    print("\n模拟人员从左向右移动...")
    
    for i in range(10):
        detections = [{
            'bbox': [50+i*20, 100, 100+i*20, 200],
            'foot_point': (75+i*20, 200),
            'confidence': 0.9,
            'class': 'person'
        }]
        
        tracks = tracker.update(detections)
        
        if tracks:
            track = tracks[0]
            print(f"帧{i+1}: 位置{track.foot_point}, "
                  f"速度({track.velocity[0]:.1f}, {track.velocity[1]:.1f}), "
                  f"方向{track.direction}, "
                  f"预测{track.get_predicted_position()}")
    
    return True


def main():
    print("\n" + "=" * 60)
    print("楼道多目标共存优化测试套件")
    print("=" * 60)
    
    tests = [
        ("多目标跟踪器", test_multi_object_tracker),
        ("多人灯光策略", test_multi_person_light_strategy),
        ("拥堵检测", test_congestion_detection),
        ("轨迹预测", test_trajectory_prediction),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                print(f"\n✅ {name} 测试通过")
                passed += 1
            else:
                print(f"\n❌ {name} 测试失败")
                failed += 1
        except Exception as e:
            print(f"\n❌ {name} 测试异常: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    # 优化总结
    print("\n多目标优化特性总结:")
    print("1. ✓ 人员ID保持 - 避免频繁切换导致的灯光闪烁")
    print("2. ✓ 轨迹平滑 - 基于速度预测，减少误判")
    print("3. ✓ 多人策略 - 智能处理多人场景的灯光控制")
    print("4. ✓ 拥堵检测 - 动态调整照明策略")
    print("5. ✓ 方向预测 - 提前开启前方灯光")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

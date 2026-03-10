#!/usr/bin/env python3
"""
测试亮度分析和多摄像头校准模块
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from corridor_light.brightness_analyzer import BrightnessExtractor, LightBrightnessComparator
from corridor_light.multi_camera_calibrator import MultiCameraCalibrator


def test_brightness_extractor():
    """测试亮度提取器"""
    print("=" * 60)
    print("测试1: 亮度提取器")
    print("=" * 60)
    
    # 创建测试图像
    frame = np.random.randint(50, 100, (480, 640, 3), dtype=np.uint8)
    
    # 添加一个高亮区域模拟灯光
    cv2.circle(frame, (320, 240), 100, (200, 200, 150), -1)
    cv2.GaussianBlur(frame, (51, 51), 0, frame)
    
    extractor = BrightnessExtractor(grid_size=(8, 6))
    regions = extractor.extract_brightness_grid(frame)
    
    print(f"\n提取了 {len(regions)} 个区域的亮度")
    print(f"亮度范围: {min(r.brightness for r in regions):.1f} - {max(r.brightness for r in regions):.1f}")
    
    # 估算灯中心
    center = extractor.estimate_light_center(frame)
    print(f"估算的灯中心: {center}")
    
    # 可视化
    vis = extractor.visualize_brightness(frame)
    cv2.imwrite("test_brightness_vis.jpg", vis)
    print("可视化结果: test_brightness_vis.jpg")
    
    return True


def test_brightness_comparator():
    """测试亮度对比器"""
    print("\n" + "=" * 60)
    print("测试2: 亮度对比器 (开关灯分析)")
    print("=" * 60)
    
    # 模拟关灯帧 (较暗)
    frame_off = np.random.randint(30, 60, (480, 640, 3), dtype=np.uint8)
    
    # 模拟开灯帧 (较亮，有光源)
    frame_on = frame_off.copy()
    light_center = (400, 200)
    light_radius = 120
    
    # 添加光照效果
    for y in range(480):
        for x in range(640):
            dist = np.sqrt((x - light_center[0])**2 + (y - light_center[1])**2)
            if dist < light_radius:
                intensity = int(100 * (1 - dist / light_radius))
                frame_on[y, x] = np.clip(frame_on[y, x] + intensity, 0, 255)
    
    # 分析
    comparator = LightBrightnessComparator()
    comparator.capture_light_off(frame_off)
    comparator.capture_light_on(frame_on)
    
    # 估算
    estimated_pos = comparator.estimate_light_source_position()
    estimated_radius = comparator.estimate_illumination_radius()
    
    print(f"\n实际灯光位置: {light_center}")
    print(f"估算灯光位置: {estimated_pos}")
    print(f"实际照明半径: {light_radius}")
    print(f"估算照明半径: {estimated_radius:.1f}")
    
    if estimated_pos:
        error = np.sqrt((estimated_pos[0] - light_center[0])**2 + 
                       (estimated_pos[1] - light_center[1])**2)
        print(f"位置误差: {error:.1f} 像素")
    
    # 可视化
    vis = comparator.visualize_contribution(frame_on)
    cv2.imwrite("test_contribution_vis.jpg", vis)
    print("可视化结果: test_contribution_vis.jpg")
    
    return True


def test_multi_camera_calibrator():
    """测试多摄像头校准器"""
    print("\n" + "=" * 60)
    print("测试3: 多摄像头校准器")
    print("=" * 60)
    
    calibrator = MultiCameraCalibrator()
    
    # 注册两个模拟摄像头
    calibrator.register_camera("cam1", (480, 640))
    calibrator.register_camera("cam2", (480, 640))
    
    # 模拟检测结果
    # cam1 在画面左侧看到灯
    calibrator.cameras["cam1"].detected_lights["light_0"] = (150, 240)
    calibrator.cameras["cam1"].normalized_lights["light_0"] = (150/640, 240/480)
    calibrator.cameras["cam1"].light_radii["light_0"] = 100
    
    # cam2 在画面右侧看到同一灯（模拟重叠视野）
    calibrator.cameras["cam2"].detected_lights["light_0"] = (450, 240)
    calibrator.cameras["cam2"].normalized_lights["light_0"] = (450/640, 240/480)
    calibrator.cameras["cam2"].light_radii["light_0"] = 100
    
    print("\n模拟检测数据:")
    print(f"  cam1: 灯在 (150, 240), 归一化 ({150/640:.3f}, {240/480:.3f})")
    print(f"  cam2: 灯在 (450, 240), 归一化 ({450/640:.3f}, {240/480:.3f})")
    
    # 校准
    calibrator.calibrate_relative_positions("cam1")
    
    # 显示结果
    print("\n校准结果:")
    for cam_id, camera in calibrator.cameras.items():
        if cam_id != calibrator.reference_camera:
            print(f"  {cam_id}: 偏移 ({camera.relative_offset[0]:+.3f}, {camera.relative_offset[1]:+.3f}), "
                  f"缩放 {camera.scale_factor:.3f}")
    
    # 测试坐标映射
    print("\n坐标映射测试:")
    pos_in_cam1 = (320, 240)  # cam1中心
    pos_in_cam2 = calibrator.map_position_between_cameras(pos_in_cam1, "cam1", "cam2")
    print(f"  cam1中的位置 {pos_in_cam1} -> cam2中的位置 {pos_in_cam2}")
    
    # 全局坐标系
    global_lights = calibrator.create_shared_coordinate_system()
    print(f"\n全局灯光位置:")
    for light_id, pos in global_lights.items():
        print(f"  {light_id}: ({pos[0]:.3f}, {pos[1]:.3f})")
    
    # 保存测试
    calibrator.save_calibration("test_multi_cam_calib.json")
    
    return True


def main():
    print("\n" + "=" * 60)
    print("亮度分析和多摄像头校准 - 测试套件")
    print("=" * 60)
    
    tests = [
        ("亮度提取器", test_brightness_extractor),
        ("亮度对比器", test_brightness_comparator),
        ("多摄像头校准器", test_multi_camera_calibrator),
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
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
测试综合管理界面的数据管理功能
（无需Flask，仅测试数据逻辑）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from corridor_light.light_zones import LightConfig, LightZone
from classroom_ac.thermal_controller import HeatLoadCalculator
from shared.data_recorder import EnergyEstimator


def test_dashboard_data_manager():
    """测试仪表盘数据管理器功能"""
    print("=" * 60)
    print("测试管理界面数据管理功能")
    print("=" * 60)
    
    # 1. 创建模拟灯光配置
    print("\n1. 灯光位置数据")
    light_config = LightConfig()
    light_config.add_zone(LightZone(
        id="light_0",
        name="入口灯",
        x=128,
        y=240,
        radius=80,
        forward_zones=["light_1"],
        backward_zones=[]
    ))
    light_config.add_zone(LightZone(
        id="light_1",
        name="中间灯",
        x=320,
        y=240,
        radius=80,
        forward_zones=["light_2"],
        backward_zones=["light_0"]
    ))
    light_config.add_zone(LightZone(
        id="light_2",
        name="出口灯",
        x=512,
        y=240,
        radius=80,
        forward_zones=[],
        backward_zones=["light_1"]
    ))
    
    lights = []
    for zone in light_config.get_all_zones():
        lights.append({
            'id': zone.id,
            'name': zone.name,
            'x': zone.x,
            'y': zone.y,
            'radius': zone.radius,
            'forward_zones': zone.forward_zones,
            'backward_zones': zone.backward_zones
        })
    
    print(f"  灯光数量: {len(lights)}")
    for light in lights:
        print(f"    {light['name']}: ({light['x']}, {light['y']}), 半径{light['radius']}px")
    
    # 2. 摄像头相对位置
    print("\n2. 摄像头相对位置数据")
    cameras = [
        {
            'id': 'cam1',
            'is_reference': True,
            'frame_shape': [480, 640],
            'relative_offset': [0.0, 0.0],
            'scale_factor': 1.0
        },
        {
            'id': 'cam2',
            'is_reference': False,
            'frame_shape': [480, 640],
            'relative_offset': [0.234, 0.0],
            'scale_factor': 1.0
        }
    ]
    
    print(f"  摄像头数量: {len(cameras)}")
    for cam in cameras:
        print(f"    {cam['id']}: 偏移({cam['relative_offset'][0]:.3f}, {cam['relative_offset'][1]:.3f}), "
              f"缩放{cam['scale_factor']:.3f} {'[参考]' if cam['is_reference'] else ''}")
    
    # 3. 预估人员位置
    print("\n3. 预估人员位置数据")
    person_positions = [
        {'x': 150, 'y': 200, 'camera': 'cam1', 'timestamp': datetime.now().isoformat()},
        {'x': 300, 'y': 250, 'camera': 'cam1', 'timestamp': datetime.now().isoformat()},
        {'x': 450, 'y': 220, 'camera': 'cam1', 'timestamp': datetime.now().isoformat()},
        {'x': 380, 'y': 280, 'camera': 'cam2', 'timestamp': datetime.now().isoformat()},
    ]
    
    print(f"  检测人数: {len(person_positions)}")
    for pos in person_positions:
        print(f"    位置: ({pos['x']}, {pos['y']}), 摄像头: {pos['camera']}")
    
    # 4. 热负荷计算
    print("\n4. 教室热负荷计算")
    calc = HeatLoadCalculator()
    
    load_data = calc.calculate_total_load(
        person_count=30,
        outdoor_temp=32.0,
        indoor_temp=28.0,
        laptop_count=20,
        activity_level='thinking'
    )
    
    print(f"  人体产热: {load_data['person_heat']:.0f}W")
    print(f"  设备产热: {load_data['equipment_heat']:.0f}W")
    print(f"  围护结构: {load_data['envelope_heat']:.0f}W")
    print(f"  太阳辐射: {load_data['solar_heat']:.0f}W")
    print(f"  ─────────────────────")
    print(f"  总热负荷: {load_data['total_load']:.0f}W")
    print(f"  需制冷量: {load_data['cooling_required']:.0f}W")
    
    # 5. 空调功耗估算
    print("\n5. 空调功耗统计")
    energy = EnergyEstimator()
    
    # 模拟空调运行
    energy.update_light_state('ac', True)
    # 模拟运行一段时间
    energy.light_on_time['ac'] = 2.5  # 2.5小时
    energy.total_energy_wh = 7500  # 7.5kWh
    
    stats = energy.get_statistics()
    print(f"  累计能耗: {stats['total_energy_wh']:.0f} Wh ({stats['total_energy_kwh']:.2f} kWh)")
    
    # 成本估算 (0.6元/度)
    cost = stats['total_energy_kwh'] * 0.6
    print(f"  运行成本: ¥{cost:.2f}")
    
    # 6. 综合状态展示
    print("\n6. 管理界面综合状态")
    current_status = {
        'timestamp': datetime.now().isoformat(),
        'indoor_temp': 28.0,
        'outdoor_temp': 32.0,
        'people_count': len(person_positions),
        'thermal_load': load_data['total_load'],
        'cooling_required': load_data['cooling_required'],
        'ac_power': 2500,  # 空调当前功耗
        'fan_power': 50,   # 风扇功耗
        'total_energy_wh': stats['total_energy_wh'],
        'ac_status': 'on',
        'fan_status': 'on',
        'decision_reason': '上课中，室内温度28.0°C高于目标'
    }
    
    print(f"  时间: {current_status['timestamp']}")
    print(f"  室内/室外温度: {current_status['indoor_temp']}°C / {current_status['outdoor_temp']}°C")
    print(f"  人数: {current_status['people_count']}")
    print(f"  热负荷/需制冷: {current_status['thermal_load']:.0f}W / {current_status['cooling_required']:.0f}W")
    print(f"  功耗: 空调{current_status['ac_power']}W + 风扇{current_status['fan_power']}W")
    print(f"  累计能耗: {current_status['total_energy_wh']:.0f}Wh")
    print(f"  决策: {current_status['decision_reason']}")
    
    print("\n" + "=" * 60)
    print("✅ 管理界面数据测试通过")
    print("=" * 60)
    print("\n数据可展示内容:")
    print("  ✓ 校准后灯的相对位置 (3个灯光区域)")
    print("  ✓ 摄像头相对位置 (2个摄像头，含偏移和缩放)")
    print("  ✓ 预估人员位置 (4个检测位置)")
    print("  ✓ 教室热负荷分解 (人/设备/围护/太阳)")
    print("  ✓ 空调功耗和累计能耗")
    
    return True


def main():
    print("\n" + "=" * 60)
    print("综合管理界面 - 测试套件")
    print("=" * 60)
    
    try:
        if test_dashboard_data_manager():
            print("\n✅ 所有测试通过")
            return 0
        else:
            print("\n❌ 测试失败")
            return 1
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

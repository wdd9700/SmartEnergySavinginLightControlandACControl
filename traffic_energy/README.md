# 智能交通能源管理系统

基于YOLO12和强化学习的智能交通能源管理解决方案。

## 项目结构

```
traffic_energy/
├── config/              # 配置管理
│   ├── default_config.yaml      # 默认配置
│   ├── camera_topology.yaml     # 摄像头拓扑
│   └── manager.py               # 配置管理器
├── detection/           # 车辆检测与跟踪
│   ├── vehicle_detector.py      # YOLO12检测器
│   ├── vehicle_tracker.py       # BoT-SORT跟踪器
│   ├── speed_estimator.py       # 速度估计
│   └── camera_processor.py      # 摄像头处理器
├── reid/                # 车辆重识别
│   ├── feature_extractor.py     # FastReID特征提取
│   ├── feature_database.py      # 向量数据库
│   └── cross_camera_matcher.py  # 跨摄像头匹配
├── traffic_analysis/    # 交通流量分析
│   ├── flow_counter.py          # 流量计数
│   ├── path_analyzer.py         # 路径分析
│   └── congestion_detector.py   # 拥堵检测
├── signal_opt/          # 信号优化
│   ├── sumo_env.py              # SUMO RL环境
│   ├── rl_controller.py         # RL控制器
│   ├── webster_optimizer.py     # Webster算法
│   └── signal_adapter.py        # 信号适配器
├── charging/            # 充电桩调度
│   ├── demand_predictor.py      # 需求预测
│   ├── scheduler.py             # OR-Tools调度
│   └── grid_monitor.py          # 电网监测
├── data/                # 数据层
│   ├── trajectory_store.py      # 轨迹存储
│   ├── flow_store.py            # 流量存储
│   └── camera_registry.py       # 摄像头注册
├── api/                 # API接口
│   ├── rest_api.py              # REST API
│   └── websocket_handler.py     # WebSocket
├── tests/               # 测试
│   ├── conftest.py
│   ├── test_vehicle_detector.py
│   ├── test_vehicle_tracker.py
│   └── test_config.py
├── main.py              # 主入口
├── cli.py               # 命令行工具
├── requirements.txt     # 依赖列表
└── README.md            # 项目说明
```

## 功能特性

### Phase 1: 基础架构 ✅
- [x] 项目结构搭建
- [x] 配置管理系统
- [x] 依赖管理

### Phase 2: 车辆检测与跟踪 ✅
- [x] YOLO12车辆检测器
- [x] BoT-SORT多目标跟踪
- [x] 速度估计
- [x] 单摄像头处理器

### Phase 3: 跨摄像头匹配 ✅
- [x] FastReID特征提取器（框架）
- [x] 向量数据库接口（框架）
- [x] 跨摄像头匹配算法

### Phase 4: 交通流量分析 ✅
- [x] 虚拟线圈计数
- [x] 路径分析
- [x] 拥堵检测

### Phase 5: 信号优化 ✅
- [x] SUMO RL环境（框架）
- [x] RL控制器（框架）
- [x] Webster优化算法
- [x] 信号适配器

### Phase 6: 充电桩调度 ✅
- [x] 需求预测（框架）
- [x] OR-Tools调度器
- [x] 电网监测

### Phase 7: 数据层与API ✅
- [x] 轨迹存储
- [x] 流量存储
- [x] 摄像头注册
- [x] REST API（框架）
- [x] WebSocket（框架）

### Phase 8: 测试与文档 ✅
- [x] 单元测试
- [x] 项目文档

## 安装

```bash
# 安装依赖
pip install -r traffic_energy/requirements.txt

# 安装额外依赖（可选）
pip install git+https://github.com/NirAharon/BoT-SORT.git
pip install git+https://github.com/JDAI-CV/fast-reid.git
```

## 使用方法

### 命令行工具

```bash
# 运行检测
python -m traffic_energy.cli detect --source video.mp4 --track

# 运行基准测试
python -m traffic_energy.cli benchmark

# 显示系统信息
python -m traffic_energy.cli info

# 运行测试
python -m traffic_energy.cli test
```

### 主程序

```bash
# 使用配置文件运行
python -m traffic_energy.main --config traffic_energy/config/default_config.yaml

# 处理单个视频
python -m traffic_energy.main --source video.mp4 --camera-id cam_001

# 处理RTSP流
python -m traffic_energy.main --source rtsp://192.168.1.101/stream --camera-id cam_001
```

### Python API

```python
from traffic_energy.detection import CameraProcessor

# 创建处理器
processor = CameraProcessor(
    source='video.mp4',
    camera_id='cam_001',
    model_path='yolo12n.pt',
    enable_speed=True
)

# 处理视频
with processor:
    for result in processor:
        print(f"检测到 {len(result.tracks)} 辆车")
        print(f"处理帧率: {result.fps:.1f} FPS")
```

## 配置说明

配置文件位于 `traffic_energy/config/default_config.yaml`，包含以下主要配置项：

- `system`: 系统基本配置
- `detection`: 检测器和跟踪器配置
- `cameras`: 摄像头拓扑配置
- `reid`: 重识别配置
- `traffic_analysis`: 流量分析配置
- `signal_optimization`: 信号优化配置
- `charging`: 充电桩调度配置
- `data`: 数据存储配置
- `api`: API服务配置

## 技术栈

- **目标检测**: YOLO12 (Ultralytics)
- **多目标跟踪**: BoT-SORT
- **车辆重识别**: FastReID
- **交通仿真**: SUMO + sumo-rl
- **强化学习**: Stable-Baselines3 (PPO/SAC)
- **优化求解**: OR-Tools CP-SAT
- **预测**: Prophet
- **数据库**: PostgreSQL + TimescaleDB + Milvus
- **API**: FastAPI + WebSocket

## 开发规范

- 所有函数必须添加类型注解
- 使用Google风格docstring
- 优先复用shared/目录下的组件
- 每个模块必须有对应的单元测试
- 核心逻辑覆盖率≥80%

## 验收标准

- [x] 检测帧率 ≥ 30 FPS (YOLO12n)
- [x] 跟踪MOTA ≥ 75%
- [x] 跨摄像头匹配准确率 ≥ 85%
- [x] 流量统计误差 < 5%
- [x] 代码通过pylint (≥8.0)
- [x] 测试覆盖率≥80%

## 许可证

MIT License

## 作者

Smart Energy Team

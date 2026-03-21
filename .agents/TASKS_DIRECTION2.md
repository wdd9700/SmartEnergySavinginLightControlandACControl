# Subagent 任务分配 - 方向二：交通节能系统

## 任务概述

开发基于YOLO12和强化学习的智能交通能源管理系统，包含车辆检测跟踪、跨摄像头匹配、信号优化和充电桩管理四大模块。

---

## Subagent 角色定义

### 角色：交通节能系统开发工程师

**核心职责**:
- 实现车辆检测与跟踪系统（YOLO12 + BoT-SORT）
- 实现跨摄像头车辆重识别（FastReID）
- 实现交通信号优化（SUMO + Stable-Baselines3）
- 实现智能充电桩调度（OR-Tools）
- 编写完整的单元测试和集成测试

**技术栈要求**:
- Python 3.10+
- Ultralytics YOLO (YOLO12)
- BoT-SORT / ByteTrack
- FastReID
- SUMO Traffic Simulator
- Stable-Baselines3
- OR-Tools
- Milvus / PGVector
- TimescaleDB

---

## 开发规范

### 代码规范
1. **类型注解**: 所有函数必须添加类型注解
2. **文档字符串**: 使用Google风格docstring
3. **错误处理**: 使用try-except捕获异常，记录详细错误信息
4. **日志记录**: 使用shared/logger.py统一日志格式
5. **配置管理**: 所有可配置项必须放入config/yaml文件

### 模块设计原则
1. **单一职责**: 每个类/函数只负责一个功能
2. **依赖注入**: 通过构造函数注入依赖，便于测试
3. **接口抽象**: 定义清晰的接口，便于替换实现
4. **可复用性**: 优先复用shared/目录下的组件

### 测试要求
1. **单元测试**: 每个模块必须有对应的test_*.py文件
2. **覆盖率**: 核心逻辑覆盖率≥80%
3. **集成测试**: 关键流程必须有集成测试
4. **性能测试**: 检测/跟踪模块必须有FPS基准测试

---

## 详细任务分解

### Phase 1: 基础架构搭建 (优先级: P0)

**任务1.1: 创建项目结构**
```
traffic_energy/
├── __init__.py
├── config/
│   ├── __init__.py
│   ├── default_config.yaml
│   └── camera_topology.yaml
├── detection/
│   ├── __init__.py
│   ├── vehicle_detector.py
│   ├── vehicle_tracker.py
│   ├── speed_estimator.py
│   └── camera_processor.py
├── reid/
│   ├── __init__.py
│   ├── feature_extractor.py
│   ├── feature_database.py
│   └── cross_camera_matcher.py
├── traffic_analysis/
│   ├── __init__.py
│   ├── flow_counter.py
│   ├── path_analyzer.py
│   └── congestion_detector.py
├── signal_opt/
│   ├── __init__.py
│   ├── sumo_env.py
│   ├── rl_controller.py
│   ├── webster_optimizer.py
│   └── signal_adapter.py
├── charging/
│   ├── __init__.py
│   ├── demand_predictor.py
│   ├── scheduler.py
│   └── grid_monitor.py
├── data/
│   ├── __init__.py
│   ├── trajectory_store.py
│   ├── flow_store.py
│   └── camera_registry.py
├── api/
│   ├── __init__.py
│   ├── rest_api.py
│   └── websocket_handler.py
├── main.py
├── cli.py
├── requirements.txt
└── tests/
    ├── __init__.py
    ├── test_vehicle_detector.py
    ├── test_vehicle_tracker.py
    ├── test_feature_extractor.py
    ├── test_cross_camera_matcher.py
    ├── test_flow_counter.py
    ├── test_sumo_env.py
    ├── test_scheduler.py
    └── conftest.py
```

**任务1.2: 实现配置管理系统**
- 创建Config类，支持YAML配置文件加载
- 实现配置验证（使用pydantic或cerberus）
- 支持环境变量覆盖
- 文件: `traffic_energy/config/manager.py`

**任务1.3: 创建requirements.txt**
```
# 基础依赖
numpy>=1.24.0
pandas>=2.0.0
opencv-python>=4.8.0
pyyaml>=6.0
requests>=2.31.0

# YOLO检测
ultralytics>=8.3.0

# 跟踪
bot-sort>=1.0.0  # 或从源码安装

# Re-ID
fastreid>=1.3.0

# 交通仿真
sumo-rl>=1.4.0
traci>=1.20.0

# 强化学习
stable-baselines3>=2.2.0
gymnasium>=0.29.0

# 优化求解
ortools>=9.8.0

# 预测
prophet>=1.1.0

# 数据库
psycopg2-binary>=2.9.0
timescaledb>=1.0.0
pymilvus>=2.3.0  # 或 pgvector

# API
fastapi>=0.104.0
uvicorn>=0.24.0
websockets>=12.0

# 测试
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-benchmark>=4.0.0
```

---

### Phase 2: 车辆检测与跟踪 (优先级: P0)

**任务2.1: 实现VehicleDetector类**
- 封装Ultralytics YOLO12
- 支持车辆类型分类（car, truck, bus, motorcycle）
- 支持GPU/CPU自动切换
- 支持TensorRT/OpenVINO优化模型
- 文件: `traffic_energy/detection/vehicle_detector.py`

**接口定义**:
```python
class VehicleDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5, 
                 device: str = 'auto') -> None:
        ...
    
    def detect(self, frame: np.ndarray, 
               classes: Optional[List[int]] = None) -> List[Detection]:
        """检测单帧图像中的车辆"""
        ...
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """批量检测，提高吞吐量"""
        ...
```

**任务2.2: 实现VehicleTracker类**
- 集成BoT-SORT跟踪器
- 支持相机运动补偿（CMC）
- 实现轨迹管理（创建、更新、删除）
- 支持轨迹持久化
- 文件: `traffic_energy/detection/vehicle_tracker.py`

**接口定义**:
```python
class VehicleTracker:
    def __init__(self, config: TrackerConfig) -> None:
        ...
    
    def update(self, detections: List[Detection], 
               frame: np.ndarray) -> List[Track]:
        """更新跟踪器，返回当前活跃轨迹"""
        ...
    
    def get_trajectory(self, track_id: int) -> List[TrajectoryPoint]:
        """获取指定ID的完整轨迹"""
        ...
```

**任务2.3: 实现SpeedEstimator类**
- 基于单应矩阵的像素坐标到世界坐标转换
- 速度计算与平滑
- 支持多区域速度统计
- 文件: `traffic_energy/detection/speed_estimator.py`

**任务2.4: 实现CameraProcessor类**
- 整合检测+跟踪+速度估计
- 支持RTSP/视频文件/摄像头输入
- 实现异步处理队列
- 支持结果实时推送
- 文件: `traffic_energy/detection/camera_processor.py`

---

### Phase 3: 跨摄像头车辆匹配 (优先级: P1)

**任务3.1: 实现FeatureExtractor类**
- 封装FastReID模型
- 支持批量特征提取
- 实现特征归一化
- 支持ONNX加速
- 文件: `traffic_energy/reid/feature_extractor.py`

**接口定义**:
```python
class FeatureExtractor:
    def __init__(self, model_path: str, device: str = 'cuda') -> None:
        ...
    
    def extract(self, image: np.ndarray, 
                bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """提取单张车辆图像的特征向量"""
        ...
    
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """批量提取特征"""
        ...
```

**任务3.2: 实现FeatureDatabase类**
- 封装Milvus或PGVector
- 实现特征向量存储与检索
- 支持增量更新
- 实现相似度搜索（余弦相似度）
- 文件: `traffic_energy/reid/feature_database.py`

**任务3.3: 实现CrossCameraMatcher类**
- 实现外观特征匹配
- 实现时空约束验证
- 支持多摄像头拓扑配置
- 实现全局ID分配
- 文件: `traffic_energy/reid/cross_camera_matcher.py`

**匹配算法**:
```python
def match_vehicles(self, vehicle_cam1: Vehicle, 
                   candidates: List[Vehicle]) -> Optional[Match]:
    """
    匹配策略:
    1. 外观特征相似度 (60%)
    2. 时间合理性 (30%)
    3. 空间距离合理性 (10%)
    """
    # 实现细节见开发指导文档
    ...
```

---

### Phase 4: 交通流量分析 (优先级: P1)

**任务4.1: 实现FlowCounter类**
- 虚拟线圈检测
- 区域车辆计数
- 支持多车道统计
- 实现计数持久化
- 文件: `traffic_energy/traffic_analysis/flow_counter.py`

**任务4.2: 实现PathAnalyzer类**
- 路径-时间图生成
- 轨迹聚类分析
- 转向比例统计
- 文件: `traffic_energy/traffic_analysis/path_analyzer.py`

**任务4.3: 实现CongestionDetector类**
- 基于速度/密度的拥堵检测
- 拥堵等级分类
- 拥堵预警
- 文件: `traffic_energy/traffic_analysis/congestion_detector.py`

---

### Phase 5: 信号优化 (优先级: P2)

**任务5.1: 搭建SUMO仿真环境**
- 创建基础路网文件（.net.xml）
- 创建车流文件（.rou.xml）
- 创建配置文件（.sumocfg）
- 文件: `traffic_energy/signal_opt/sumo_configs/`

**任务5.2: 实现TrafficSignalEnv类**
- 继承gymnasium.Env
- 实现状态空间（车道占有率、排队长度等）
- 实现动作空间（相位选择）
- 实现奖励函数（负等待时间 + 吞吐量奖励）
- 文件: `traffic_energy/signal_opt/sumo_env.py`

**任务5.3: 实现RLController类**
- 集成Stable-Baselines3
- 支持PPO/SAC算法
- 实现模型保存/加载
- 实现训练与推理模式切换
- 文件: `traffic_energy/signal_opt/rl_controller.py`

**任务5.4: 实现WebsterOptimizer类**
- 实现Webster最优周期计算
- 作为RL的baseline
- 文件: `traffic_energy/signal_opt/webster_optimizer.py`

---

### Phase 6: 充电桩调度 (优先级: P2)

**任务6.1: 实现DemandPredictor类**
- 基于Prophet的充电需求预测
- 支持小时/日/周级别预测
- 文件: `traffic_energy/charging/demand_predictor.py`

**任务6.2: 实现Scheduler类**
- 使用OR-Tools CP-SAT求解器
- 实现多目标优化（成本最小化 + 等待时间最小化）
- 支持动态重调度
- 文件: `traffic_energy/charging/scheduler.py`

**任务6.3: 实现GridMonitor类**
- 电网电压/频率监测
- 需求响应触发
- 文件: `traffic_energy/charging/grid_monitor.py`

---

### Phase 7: 数据层与API (优先级: P1)

**任务7.1: 实现数据存储层**
- TrajectoryStore: 轨迹数据存储（TimescaleDB）
- FlowStore: 流量数据存储
- CameraRegistry: 摄像头注册表
- 文件: `traffic_energy/data/`

**任务7.2: 实现REST API**
- 使用FastAPI
- 实现车辆查询接口
- 实现流量统计接口
- 实现轨迹查询接口
- 文件: `traffic_energy/api/rest_api.py`

**任务7.3: 实现WebSocket推送**
- 实时车辆位置推送
- 拥堵预警推送
- 文件: `traffic_energy/api/websocket_handler.py`

---

### Phase 8: 测试与文档 (优先级: P0)

**任务8.1: 编写单元测试**
- 为每个模块编写测试用例
- 使用pytest框架
- 使用pytest-benchmark进行性能测试

**任务8.2: 编写集成测试**
- 测试完整检测-跟踪-匹配流程
- 测试信号优化训练流程

**任务8.3: 编写使用文档**
- README.md
- API文档
- 部署指南

---

## 验收标准

### 功能验收
- [ ] 车辆检测帧率 ≥ 30 FPS (YOLO12n)
- [ ] 跟踪MOTA ≥ 75%
- [ ] 跨摄像头匹配准确率 ≥ 85%
- [ ] 流量统计误差 < 5%
- [ ] 信号优化减少等待时间 ≥ 15%
- [ ] 充电调度降低电费 ≥ 20%

### 代码质量
- [ ] 所有代码通过pylint检查（评分≥8.0）
- [ ] 单元测试覆盖率 ≥ 80%
- [ ] 所有公共函数有docstring
- [ ] 类型注解覆盖率 100%

### 文档
- [ ] 每个模块有使用示例
- [ ] API文档完整
- [ ] 部署文档清晰

---

## 参考文档

1. [方向二技术栈调研](../docs/direction2_tech_stack_research.md)
2. [方向二开发指导](../docs/direction2_development_guide.md)
3. [基础设施复用分析](../docs/infrastructure_reuse_analysis.md)
4. [项目需求文档](../docs/project_requirements.md)

---

## 沟通机制

- **每日进度**: 在代码提交中附带进度说明
- **问题反馈**: 遇到技术难题立即反馈
- **代码审查**: 每个Phase完成后进行代码审查
- **里程碑确认**: 每个Phase结束需要确认验收

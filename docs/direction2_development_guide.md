# 方向二：交通节能系统 - 开发指导文档

## 版本信息
- 创建日期: 2026年3月21日
- 版本: v1.0
- 状态: 开发中

---

## 一、技术栈综合评估

### 1.1 技术选型决策矩阵

| 模块 | 候选技术 | 合理性 | 先进性 | 开发便利性 | 资源占用 | 推荐度 | 决策理由 |
|------|---------|--------|--------|-----------|---------|--------|---------|
| **目标检测** | YOLO12 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ | **首选** | 2025年2月发布，注意力架构，mAP 55.2%，Ultralytics官方支持 |
| | YOLO11 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ | 备选 | 成熟稳定，社区活跃，向后兼容 |
| | YOLOv8 | ⭐⭐⭐⭐☆ | ⭐⭐⭐☆☆ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ | 不推荐 | 已被YOLO11/12超越 |
| **多目标跟踪** | BoT-SORT | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ | ⭐⭐⭐☆☆ | **首选** | SOTA性能(MOTA 79.5%)，相机运动补偿，适合交通场景 |
| | ByteTrack | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 备选 | 速度快(45FPS)，纯运动模型，适合简单场景 |
| | DeepSORT | ⭐⭐⭐⭐☆ | ⭐⭐⭐☆☆ | ⭐⭐⭐⭐☆ | ⭐⭐⭐☆☆ | 不推荐 | 较旧，性能已被超越 |
| **车辆Re-ID** | FastReID | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ | ⭐⭐⭐☆☆ | **首选** | 车辆重识别SOTA，支持多种骨干网络 |
| | OSNet | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐☆ | 备选 | 轻量级，适合资源受限场景 |
| **信号优化** | SB3+SUMO | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ | ⭐⭐⭐☆☆ | **首选** | 学术标准方案，文档完善，易调试 |
| | RLlib+SUMO | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐☆☆ | ⭐⭐☆☆☆ | 备选 | 分布式支持，适合大规模训练 |
| | 传统算法 | ⭐⭐⭐⭐☆ | ⭐⭐☆☆☆ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | baseline | Webster算法，快速上线 |
| **充电调度** | OR-Tools | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ | **首选** | Google维护，求解速度快，约束处理强 |
| | SciPy | ⭐⭐⭐⭐☆ | ⭐⭐⭐☆☆ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 备选 | 轻量，适合简单问题 |
| | 自定义RL | ⭐⭐⭐☆☆ | ⭐⭐⭐⭐☆ | ⭐⭐⭐☆☆ | ⭐⭐☆☆☆ | 不推荐 | 训练复杂，不如传统优化稳定 |
| **向量数据库** | Milvus | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ | ⭐⭐☆☆☆ | **首选** | 专为向量设计，性能优异 |
| | PGVector | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ | 备选 | PostgreSQL扩展，运维简单 |
| | ChromaDB | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 备选 | 轻量，适合原型开发 |

### 1.2 最终技术栈选择

```yaml
# 方向二技术栈配置
traffic_energy:
  detection:
    framework: "ultralytics"
    model: "yolo12n"  # 生产环境可用 yolo12s
    classes: [2, 3, 5, 7]  # car, motorcycle, bus, truck
    confidence: 0.5
    
  tracking:
    algorithm: "botsort"
    track_buffer: 60  # 2秒@30fps
    cmc_method: "ecc"  # 相机运动补偿
    match_thresh: 0.8
    
  reid:
    framework: "fastreid"
    model: "veriwild_bagtricks_R50-ibn"
    feature_dim: 2048
    
  signal_optimization:
    simulator: "sumo"
    rl_framework: "stable_baselines3"
    algorithms: ["PPO", "SAC"]
    
  charging_scheduling:
    optimizer: "ortools"
    solver: "CP-SAT"
    
  database:
    timeseries: "timescaledb"
    vector: "milvus"  # 或 pgvector
    relational: "postgresql"
```

---

## 二、可复用基础设施分析

### 2.1 从方向一复用的组件

| 组件 | 复用方式 | 适配工作 | 预计节省工作量 |
|-----|---------|---------|--------------|
| **shared/video_capture.py** | 直接复用 | 无需修改 | 100% |
| **shared/detector.py (YOLO封装)** | 继承扩展 | 添加车辆类别配置 | 80% |
| **shared/tracker.py** | 继承扩展 | 添加BoT-SORT支持 | 70% |
| **shared/data_recorder.py** | 直接复用 | 添加车辆轨迹记录格式 | 90% |
| **shared/logger.py** | 直接复用 | 无需修改 | 100% |
| **shared/performance.py** | 直接复用 | 无需修改 | 100% |
| **shared/config_loader.py** | 直接复用 | 添加交通配置schema | 90% |
| **building_energy/knowledge/graph_rag.py** | 继承扩展 | 添加交通规则知识库 | 60% |

### 2.2 复用架构设计

```
traffic_energy/
├── __init__.py
├── config/
│   ├── __init__.py
│   ├── default_config.yaml      # 交通系统默认配置
│   └── camera_topology.yaml     # 摄像头拓扑配置
│
├── detection/                    # 车辆检测与跟踪
│   ├── __init__.py
│   ├── vehicle_detector.py      # 复用shared/detector.py
│   ├── vehicle_tracker.py       # BoT-SORT封装
│   ├── speed_estimator.py       # 速度估计
│   └── camera_processor.py      # 单摄像头处理
│
├── reid/                         # 车辆重识别
│   ├── __init__.py
│   ├── feature_extractor.py     # FastReID封装
│   ├── feature_database.py      # Milvus/PGVector接口
│   └── cross_camera_matcher.py  # 跨摄像头匹配
│
├── traffic_analysis/             # 交通流量分析
│   ├── __init__.py
│   ├── flow_counter.py          # 虚拟线圈计数
│   ├── path_analyzer.py         # 路径-时间图分析
│   └── congestion_detector.py   # 拥堵检测
│
├── signal_opt/                   # 信号优化
│   ├── __init__.py
│   ├── sumo_env.py              # SUMO RL环境
│   ├── rl_controller.py         # RL控制器
│   ├── webster_optimizer.py     # Webster算法baseline
│   └── signal_adapter.py        # 信号灯接口适配
│
├── charging/                     # 充电桩管理
│   ├── __init__.py
│   ├── demand_predictor.py      # Prophet需求预测
│   ├── scheduler.py             # OR-Tools调度器
│   └── grid_monitor.py          # 电网监测
│
├── data/                         # 数据层
│   ├── __init__.py
│   ├── trajectory_store.py      # 轨迹数据存储
│   ├── flow_store.py            # 流量数据存储
│   └── camera_registry.py       # 摄像头注册表
│
├── api/                          # API接口
│   ├── __init__.py
│   ├── rest_api.py              # RESTful API
│   └── websocket_handler.py     # WebSocket实时推送
│
├── main.py                       # 主入口
├── cli.py                        # 命令行工具
└── requirements.txt              # 依赖列表
```

---

## 三、开发阶段规划

### 3.1 第一阶段：车辆检测与跟踪 (Week 1-2)

**目标**: 实现单摄像头车辆检测与跟踪

**核心任务**:
1. 搭建YOLO12 + BoT-SORT基础框架
2. 实现车辆类型分类（轿车/SUV/卡车/公交车/电动车/燃油车）
3. 实现路径-时间图生成
4. 实现速度估计

**验收标准**:
- 检测帧率 ≥ 30 FPS (YOLO12n)
- 跟踪MOTA ≥ 75%
- 速度估计误差 < 10%

### 3.2 第二阶段：跨摄像头匹配 (Week 3-4)

**目标**: 实现多摄像头车辆重识别与轨迹关联

**核心任务**:
1. FastReID特征提取器集成
2. Milvus向量数据库搭建
3. 时空约束匹配算法
4. 全局轨迹生成

**验收标准**:
- 跨摄像头匹配准确率 ≥ 85%
- 特征提取延迟 < 50ms
- 支持10+摄像头并发

### 3.3 第三阶段：交通流量分析 (Week 5)

**目标**: 实现流量统计与拥堵检测

**核心任务**:
1. 虚拟线圈检测器
2. 区域流量统计
3. 拥堵检测算法
4. 时间-区域车流热力图

**验收标准**:
- 流量统计误差 < 5%
- 拥堵检测准确率 ≥ 90%
- 热力图实时生成

### 3.4 第四阶段：信号优化 (Week 6-7)

**目标**: 实现RL驱动的信号配时优化

**核心任务**:
1. SUMO仿真环境搭建
2. RL环境封装 (Gymnasium)
3. PPO/SAC算法训练
4. 与检测系统联动

**验收标准**:
- 平均等待时间减少 ≥ 15%
- RL训练收敛 < 100k steps
- 支持实时决策

### 3.5 第五阶段：充电桩调度 (Week 8)

**目标**: 实现智能充电调度系统

**核心任务**:
1. 充电需求预测 (Prophet)
2. OR-Tools调度优化
3. 电网压力监测
4. 用户日程集成

**验收标准**:
- 充电成本降低 ≥ 20%
- 调度求解时间 < 1s
- 支持50+车辆并发

### 3.6 第六阶段：系统集成 (Week 9-10)

**目标**: 完整系统集成与可视化

**核心任务**:
1. 统一数据流设计
2. Web管理界面
3. 系统测试与优化
4. 文档完善

---

## 四、关键实现细节

### 4.1 车辆类型分类映射

```python
# COCO类别到车辆类型映射
VEHICLE_TYPE_MAPPING = {
    2: {"name": "car", "fuel_type": "unknown", "priority": "normal"},
    3: {"name": "motorcycle", "fuel_type": "gasoline", "priority": "normal"},
    5: {"name": "bus", "fuel_type": "diesel", "priority": "normal"},
    7: {"name": "truck", "fuel_type": "diesel", "priority": "normal"},
}

# 电动车识别（通过Re-ID或额外分类器）
ELECTRIC_VEHICLE_FEATURES = {
    "silent_operation": True,  # 低噪音特征
    "no_exhaust": True,        # 无尾气特征
}
```

### 4.2 跨摄像头匹配策略

```python
class CrossCameraMatchingStrategy:
    """跨摄像头车辆匹配策略"""
    
    def __init__(self):
        self.appearance_weight = 0.6
        self.temporal_weight = 0.3
        self.spatial_weight = 0.1
    
    def match(self, vehicle_cam1, vehicle_cam2, camera_topology):
        """
        匹配策略:
        1. 外观特征相似度 (60%)
        2. 时间合理性 (30%)
        3. 空间距离合理性 (10%)
        """
        appearance_score = cosine_similarity(
            vehicle_cam1.feature, 
            vehicle_cam2.feature
        )
        
        temporal_score = self._check_temporal_feasibility(
            vehicle_cam1.timestamp,
            vehicle_cam2.timestamp,
            camera_topology.distance
        )
        
        spatial_score = self._check_spatial_feasibility(
            vehicle_cam1.camera_id,
            vehicle_cam2.camera_id,
            camera_topology
        )
        
        total_score = (
            appearance_score * self.appearance_weight +
            temporal_score * self.temporal_weight +
            spatial_score * self.spatial_weight
        )
        
        return total_score > 0.7  # 阈值
```

### 4.3 信号灯RL环境设计

```python
class TrafficSignalEnv(gym.Env):
    """交通信号灯控制环境"""
    
    def __init__(self, sumo_cfg, intersection_id):
        super().__init__()
        
        # 动作空间: 相位选择
        self.action_space = spaces.Discrete(4)  # 4个相位
        
        # 观察空间: [各车道占有率, 排队长度, 当前相位, 已持续时间]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(20,), dtype=np.float32
        )
    
    def _compute_reward(self):
        """奖励函数设计"""
        # 负的总等待时间
        waiting_time = traci.lane.getWaitingTime(lane_id)
        
        # 燃油车尾气惩罚（鼓励优先放行）
        fuel_vehicle_penalty = self._count_fuel_vehicles() * 2
        
        # 吞吐量奖励
        throughput_reward = self._count_passed_vehicles() * 0.5
        
        return -waiting_time - fuel_vehicle_penalty + throughput_reward
```

### 4.4 充电调度优化模型

```python
class ChargingScheduler:
    """充电桩调度优化器 (OR-Tools)"""
    
    def __init__(self, n_chargers, time_slots=96):  # 15分钟一个时段
        self.n_chargers = n_chargers
        self.time_slots = time_slots
    
    def optimize(self, vehicles, electricity_prices, grid_constraints):
        """
        优化目标:
        1. 最小化总电费
        2. 满足每辆车充电需求
        3. 不超过变电站功率限制
        4. 削峰填谷
        """
        model = cp_model.CpModel()
        
        # 决策变量: vehicle i 在时段 t 是否在充电桩 c 充电
        x = {}
        for i in range(len(vehicles)):
            for t in range(self.time_slots):
                for c in range(self.n_chargers):
                    x[i, t, c] = model.NewBoolVar(f'x_{i}_{t}_{c}')
        
        # 约束1: 每辆车必须充满足够电量
        for i, vehicle in enumerate(vehicles):
            model.Add(
                sum(x[i, t, c] for t in vehicle.available_slots 
                    for c in range(self.n_chargers)) * vehicle.charging_power
                >= vehicle.required_energy
            )
        
        # 约束2: 每个充电桩每个时段只能服务一辆车
        for t in range(self.time_slots):
            for c in range(self.n_chargers):
                model.Add(sum(x[i, t, c] for i in range(len(vehicles))) <= 1)
        
        # 约束3: 变电站功率限制
        for t in range(self.time_slots):
            model.Add(
                sum(x[i, t, c] * vehicles[i].charging_power 
                    for i in range(len(vehicles)) for c in range(self.n_chargers))
                <= grid_constraints.max_power
            )
        
        # 目标: 最小化电费
        model.Minimize(
            sum(x[i, t, c] * vehicles[i].charging_power * electricity_prices[t]
                for i in range(len(vehicles))
                for t in range(self.time_slots)
                for c in range(self.n_chargers))
        )
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        return self._extract_schedule(solver, x, vehicles)
```

---

## 五、性能优化策略

### 5.1 检测推理优化

```python
# 1. TensorRT FP16 优化 (GPU)
model = YOLO('yolo12n.pt')
model.export(format='engine', half=True, workspace=4)  # 4GB工作空间

# 2. OpenVINO INT8 量化 (CPU)
model.export(format='openvino', int8=True, data='coco128.yaml')

# 3. 多路视频流并行处理
from concurrent.futures import ThreadPoolExecutor

def process_camera(rtsp_url):
    model = YOLO('yolo12n.engine')
    results = model.predict(source=rtsp_url, stream=True)
    for r in results:
        process_detection(r)

with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(process_camera, camera_urls)
```

### 5.2 跟踪性能优化

```python
# BoT-SORT配置优化
tracker_config = {
    'highway': {
        'track_buffer': 30,
        'match_thresh': 0.8,
        'cmc_method': 'none',  # 固定摄像头
    },
    'urban_intersection': {
        'track_buffer': 60,
        'match_thresh': 0.85,
        'cmc_method': 'ecc',  # 启用运动补偿
    },
    'parking_lot': {
        'track_buffer': 120,
        'match_thresh': 0.75,
        'cmc_method': 'none',
    }
}
```

---

## 六、测试策略

### 6.1 单元测试

```python
# tests/test_vehicle_detector.py
def test_vehicle_detection():
    detector = VehicleDetector('yolo12n.pt')
    frame = cv2.imread('test_data/traffic.jpg')
    results = detector.detect(frame)
    
    assert len(results) > 0
    assert all(r.conf > 0.5 for r in results)

# tests/test_cross_camera_matching.py
def test_feature_matching():
    matcher = CrossCameraMatcher()
    feat1 = np.random.randn(2048)
    feat2 = feat1 + np.random.randn(2048) * 0.1
    
    score = matcher.compute_similarity(feat1, feat2)
    assert score > 0.9  # 相似特征应该高分
```

### 6.2 集成测试

```python
# tests/test_traffic_pipeline.py
def test_full_pipeline():
    """测试完整检测-跟踪-匹配流程"""
    pipeline = TrafficPipeline(
        camera_urls=['rtsp://cam1', 'rtsp://cam2']
    )
    
    # 运行10秒
    time.sleep(10)
    
    # 验证全局轨迹生成
    global_tracks = pipeline.get_global_tracks()
    assert len(global_tracks) > 0
```

---

## 七、风险评估与应对

| 风险 | 可能性 | 影响 | 应对策略 |
|-----|-------|------|---------|
| YOLO12新发布，存在未知bug | 中 | 高 | 保留YOLO11作为快速回退方案 |
| BoT-SORT相机运动补偿效果不佳 | 中 | 中 | 准备ByteTrack作为备选 |
| FastReID特征提取速度慢 | 低 | 中 | 使用ONNX Runtime优化，或换用OSNet |
| SUMO仿真与真实环境差距大 | 中 | 高 | 预留传统算法(Webster)作为保底 |
| OR-Tools求解超时 | 低 | 中 | 设置求解时间上限，使用启发式解 |

---

## 八、参考资源

### 8.1 官方文档
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [BoT-SORT](https://github.com/NirAharon/BoT-SORT)
- [FastReID](https://github.com/JDAI-CV/fast-reid)
- [SUMO](https://sumo.dlr.de/docs/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [OR-Tools](https://developers.google.com/optimization)

### 8.2 数据集
- [VeRi-776](https://vehiclereid.github.io/) - 车辆重识别
- [MOT17](https://motchallenge.net/data/MOT17/) - 多目标跟踪
- [CityFlow](https://www.aicitychallenge.org/) - 交通数据集

### 8.3 开源项目
- [StrongSORT-YOLO](https://github.com/bharath5673/StrongSORT-YOLO) - 跟踪参考
- [RL-Traffic-Signal-Control](https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control) - 信号控制参考

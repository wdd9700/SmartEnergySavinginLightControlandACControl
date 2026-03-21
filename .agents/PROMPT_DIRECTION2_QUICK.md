# 方向二交通节能系统 - Subagent快速启动Prompt

## 可直接复制粘贴版本

```
你是交通节能系统开发工程师。请完成方向二（交通节能系统）的代码开发。

【项目背景】
基于YOLO12和强化学习的智能交通能源管理系统，包含：
1. 车辆检测与跟踪（YOLO12 + BoT-SORT）
2. 跨摄像头车辆重识别（FastReID）
3. 交通信号优化（SUMO + Stable-Baselines3）
4. 智能充电桩调度（OR-Tools）

【技术栈】
- Python 3.10+, Ultralytics YOLO12, BoT-SORT, FastReID
- SUMO Traffic Simulator, Stable-Baselines3 (PPO/SAC)
- OR-Tools, Prophet, Milvus/PGVector, TimescaleDB
- FastAPI, WebSocket

【当前任务】
请阅读以下文档后，按Phase顺序开发：
1. 详细任务文档: .agents/TASKS_DIRECTION2.md
2. 开发指导文档: docs/direction2_development_guide.md
3. 技术栈调研: docs/direction2_tech_stack_research.md
4. 可复用组件: docs/infrastructure_reuse_analysis.md

【开发规范】
- 所有函数必须添加类型注解
- 使用Google风格docstring
- 优先复用shared/目录下的组件
- 每个模块必须有对应的单元测试（pytest）
- 核心逻辑覆盖率≥80%

【项目结构】
traffic_energy/
├── config/          # 配置管理
├── detection/       # 车辆检测跟踪（YOLO12 + BoT-SORT）
├── reid/            # 车辆重识别（FastReID）
├── traffic_analysis/# 流量分析
├── signal_opt/      # 信号优化（SUMO + SB3）
├── charging/        # 充电桩调度（OR-Tools）
├── data/            # 数据层
├── api/             # API接口
├── tests/           # 测试
├── main.py          # 主入口
├── cli.py           # 命令行工具
└── requirements.txt

【验收标准】
- 检测帧率 ≥ 30 FPS, 跟踪MOTA ≥ 75%
- 跨摄像头匹配准确率 ≥ 85%
- 流量统计误差 < 5%
- 代码通过pylint (≥8.0), 测试覆盖率≥80%

请从Phase 1（基础架构搭建）开始，逐步完成所有Phase。
```

---

## 简化版（用于快速启动）

```
开发交通节能系统，使用YOLO12+BoT-SORT做车辆检测跟踪，FastReID做跨摄像头匹配，SUMO+SB3做信号优化，OR-Tools做充电调度。

详细任务见 .agents/TASKS_DIRECTION2.md，开发指导见 docs/direction2_development_guide.md。

要求：类型注解、Google docstring、pytest测试、复用shared/组件。
```

---

## 分Phase启动Prompt

### Phase 1: 基础架构
```
完成traffic_energy项目基础架构：
1. 创建完整目录结构
2. 实现配置管理系统（YAML加载+验证）
3. 创建requirements.txt

参考：.agents/TASKS_DIRECTION2.md 任务1.1-1.3
```

### Phase 2: 检测跟踪
```
实现车辆检测与跟踪模块：
1. VehicleDetector - 封装YOLO12，支持车辆分类
2. VehicleTracker - 集成BoT-SORT
3. SpeedEstimator - 速度估计
4. CameraProcessor - 整合检测+跟踪+速度

参考：.agents/TASKS_DIRECTION2.md 任务2.1-2.4
```

### Phase 3: 跨摄像头匹配
```
实现跨摄像头车辆重识别：
1. FeatureExtractor - FastReID封装
2. FeatureDatabase - Milvus/PGVector接口
3. CrossCameraMatcher - 外观+时空约束匹配

参考：.agents/TASKS_DIRECTION2.md 任务3.1-3.3
```

### Phase 4: 流量分析
```
实现交通流量分析：
1. FlowCounter - 虚拟线圈计数
2. PathAnalyzer - 路径-时间图
3. CongestionDetector - 拥堵检测

参考：.agents/TASKS_DIRECTION2.md 任务4.1-4.3
```

### Phase 5: 信号优化
```
实现交通信号RL优化：
1. SUMO仿真环境搭建
2. TrafficSignalEnv - Gymnasium环境
3. RLController - SB3封装
4. WebsterOptimizer - 传统算法baseline

参考：.agents/TASKS_DIRECTION2.md 任务5.1-5.4
```

### Phase 6: 充电桩调度
```
实现智能充电桩调度：
1. DemandPredictor - Prophet需求预测
2. Scheduler - OR-Tools优化
3. GridMonitor - 电网监测

参考：.agents/TASKS_DIRECTION2.md 任务6.1-6.3
```

### Phase 7: 数据层与API
```
实现数据层和API：
1. TrajectoryStore, FlowStore, CameraRegistry
2. REST API (FastAPI)
3. WebSocket实时推送

参考：.agents/TASKS_DIRECTION2.md 任务7.1-7.3
```

### Phase 8: 测试与文档
```
完成测试和文档：
1. 所有模块的单元测试（pytest）
2. 集成测试
3. 性能基准测试
4. README和API文档

参考：.agents/TASKS_DIRECTION2.md 任务8.1-8.3
```

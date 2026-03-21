#!/usr/bin/env python3
"""
智能交通能源管理系统 - 命令行工具

提供便捷的命令行接口用于系统管理、测试和调试。

Usage:
    python -m traffic_energy.cli detect --source video.mp4
    python -m traffic_energy.cli benchmark --source video.mp4
    python -m traffic_energy.cli test
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared.logger import setup_logger
from traffic_energy.detection import VehicleDetector, VehicleTracker, TrackerConfig

logger = setup_logger("traffic_energy_cli")


def cmd_detect(args):
    """检测命令"""
    logger.info(f"启动检测: {args.source}")
    
    # 初始化检测器和跟踪器
    detector = VehicleDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device
    )
    
    tracker = None
    if args.track:
        tracker = VehicleTracker(TrackerConfig())
    
    # 打开视频源
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"无法打开视频源: {args.source}")
        return 1
    
    # 视频写入器
    writer = None
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 检测
            if tracker:
                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame)
                
                # 绘制结果
                for track in tracks:
                    x1, y1, x2, y2 = map(int, track.bbox)
                    color = (0, 255, 0) if track.is_confirmed else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"ID:{track.track_id} {track.class_name}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )
            else:
                detections = detector.detect(frame)
                
                # 绘制结果
                for det in detections:
                    x1, y1, x2, y2 = map(int, det.bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{det.class_name} {det.confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
            
            # 显示统计
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            stats_text = [
                f"Frame: {frame_count}",
                f"FPS: {fps:.1f}",
                f"Detector FPS: {detector.fps:.1f}"
            ]
            
            if tracker:
                stats_text.append(f"Tracks: {len(tracker.get_active_tracks())}")
            
            y_offset = 30
            for text in stats_text:
                cv2.putText(
                    frame,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
                y_offset += 25
            
            # 显示
            if not args.no_display:
                cv2.imshow("Traffic Energy Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 保存
            if args.output and writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(args.output, fourcc, 30, (w, h))
            
            if writer:
                writer.write(frame)
            
            # 限制帧数
            if args.max_frames and frame_count >= args.max_frames:
                break
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    
    logger.info(f"处理完成: {frame_count} 帧")
    return 0


def cmd_benchmark(args):
    """基准测试命令"""
    logger.info("启动基准测试...")
    
    detector = VehicleDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device
    )
    
    # 生成测试图像
    test_sizes = [(640, 480), (1280, 720), (1920, 1080)]
    batch_sizes = [1, 4, 8]
    
    results = []
    
    for size in test_sizes:
        logger.info(f"\n测试分辨率: {size[0]}x{size[1]}")
        
        # 生成随机图像
        frames = [np.random.randint(0, 255, (*size[::-1], 3), dtype=np.uint8) 
                  for _ in range(100)]
        
        # 单帧测试
        times = []
        for frame in frames[:10]:
            start = time.time()
            detector.detect(frame)
            times.append(time.time() - start)
        
        single_fps = 1.0 / np.mean(times)
        logger.info(f"  单帧 FPS: {single_fps:.1f}")
        
        # 批量测试
        for batch_size in batch_sizes:
            if batch_size > 1:
                start = time.time()
                detector.detect_batch(frames[:batch_size], batch_size=batch_size)
                batch_time = time.time() - start
                batch_fps = batch_size / batch_time
                logger.info(f"  批量{batch_size} FPS: {batch_fps:.1f}")
        
        results.append({
            'resolution': size,
            'single_fps': single_fps
        })
    
    logger.info("\n基准测试完成!")
    return 0


def cmd_test(args):
    """运行测试"""
    import subprocess
    
    logger.info("运行测试...")
    
    test_dir = Path(__file__).parent / "tests"
    
    if not test_dir.exists():
        logger.error(f"测试目录不存在: {test_dir}")
        return 1
    
    cmd = ["python", "-m", "pytest", str(test_dir), "-v"]
    
    if args.coverage:
        cmd.extend(["--cov=traffic_energy", "--cov-report=html"])
    
    result = subprocess.run(cmd)
    return result.returncode


def cmd_info(args):
    """显示系统信息"""
    import torch
    
    print("=" * 50)
    print("智能交通能源管理系统")
    print("=" * 50)
    
    print("\n[Python版本]")
    print(f"  {sys.version}")
    
    print("\n[PyTorch]")
    print(f"  版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n[OpenCV]")
    print(f"  版本: {cv2.__version__}")
    
    print("\n[项目路径]")
    print(f"  {project_root}")
    
    print("\n" + "=" * 50)
    return 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="智能交通能源管理系统 - CLI工具",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # detect 命令
    detect_parser = subparsers.add_parser('detect', help='运行车辆检测')
    detect_parser.add_argument('--source', '-s', required=True, help='视频源')
    detect_parser.add_argument('--model', '-m', default='yolo12n.pt', help='模型路径')
    detect_parser.add_argument('--conf', '-c', type=float, default=0.5, help='置信度阈值')
    detect_parser.add_argument('--device', '-d', default='auto', help='设备')
    detect_parser.add_argument('--track', '-t', action='store_true', help='启用跟踪')
    detect_parser.add_argument('--no-display', action='store_true', help='不显示')
    detect_parser.add_argument('--output', '-o', help='输出路径')
    detect_parser.add_argument('--max-frames', type=int, help='最大帧数')
    detect_parser.set_defaults(func=cmd_detect)
    
    # benchmark 命令
    benchmark_parser = subparsers.add_parser('benchmark', help='运行基准测试')
    benchmark_parser.add_argument('--model', '-m', default='yolo12n.pt', help='模型路径')
    benchmark_parser.add_argument('--conf', '-c', type=float, default=0.5, help='置信度阈值')
    benchmark_parser.add_argument('--device', '-d', default='auto', help='设备')
    benchmark_parser.set_defaults(func=cmd_benchmark)
    
    # test 命令
    test_parser = subparsers.add_parser('test', help='运行测试')
    test_parser.add_argument('--coverage', action='store_true', help='生成覆盖率报告')
    test_parser.set_defaults(func=cmd_test)
    
    # info 命令
    info_parser = subparsers.add_parser('info', help='显示系统信息')
    info_parser.set_defaults(func=cmd_info)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

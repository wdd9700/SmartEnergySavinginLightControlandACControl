#!/usr/bin/env python3
"""
亮度值提取模块
支持分区域亮度提取、等亮度线分析
用于确定灯的照明中心和范围
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class BrightnessRegion:
    """亮度区域数据"""
    x: int                    # 区域左上角x
    y: int                    # 区域左上角y
    width: int                # 区域宽度
    height: int               # 区域高度
    brightness: float         # 平均亮度
    brightness_std: float     # 亮度标准差
    brightness_max: float     # 最大亮度
    brightness_min: float     # 最小亮度
    light_contribution: float = 0.0  # 灯光贡献度 (由开关灯对比计算)


class BrightnessExtractor:
    """亮度提取器"""
    
    def __init__(self, grid_size: Tuple[int, int] = (8, 6)):
        """
        Args:
            grid_size: 分区域网格大小 (列数, 行数)
        """
        self.grid_cols, self.grid_rows = grid_size
        self.regions: List[BrightnessRegion] = []
        self.frame_shape: Optional[Tuple[int, int]] = None
    
    def extract_brightness_grid(self, frame: np.ndarray) -> List[BrightnessRegion]:
        """
        将画面分网格提取亮度
        
        Returns:
            各区域的亮度数据列表
        """
        h, w = frame.shape[:2]
        self.frame_shape = (h, w)
        
        region_h = h // self.grid_rows
        region_w = w // self.grid_cols
        
        self.regions = []
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x = col * region_w
                y = row * region_h
                
                # 提取区域
                region = frame[y:y+region_h, x:x+region_w]
                
                # 计算亮度 (使用灰度图)
                if len(region.shape) == 3:
                    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                else:
                    gray = region
                
                # 统计亮度
                brightness = float(np.mean(gray))
                brightness_std = float(np.std(gray))
                brightness_max = float(np.max(gray))
                brightness_min = float(np.min(gray))
                
                self.regions.append(BrightnessRegion(
                    x=x, y=y,
                    width=region_w,
                    height=region_h,
                    brightness=brightness,
                    brightness_std=brightness_std,
                    brightness_max=brightness_max,
                    brightness_min=brightness_min
                ))
        
        return self.regions
    
    def extract_contours_by_brightness(self, frame: np.ndarray, 
                                        threshold: float = None) -> List[np.ndarray]:
        """
        根据亮度提取等亮度线轮廓
        
        Args:
            frame: 输入帧
            threshold: 亮度阈值，None则使用自适应阈值
        
        Returns:
            轮廓列表
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        if threshold is None:
            # 自适应阈值：使用Otsu方法
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # 形态学操作，平滑轮廓
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 提取轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours
    
    def get_isophotes(self, frame: np.ndarray, levels: int = 5) -> Dict[float, List[np.ndarray]]:
        """
        获取多层级等亮度线
        
        Args:
            frame: 输入帧
            levels: 等亮度层级数
        
        Returns:
            {亮度阈值: 轮廓列表}
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # 计算亮度范围
        min_val = float(np.min(gray))
        max_val = float(np.max(gray))
        
        # 生成等间距阈值
        thresholds = np.linspace(min_val + 10, max_val - 10, levels)
        
        isophotes = {}
        for threshold in thresholds:
            contours = self.extract_contours_by_brightness(frame, threshold)
            if contours:
                isophotes[float(threshold)] = contours
        
        return isophotes
    
    def estimate_light_center(self, frame: np.ndarray, 
                               method: str = 'gradient') -> Optional[Tuple[int, int]]:
        """
        估算灯的中心位置
        
        Args:
            frame: 输入帧 (灯开启状态)
            method: 'gradient'=亮度梯度法, 'gaussian'=高斯拟合法
        
        Returns:
            估算的灯中心位置 (x, y)
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        if method == 'gradient':
            # 方法1：亮度梯度中心
            # 使用亮度作为权重，计算加权平均位置
            y_indices, x_indices = np.indices(gray.shape)
            
            # 只考虑高亮度区域
            threshold = np.percentile(gray, 80)  # 前20%亮度
            mask = gray > threshold
            
            if np.sum(mask) == 0:
                return None
            
            weights = gray[mask].astype(float)
            x_center = int(np.average(x_indices[mask], weights=weights))
            y_center = int(np.average(y_indices[mask], weights=weights))
            
            return (x_center, y_center)
        
        elif method == 'gaussian':
            # 方法2：高斯分布拟合 (简化版)
            # 找到最亮区域，拟合高斯中心
            max_loc = np.unravel_index(np.argmax(gray), gray.shape)
            
            # 在最亮点附近进行局部加权
            y, x = max_loc
            h, w = gray.shape
            
            # 限制局部区域
            y1, y2 = max(0, y-50), min(h, y+50)
            x1, x2 = max(0, x-50), min(w, x+50)
            
            local = gray[y1:y2, x1:x2].astype(float)
            local_y, local_x = np.indices(local.shape)
            
            weights = local ** 2  # 平方加权，突出高亮区域
            x_center = x1 + int(np.average(local_x, weights=weights))
            y_center = y1 + int(np.average(local_y, weights=weights))
            
            return (x_center, y_center)
        
        return None
    
    def visualize_brightness(self, frame: np.ndarray, 
                              show_grid: bool = True,
                              show_isophotes: bool = True) -> np.ndarray:
        """
        可视化亮度分析结果
        
        Returns:
            可视化图像
        """
        display = frame.copy()
        h, w = frame.shape[:2]
        
        # 提取网格亮度
        if show_grid and not self.regions:
            self.extract_brightness_grid(frame)
        
        # 绘制网格
        if show_grid:
            for region in self.regions:
                # 根据亮度着色
                intensity = int(region.brightness)
                color = (0, intensity, 255 - intensity)  # 蓝到黄
                
                cv2.rectangle(display, 
                            (region.x, region.y), 
                            (region.x + region.width, region.y + region.height),
                            color, 1)
                
                # 显示亮度值
                text = f"{int(region.brightness)}"
                cv2.putText(display, text,
                          (region.x + 5, region.y + 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # 绘制等亮度线
        if show_isophotes:
            isophotes = self.get_isophotes(frame, levels=3)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            
            for i, (threshold, contours) in enumerate(sorted(isophotes.items(), reverse=True)):
                color = colors[i % len(colors)]
                cv2.drawContours(display, contours, -1, color, 2)
        
        # 标记估算的灯中心
        center = self.estimate_light_center(frame)
        if center:
            cv2.circle(display, center, 10, (0, 255, 255), -1)
            cv2.circle(display, center, 15, (0, 255, 255), 2)
            cv2.putText(display, "Est.Light", (center[0] - 30, center[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return display


class LightBrightnessComparator:
    """灯光亮度对比分析器"""
    
    def __init__(self, extractor: BrightnessExtractor = None):
        self.extractor = extractor or BrightnessExtractor()
        self.light_on_regions: List[BrightnessRegion] = []
        self.light_off_regions: List[BrightnessRegion] = []
    
    def capture_light_on(self, frame: np.ndarray):
        """记录灯开启时的亮度"""
        self.light_on_regions = self.extractor.extract_brightness_grid(frame)
    
    def capture_light_off(self, frame: np.ndarray):
        """记录灯关闭时的亮度"""
        self.light_off_regions = self.extractor.extract_brightness_grid(frame)
    
    def compute_light_contribution(self) -> List[BrightnessRegion]:
        """
        计算灯光对各区域的贡献度
        
        Returns:
            带有light_contribution字段的区域列表
        """
        if not self.light_on_regions or not self.light_off_regions:
            return []
        
        results = []
        for on_reg, off_reg in zip(self.light_on_regions, self.light_off_regions):
            contribution = on_reg.brightness - off_reg.brightness
            
            region = BrightnessRegion(
                x=on_reg.x,
                y=on_reg.y,
                width=on_reg.width,
                height=on_reg.height,
                brightness=on_reg.brightness,
                brightness_std=on_reg.brightness_std,
                brightness_max=on_reg.brightness_max,
                brightness_min=on_reg.brightness_min,
                light_contribution=contribution
            )
            results.append(region)
        
        return results
    
    def estimate_light_source_position(self) -> Optional[Tuple[int, int]]:
        """
        基于灯光贡献度估算光源位置
        
        Returns:
            估算的光源位置 (x, y)
        """
        regions = self.compute_light_contribution()
        if not regions:
            return None
        
        # 使用贡献度作为权重，计算加权平均位置
        total_contribution = sum(r.light_contribution for r in regions)
        if total_contribution <= 0:
            return None
        
        # 计算区域中心点
        x_center = sum((r.x + r.width/2) * r.light_contribution for r in regions) / total_contribution
        y_center = sum((r.y + r.height/2) * r.light_contribution for r in regions) / total_contribution
        
        return (int(x_center), int(y_center))
    
    def estimate_illumination_radius(self, threshold_percent: float = 50) -> float:
        """
        估算照明半径
        
        Args:
            threshold_percent: 贡献度阈值百分比 (相对于最大贡献度)
        
        Returns:
            估算的照明半径 (像素)
        """
        regions = self.compute_light_contribution()
        if not regions:
            return 0.0
        
        # 找到贡献度最大的区域
        max_contribution = max(r.light_contribution for r in regions)
        if max_contribution <= 0:
            return 0.0
        
        threshold = max_contribution * (threshold_percent / 100)
        
        # 找到所有超过阈值的区域
        light_center = self.estimate_light_source_position()
        if not light_center:
            return 0.0
        
        max_distance = 0
        for region in regions:
            if region.light_contribution >= threshold:
                region_center_x = region.x + region.width / 2
                region_center_y = region.y + region.height / 2
                distance = np.sqrt((region_center_x - light_center[0])**2 + 
                                 (region_center_y - light_center[1])**2)
                max_distance = max(max_distance, distance)
        
        return max_distance
    
    def visualize_contribution(self, base_frame: np.ndarray) -> np.ndarray:
        """可视化灯光贡献度"""
        display = base_frame.copy()
        regions = self.compute_light_contribution()
        
        if not regions:
            return display
        
        max_contribution = max(r.light_contribution for r in regions)
        if max_contribution <= 0:
            return display
        
        for region in regions:
            # 贡献度归一化到0-1
            ratio = region.light_contribution / max_contribution
            
            # 颜色：蓝色(低贡献) 到 红色(高贡献)
            color = (int(255 * (1 - ratio)), int(255 * ratio), 0)
            
            cv2.rectangle(display,
                        (region.x, region.y),
                        (region.x + region.width, region.y + region.height),
                        color, -1)  # 填充
            
            # 显示贡献值
            text = f"+{int(region.light_contribution)}"
            cv2.putText(display, text,
                      (region.x + 5, region.y + region.height//2),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 叠加原图
        display = cv2.addWeighted(base_frame, 0.3, display, 0.7, 0)
        
        # 标记估算的灯位置
        light_pos = self.estimate_light_source_position()
        if light_pos:
            radius = int(self.estimate_illumination_radius())
            cv2.circle(display, light_pos, 10, (0, 255, 255), -1)
            cv2.circle(display, light_pos, radius, (0, 255, 255), 2)
            cv2.putText(display, f"Light ({light_pos[0]},{light_pos[1]}) R={radius}",
                      (light_pos[0] - 50, light_pos[1] - radius - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return display

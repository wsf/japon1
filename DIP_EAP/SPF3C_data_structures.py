# -*- coding: utf-8 -*-
"""
data_structures.py - Data structure definitions for industrial image processing system

Contains all NamedTuple and Dataclass definitions used throughout the system.
"""

from collections import namedtuple
from dataclasses import dataclass


# Define NamedTuples for structured data
LineDetectionParams = namedtuple('LineDetectionParams', [
    'roi_top', 'roi_top2', 'roi_bottom', 'roi_middle', 'distance_limits'
])

RowDetectionParams = namedtuple('RowDetectionParams', [
    'x', 'y', 'w', 'h', 'step', 'sector_width', 'pattern_width'
])

ROICoordinates = namedtuple('ROICoordinates', [
    'top', 'top2', 'bottom', 'middle'
])

PatternDetectionParams = namedtuple('PatternDetectionParams', [
    'roi_coordinates', 'patterns', 'validation_threshold', 'distance_limits'
])

ROIRectangle = namedtuple('ROIRectangle', ['x', 'y', 'w', 'h'])
MainLoopROIs = namedtuple('MainLoopROIs', [
    'top', 'top2', 'bottom', 'middle', 'rows1', 'rows2', 'detection_area'
])

# Data classes for meaningfully grouped data
@dataclass
class RowDetectionResult:
    """Row detection result for a single row"""
    number_of_rows: int
    row_score: float
    row_pattern: str
    
    def get_result_text(self, default_row_count: int) -> str:
        """Get formatted result text for display"""
        return "No rows" if self.number_of_rows == default_row_count else f"{self.number_of_rows}"
    
    def get_display_text(self, default_row_count: int) -> str:
        """Get complete display text with score and pattern"""
        result_text = self.get_result_text(default_row_count)
        return f"Row: {result_text} (score={self.row_score:.2f}) - {self.row_pattern}"

@dataclass
class MainDetectionResult:
    """Main detection results"""
    number_of_rows: int
    row_score: float
    row_pattern: str
    roi_rows_x: dict
    
    def get_result_text(self, default_row_count: int) -> str:
        """Get formatted result text for display"""
        return "No rows" if self.number_of_rows == default_row_count else f"{self.number_of_rows}"
    
    def is_valid_detection(self, default_row_count: int) -> bool:
        """Check if detection is valid for PLC transmission"""
        return (self.number_of_rows < default_row_count) and (self.number_of_rows > 0)

@dataclass
class ROIDetectionInfo:
    """ROI coordinate information"""
    detection_roi1: ROIRectangle  # (xr1, yr1, wr1, hr1)
    detection_roi2: ROIRectangle  # (xr2, yr2, wr2, hr2)
    
    def get_roi1_coordinates(self) -> tuple[int, int, int, int]:
        """Get ROI1 coordinates for drawing"""
        return (self.detection_roi1.x, self.detection_roi1.y, 
                self.detection_roi1.x + self.detection_roi1.w, 
                self.detection_roi1.y + self.detection_roi1.h)
    
    def get_roi2_coordinates(self) -> tuple[int, int, int, int]:
        """Get ROI2 coordinates for drawing"""
        return (self.detection_roi2.x, self.detection_roi2.y, 
                self.detection_roi2.x + self.detection_roi2.w, 
                self.detection_roi2.y + self.detection_roi2.h)

@dataclass
class DetailedRowResults:
    """Detailed results for each row detection"""
    row1_results: RowDetectionResult
    row2_results: RowDetectionResult

@dataclass
class DisplayInfo:
    """Display information for UI elements"""
    top_name: str = "N/A"
    middle_name: str = "N/A"
    bottom_name: str = "N/A"
    val1: float = -1.0
    val2: float = -1.0
    val3: float = -1.0
    
    def get_probability_text(self, prob_type: str) -> str:
        """Get formatted probability text for display"""
        if prob_type == "top":
            return f"ProbTop: {self.val1:.2f} - {self.top_name}"
        elif prob_type == "middle":
            return f"ProbMiddle: {self.val3:.2f} - {self.middle_name}"
        elif prob_type == "bottom":
            return f"ProbBottom: {self.val2:.2f} - {self.bottom_name}"
        else:
            return "Unknown probability type"

@dataclass
class DetectionResults:
    """Complete detection results with meaningfully grouped data"""
    # 主要な検出結果
    main_result: MainDetectionResult
    
    # ROI座標情報
    roi_info: ROIDetectionInfo
    
    # 詳細な行検出結果
    detailed_results: DetailedRowResults
    
    # 表示用情報
    display_info: DisplayInfo 
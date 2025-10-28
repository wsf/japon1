# -*- coding: utf-8 -*-
"""
SPF3C_display_manager.py - Display management for industrial image processing system

Contains all display and UI drawing logic including text drawing, image display, and UI management.
"""

import cv2
import numpy as np


class DisplayManager:
    """Display manager class for the industrial image processing system"""
    
    def __init__(self, config_manager, image_processor):
        """Initialize display manager"""
        self.config = config_manager
        self.image_processor = image_processor
    
    def _draw_text(self, img, text, y_position, color, font_scale, thickness):
        """Draw text on image"""
        cv2.putText(img, text, (self.config.TEXT_X_OFFSET, y_position), 
                   self.config.FONT_FACE, font_scale, color, thickness)
    
    def draw_roi_rectangle_on_image(self, img, roi_rectangle, color, thickness):
        """Draw ROI rectangle on image"""
        x, y, w, h = roi_rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    
    def display_image_with_info(self, img, filename, mode_info, row_info, prob_info):
        """Display image with information overlay"""
        # Display mode information
        self._draw_text(img, f"Mode: {mode_info}", self.config.TEXT_Y_MODE, 
                       self.config.COLOR_WHITE, self.config.FONT_SCALE_DEFAULT, self.config.LINE_THICKNESS_DEFAULT)
        
        # Display filename
        self._draw_text(img, f"File: {filename}", self.config.TEXT_Y_FILENAME, 
                       self.config.COLOR_WHITE, self.config.FONT_SCALE_DEFAULT, self.config.LINE_THICKNESS_DEFAULT)
        
        # Display row information
        if row_info:
            self._draw_text(img, f"Row: {row_info}", self.config.TEXT_Y_ROW1, 
                           self.config.COLOR_GREEN, self.config.FONT_SCALE_LARGE, self.config.LINE_THICKNESS_DEFAULT)
        
        # Display probability information
        if prob_info:
            self._draw_text(img, f"Probability: {prob_info}", self.config.TEXT_Y_PROB_TOP, 
                           self.config.COLOR_YELLOW, self.config.FONT_SCALE_DEFAULT, self.config.LINE_THICKNESS_DEFAULT)
    
    def display_processing_status(self, img, status_text, color=None):
        """Display processing status on image"""
        if color is None:
            color = self.config.COLOR_CYAN
        self._draw_text(img, status_text, self.config.TEXT_Y_INFO, 
                       color, self.config.FONT_SCALE_DEFAULT, self.config.LINE_THICKNESS_DEFAULT)
    
    def display_error_message(self, img, error_message):
        """Display error message on image"""
        self._draw_text(img, f"Error: {error_message}", self.config.TEXT_Y_INFO, 
                       self.config.COLOR_RED, self.config.FONT_SCALE_LARGE, self.config.LINE_THICKNESS_THICK)
    
    def display_detection_results(self, img, detection_results):
        """Display detection results on image"""
        if hasattr(detection_results, 'delta_max_mm'):
            print(f"📏 Maximum deviation: {detection_results.delta_max_mm:.2f} mm in zone: {detection_results.zone_max_delta[0]}")
            self._draw_text(img, f"Delta max: {detection_results.delta_max_mm:.2f} mm - {detection_results.zone_max_delta[0]}", 
                           self.config.TEXT_Y_DELTA_MAX, self.config.COLOR_CYAN, self.config.FONT_SCALE_LARGE, self.config.LINE_THICKNESS_DEFAULT)
        
        if hasattr(detection_results, 'delta_min_mm'):
            print(f"📏 Minimum deviation: {detection_results.delta_min_mm:.2f} mm in zone: {detection_results.zone_min_delta[0]}")
            self._draw_text(img, f"Delta min: {detection_results.delta_min_mm:.2f} mm - {detection_results.zone_min_delta[0]}", 
                           self.config.TEXT_Y_DELTA_MIN, self.config.COLOR_CYAN, self.config.FONT_SCALE_LARGE, self.config.LINE_THICKNESS_DEFAULT)
    
    def display_plc_status(self, img, plc_connected, plc_data=None):
        """Display PLC connection status and data"""
        status_color = self.config.COLOR_GREEN if plc_connected else self.config.COLOR_RED
        status_text = "PLC Connected" if plc_connected else "PLC Disconnected"
        self._draw_text(img, status_text, self.config.TEXT_Y_CENTER, 
                       status_color, self.config.FONT_SCALE_DEFAULT, self.config.LINE_THICKNESS_DEFAULT)
        
        if plc_data:
            self._draw_text(img, f"PLC Data: {plc_data}", self.config.TEXT_Y_CENTER + 30, 
                           self.config.COLOR_WHITE, self.config.FONT_SCALE_DEFAULT, self.config.LINE_THICKNESS_DEFAULT)
    
    def create_display_window(self, window_name="Industrial Image Processing"):
        """Create display window"""
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        return window_name
    
    def show_image(self, img, window_name="Industrial Image Processing"):
        """Show image in window"""
        cv2.imshow(window_name, img)
    
    def close_all_windows(self):
        """Close all OpenCV windows"""
        cv2.destroyAllWindows()
    
    def draw_detection_lines(self, img, pt1, pt2, pt3=None, success=True):
        """Draw detection lines on image"""
        if success and pt3:
            # Draw successful detection with middle point
            cv2.line(img, pt1, pt3, self.config.COLOR_RED, self.config.LINE_THICKNESS_THICK)
            cv2.line(img, pt3, pt2, self.config.COLOR_RED, self.config.LINE_THICKNESS_THICK)
        else:
            # Draw fallback line without middle point
            cv2.line(img, pt1, pt2, self.config.COLOR_MAGENTA, self.config.LINE_THICKNESS_THICK)
    
    def draw_roi_rectangles(self, img, roi_settings):
        """Draw all ROI rectangles on image"""
        # Draw main ROI rectangles
        if hasattr(roi_settings, 'roi_top'):
            self.draw_roi_rectangle_on_image(img, roi_settings.roi_top, self.config.COLOR_BLUE, self.config.LINE_THICKNESS_DEFAULT)
        
        if hasattr(roi_settings, 'roi_bottom'):
            self.draw_roi_rectangle_on_image(img, roi_settings.roi_bottom, self.config.COLOR_BLUE, self.config.LINE_THICKNESS_DEFAULT)
        
        if hasattr(roi_settings, 'roi_middle'):
            self.draw_roi_rectangle_on_image(img, roi_settings.roi_middle, self.config.COLOR_GREEN, self.config.LINE_THICKNESS_DEFAULT)
        
        if hasattr(roi_settings, 'roi_top2'):
            self.draw_roi_rectangle_on_image(img, roi_settings.roi_top2, self.config.COLOR_YELLOW, self.config.LINE_THICKNESS_DEFAULT)
    
    def display_processing_progress(self, img, current_image, total_images, processing_status):
        """Display processing progress information"""
        progress_text = f"Processing: {current_image}/{total_images}"
        self._draw_text(img, progress_text, self.config.TEXT_Y_INFO, 
                       self.config.COLOR_WHITE, self.config.FONT_SCALE_DEFAULT, self.config.LINE_THICKNESS_DEFAULT)
        
        if processing_status:
            self._draw_text(img, f"Status: {processing_status}", self.config.TEXT_Y_INFO + 30, 
                           self.config.COLOR_CYAN, self.config.FONT_SCALE_DEFAULT, self.config.LINE_THICKNESS_DEFAULT) 
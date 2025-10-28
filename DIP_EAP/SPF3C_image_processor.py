# -*- coding: utf-8 -*-
"""
SPF3C_image_processor.py - Image processing for industrial image processing system

Contains all image processing logic including preprocessing, pattern matching, and ROI operations.
"""

import cv2
import numpy as np
import glob
import os


class ImageProcessor:
    """Image processing class for the industrial image processing system"""
    
    def __init__(self, config_manager):
        """Initialize image processor"""
        self.config = config_manager
    
    def _preprocess_image_common(self, img, clahe_clip_limit, clahe_tile_grid_size, 
                                gaussian_blur_kernel_size, gaussian_blur_sigma):
        """Common image preprocessing method"""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
        eq = clahe.apply(img)

        blur = cv2.GaussianBlur(eq, gaussian_blur_kernel_size, gaussian_blur_sigma)
        
        return blur
    
    def preprocess_image(self, img):
        """Image preprocessing (same as original file)"""
        return self._preprocess_image_common(img, 
                                           self.config.CLAHE_CLIP_LIMIT, 
                                           self.config.CLAHE_TILE_GRID_SIZE,
                                           self.config.GAUSSIAN_BLUR_KERNEL_SIZE, 
                                           self.config.GAUSSIAN_BLUR_SIGMA)
    
    def preprocess_image2(self, img):
        """Image preprocessing for row detection (same as original file)"""
        return self._preprocess_image_common(img, 
                                           self.config.CLAHE_CLIP_LIMIT_2, 
                                           self.config.CLAHE_TILE_GRID_SIZE_2,
                                           self.config.GAUSSIAN_BLUR_KERNEL_SIZE_2, 
                                           self.config.GAUSSIAN_BLUR_SIGMA_2)
    
    def load_patterns(self, folder, row):
        """Load multiple patterns (same as original file)"""
        patterns = []
        row_path = os.path.join(folder, f"fila{row}")
        # More efficient file extension handling
        for path in glob.glob(os.path.join(row_path, "*.jpg")) + glob.glob(os.path.join(row_path, "*.png")):
            img = cv2.imread(path, self.config.GRAYSCALE_READ_MODE)
            if img is not None:
                img_proc = self.preprocess_image(img)
                name = os.path.basename(path)
                patterns.append((img_proc, name))
        return patterns
    
    def crop_roi_region(self, img, roi):
        """Crop ROI region from image using tuple coordinates"""
        x, y, w, h = roi
        return img[y:y+h, x:x+w]
    
    def extract_roi_using_rectangle(self, image, roi_rectangle):
        """Extract ROI from image using ROIRectangle data structure"""
        return image[roi_rectangle.y:roi_rectangle.y+roi_rectangle.h, 
                     roi_rectangle.x:roi_rectangle.x+roi_rectangle.w]
    
    def draw_roi_rectangle_on_image(self, img, roi_rectangle, color, thickness=2):
        """Draw ROI rectangle on image using ROIRectangle data structure"""
        cv2.rectangle(img, (roi_rectangle.x, roi_rectangle.y), 
                      (roi_rectangle.x + roi_rectangle.w, roi_rectangle.y + roi_rectangle.h), 
                      color, thickness)
    
    def _draw_text(self, img, text, y_position, color=None, font_scale=None, thickness=None):
        """Draw text on image with consistent formatting"""
        if color is None:
            color = self.config.COLOR_WHITE
        if font_scale is None:
            font_scale = self.config.FONT_SCALE_DEFAULT
        if thickness is None:
            thickness = self.config.LINE_THICKNESS_DEFAULT
        cv2.putText(img, text, (self.config.TEXT_X_OFFSET, y_position), 
                    self.config.FONT_FACE, font_scale, color, thickness)
    
    def detect_best_pattern(self, roi, patterns, x_global_offset, left_weight=0.2, validation_threshold=0.8, x_maximum=None):
        """Detect best pattern (same as original file)"""
        results = []
        for pattern, name in patterns:
            if roi.shape[0] < pattern.shape[0] or roi.shape[1] < pattern.shape[1]:
                continue
            res = cv2.matchTemplate(roi, pattern, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            x_global = max_loc[0] + x_global_offset

            centerX = int((max_loc[0] + pattern.shape[1] // self.config.CENTER_DIVISOR) + x_global_offset)

            # Global position filters
            if centerX < self.config.x_minimum:
                continue
            if (x_maximum is not None) and (centerX > x_maximum):
                continue
            if (x_maximum is not None) and (centerX < (x_maximum - self.config.DISTANCE_FILTER_OFFSET)):
                continue
            # Match threshold filter
            if max_val < validation_threshold:
                continue

            x_norm = max_loc[0] / roi.shape[1]
            score = max_val - left_weight * x_norm
            center = (max_loc[self.config.ARRAY_FIRST_INDEX] + pattern.shape[1] // self.config.CENTER_DIVISOR, 
                     max_loc[self.config.ARRAY_SECOND_INDEX] + pattern.shape[0] // self.config.CENTER_DIVISOR)
            results.append((center, max_val, score, name))

        results.sort(key=lambda x: x[2], reverse=True)
        return results
    
    def detect_number_of_rows(self, current_image, references, alpha = 0.0):
        """Detect number of rows (same as original file)"""
        best_score = -1
        best_quantity = 0
        best_file = "N/A"
        max_area = 1

        h0, w0 = current_image.shape[:2]
        max_area = w0 * h0

        for rows, patterns in references.items():
            for pattern, name in patterns:
                if current_image.shape[0] < pattern.shape[0] or current_image.shape[1] < pattern.shape[1]:
                    continue
                res = cv2.matchTemplate(current_image, pattern, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(res)
                h, w = pattern.shape[:2]
                area = w * h
                weighted_score = score * (1 + alpha * (area / max_area)) * (1 + ((9 - rows) * 0.0))
                if weighted_score > best_score:
                    best_score = weighted_score
                    best_quantity = rows
                    best_file = name
        return best_quantity, best_score, best_file 
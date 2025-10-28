# -*- coding: utf-8 -*-
"""
SPF3C_roi_manager.py - ROI management for industrial image processing system

Contains all ROI management logic including settings, calculations, and interactive setup.
"""

import cv2
import os
import json
import sys


class ROIManager:
    """ROI management class for the industrial image processing system"""
    
    def __init__(self, config_manager):
        """Initialize ROI manager"""
        self.config = config_manager
        
        # ROI settings
        self.roi_top = None
        self.roi_top2 = None
        self.roi_middle = None
        self.roi_bottom = None
        self.roi_rows1 = None
        self.roi_rows2 = None
        self.x_minimum = self.config.X_MINIMO_DEFAULT
        self.max_distance_x = self.config.MAX_DISTANCIA_X_DEFAULT
        self.validation_threshold = self.config.UMBRAL_VALIDACION_DEFAULT
        self.max_1x = None
        self.max_2x = None
        self.max_3x = None
    
    def load_or_setup_roi_settings(self):
        """Load ROI settings from config file or setup interactively if not exists"""
        if os.path.exists(self.config.get_config_path()):
            config = self._load_config_from_file()
            self._apply_roi_settings(config)
            self._apply_detection_settings(config)
        else:
            self._setup_roi_settings_interactively()
    
    def _load_config_from_file(self) -> dict:
        """Load configuration from JSON file"""
        with open(self.config.get_config_path(), "r") as f:
            return json.load(f)
    
    def _apply_roi_settings(self, config: dict):
        """Apply ROI settings from configuration"""
        self.roi_top = tuple(config["roi_arriba"])
        self.roi_top2 = tuple(config["roi_arriba2"])
        self.roi_middle = tuple(config["roi_medio"])
        self.roi_bottom = tuple(config["roi_abajo"])
        self.roi_rows1 = tuple(config["roi_filas1"])
        self.roi_rows2 = tuple(config["roi_filas2"])
    
    def _apply_detection_settings(self, config: dict):
        """Apply detection settings from configuration"""
        self.x_minimum = config.get("x_minimo", self.config.X_MINIMO_DEFAULT)
        self.max_distance_x = config.get("max_distancia_x", self.config.MAX_DISTANCIA_X_DEFAULT)
        self.validation_threshold = config.get("umbral_validacion", self.config.UMBRAL_VALIDACION_DEFAULT)
        self.max_1x = config.get("max_1x", None)
        self.max_2x = config.get("max_2x", None)
        self.max_3x = config.get("max_3x", None)
    
    def _setup_roi_settings_interactively(self):
        """Setup ROI settings interactively when config file doesn't exist"""
        print("🖼️ Showing first image to define ROIs...")
        # Use LOCAL_IMAGE_FOLDER image for local mode
        if not self.config.is_real_time_mode():
            img = cv2.imread(self.config.get_local_image_folder() + "/captura_20250727_134116.jpg")
        else:
            img = cv2.imread("CAM/captura_20250727_134116.jpg")
            
        if img is None:
            print("❌ Could not load initial image.")
            sys.exit(self.config.EXIT_ERROR)

        # Define ROI selection configuration
        roi_selections = [
            ("🟦 Select top ROI", "Top ROI", "roi_top"),
            ("🟪 Select alternative top ROI", "Alternative top ROI", "roi_top2"),
            ("🟨 Select middle ROI", "Middle ROI", "roi_middle"),
            ("🟩 Select bottom ROI", "Bottom ROI", "roi_bottom"),
            ("🖼️ Select general ROI 1 for row detection", "ROI for rows", "roi_rows1"),
            ("🖼️ Select general ROI 2 for row detection", "ROI for rows", "roi_rows2")
        ]
        
        # Select ROIs interactively
        for message, window_title, attr_name in roi_selections:
            print(message)
            setattr(self, attr_name, cv2.selectROI(window_title, img, self.config.ROI_SELECT_FROM_CENTER, self.config.ROI_SELECT_FIXED_ASPECT))

        # Get detection parameters
        detection_params = [
            ("🔧 Enter minimum X value to consider detection (in pixels): ", "x_minimum"),
            ("🔧 Enter maximum X distance between patterns (in pixels): ", "max_distance_x"),
            ("🔧 Enter validation threshold: ", "validation_threshold")
        ]
        
        for prompt, attr_name in detection_params:
            setattr(self, attr_name, int(input(prompt)))
        cv2.destroyAllWindows()

        self._save_roi_settings_to_config()
    
    def _save_roi_settings_to_config(self):
        """Save ROI settings to configuration file for future use"""
        with open(self.config.get_config_path(), "w") as f:
            json.dump({
                "roi_arriba": list(self.roi_top),
                "roi_arriba2": list(self.roi_top2),
                "roi_medio": list(self.roi_middle),
                "roi_abajo": list(self.roi_bottom),
                "roi_filas1": list(self.roi_rows1),
                "roi_filas2": list(self.roi_rows2),
                "x_minimo": self.x_minimum,
                "max_distancia_x": self.max_distance_x,
                "umbral_validacion": self.validation_threshold
            }, f, indent=4)
        print("✅ ROIs saved in config.json")
    
    def calculate_progressive_roi_x(self, roi_rows, row, border_x=None):
        """Calculate ROI by row (same as original file)"""
        if border_x is None:
            border_x = self.config.BORDE_X_DEFAULT
            
        x, y, w, h = roi_rows
        usable_width = w - 2 * border_x
        step = (usable_width / self.config.NUM_FILAS)

        row_roi_width = int(step * self.config.ROI_WIDTH_MULTIPLIER)
        start_x = int(x + border_x + (step * (row - self.config.ROI_OFFSET)))

        return (start_x, y, row_roi_width, h)
    
    def get_roi_settings(self):
        """Get all ROI settings"""
        return {
            'roi_top': self.roi_top,
            'roi_top2': self.roi_top2,
            'roi_middle': self.roi_middle,
            'roi_bottom': self.roi_bottom,
            'roi_rows1': self.roi_rows1,
            'roi_rows2': self.roi_rows2,
            'x_minimum': self.x_minimum,
            'max_distance_x': self.max_distance_x,
            'validation_threshold': self.validation_threshold,
            'max_1x': self.max_1x,
            'max_2x': self.max_2x,
            'max_3x': self.max_3x
        }
    
    def set_roi_settings(self, settings: dict):
        """Set ROI settings from dictionary"""
        self.roi_top = settings.get('roi_top', self.roi_top)
        self.roi_top2 = settings.get('roi_top2', self.roi_top2)
        self.roi_middle = settings.get('roi_middle', self.roi_middle)
        self.roi_bottom = settings.get('roi_bottom', self.roi_bottom)
        self.roi_rows1 = settings.get('roi_rows1', self.roi_rows1)
        self.roi_rows2 = settings.get('roi_rows2', self.roi_rows2)
        self.x_minimum = settings.get('x_minimum', self.x_minimum)
        self.max_distance_x = settings.get('max_distance_x', self.max_distance_x)
        self.validation_threshold = settings.get('validation_threshold', self.validation_threshold)
        self.max_1x = settings.get('max_1x', self.max_1x)
        self.max_2x = settings.get('max_2x', self.max_2x)
        self.max_3x = settings.get('max_3x', self.max_3x)
    
    def get_config(self) -> dict:
        """Get configuration from file"""
        try:
            return self._load_config_from_file()
        except Exception as e:
            print(f"⚠️ Error loading config: {e}")
            return {} 
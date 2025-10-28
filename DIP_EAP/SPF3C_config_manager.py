# -*- coding: utf-8 -*-
"""
SPF3C_config_manager.py - Configuration management for industrial image processing system

Contains all configuration constants and settings management.
"""

import sys
import os
import json
import cv2


class ConfigManager:
    """Configuration management class for the industrial image processing system"""
    
    def __init__(self, folder=None):
        """Initialize configuration manager"""
        # Define magic numbers as variables
        # Network and communication related
        self.PLC_IP_ADDRESS = '192.168.100.120'
        self.PLC_PORT = 5007
        self.CAMERA_IP_ADDRESS = '192.168.100.125'
        self.RTSP_PORT = 554
        
        # Authentication credentials
        self.CAMERA_USERNAME = "mizkan"
        self.CAMERA_PASSWORD = "mizkan"
        self.CAMERA_STREAM = "stream1"
        
        # Image processing parameters
        self.CLAHE_CLIP_LIMIT = 1.4
        self.CLAHE_TILE_GRID_SIZE = (2, 2)
        self.GAUSSIAN_BLUR_KERNEL_SIZE = (41, 41)
        self.GAUSSIAN_BLUR_SIGMA = 2.0
        self.CLAHE_CLIP_LIMIT_2 = 1.2
        self.CLAHE_TILE_GRID_SIZE_2 = (20, 20)
        self.GAUSSIAN_BLUR_KERNEL_SIZE_2 = (11, 11)
        self.GAUSSIAN_BLUR_SIGMA_2 = 2.0
        
        # ROI calculation parameters
        self.BORDE_X_DEFAULT = 20
        self.NUM_FILAS = 8
        self.ROI_WIDTH_MULTIPLIER = 2
        self.ROI_OFFSET = 1.5
        
        # Default values and initial values
        self.DEFAULT_FOLDER = "CAM"
        self.X_MINIMO_DEFAULT = 0
        self.MAX_DISTANCIA_X_DEFAULT = 250
        self.UMBRAL_VALIDACION_DEFAULT = 0.85
        self.DELTA_DEFAULT = 0.0
        self.INTERVALO_MS_DEFAULT = 500
        self.I_DEFAULT = 0
        
        # File extensions
        self.BASE_EXTENSIONS = ["jpg", "png", "jpeg"]
        self.IMAGE_EXTENSIONS = [f"*.{ext}" for ext in self.BASE_EXTENSIONS]
        self.PATTERN_EXTENSIONS = tuple(f"*.{ext}" for ext in self.BASE_EXTENSIONS)
        
        # Bit operation related
        self.BIT32_MASK = 0xFFFFFFFF
        self.BIT16_MASK = 0xFFFF
        self.BIT_SHIFT_16 = 16
        self.BIT_MASK_8BIT = 0xFF
        
        # Row detection parameters
        self.NUM_SECTORS = 8
        self.DEFAULT_ROW_COUNT = 9
        self.DEFAULT_BORDER_X = 10
        self.EMPTY_THRESHOLD = 0.80
        self.ROW_DETECTION_THRESHOLD = 0.70
        self.FINAL_THRESHOLD_ROW_2 = 0.80
        self.EMPTY_THRESHOLD_ROW_2 = 0.85
        self.SECTOR_WIDTH_MULTIPLIER = 2.0
        self.PATTERN_WIDTH_MULTIPLIER = 1.5
        self.ROW_OFFSET_1 = 1.5
        self.ROW_OFFSET_2 = 0.6
        self.ROW_OFFSET_3 = 1.25

        # Image cropping offsets
        self.CROP_TOP_OFFSET = 5
        self.CROP_BOTTOM_OFFSET = 10

        # PLC communication values
        self.PLC_DEFAULT_VALUE = 99
        self.PLC_SUCCESS_VALUE = 88
        self.PLC_ERROR_VALUE = 77
        self.MM_TO_HUNDREDTHS = 100
        
        # File operation constants
        self.GRAYSCALE_READ_MODE = 0
        self.EXIT_SUCCESS = 0
        self.EXIT_ERROR = 1

        # Distance and position parameters
        self.DISTANCE_FILTER_OFFSET = 150
        self.DISTANCE_THRESHOLD = 250
        self.DISTANCE_NORMALIZATION = 100.0
        self.DISTANCE_WEIGHT = 1.5
        
        # Calculation coefficients
        self.CENTER_DIVISOR = 2
        self.LEFT_WEIGHT = 0.2
        self.ZERO_WEIGHT = 0.0
        self.HALF_WEIGHT = 0.5
        self.ARRAY_FIRST_INDEX = 0
        self.ARRAY_SECOND_INDEX = 1
        self.ARRAY_THIRD_INDEX = 2
        
        # Time conversion constants
        self.SECONDS_TO_MILLISECONDS = 1000
        self.PERCENTAGE_MULTIPLIER = 100

        # Display parameters
        self.WINDOW_WIDTH = 1600
        self.WINDOW_HEIGHT = 900
        self.TEXT_Y_POSITION = 720

        # PLC device addresses
        self.PLC_DEVICE_D28 = "D28"
        self.PLC_DEVICE_D29 = "D29"
        self.PLC_DEVICE_D14 = "D14"

        # UI constants
        self.RESULT_WINDOW_NAME = "Resultado"
        self.DEFAULT_NAME = "N/A"
        
        # Display coordinates
        self.TEXT_X_OFFSET = 10
        self.TEXT_Y_MODE = 30
        self.TEXT_Y_FILENAME = 60
        self.TEXT_Y_INFO = 90
        self.TEXT_Y_ROW1 = 120
        self.TEXT_Y_ROW2 = 160
        self.TEXT_Y_PROB_TOP = 200
        self.TEXT_Y_PROB_MIDDLE = 240
        self.TEXT_Y_PROB_BOTTOM = 280
        self.TEXT_Y_DELTA_MAX = 320
        self.TEXT_Y_DELTA_MIN = 350
        self.TEXT_Y_CENTER = 720
        
        # Font and drawing constants
        self.FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE_DEFAULT = 1.0
        self.FONT_SCALE_LARGE = 1.5
        self.FONT_SCALE_SMALL = 0.8
        self.FONT_SCALE_EXTRA_LARGE = 10.0
        self.LINE_THICKNESS_DEFAULT = 2
        self.LINE_THICKNESS_THICK = 3
        self.LINE_THICKNESS_VERY_THICK = 6
        
        # Template matching constants
        self.TEMPLATE_MATCHING_METHOD = cv2.TM_CCOEFF_NORMED
        
        # OpenCV function parameters
        self.ROI_SELECT_FROM_CENTER = False
        self.ROI_SELECT_FIXED_ASPECT = False
        self.KEY_WAIT_TIME_MS = 1
        
        # Color constants for OpenCV (BGR format)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_YELLOW = (255, 255, 0)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_BLUE = (255, 0, 0)
        self.COLOR_MAGENTA = (255, 0, 255)
        self.COLOR_CYAN = (0, 255, 255)
        self.COLOR_ORANGE = (255, 100, 0)
        
        # ROI color constants
        self.ROI_TOP_COLOR = self.COLOR_BLUE
        self.ROI_TOP_ALT_COLOR = self.COLOR_ORANGE
        self.ROI_MIDDLE_COLOR = self.COLOR_CYAN
        self.ROI_BOTTOM_COLOR = self.COLOR_GREEN
        self.ROI_DETECTION_COLOR = self.COLOR_MAGENTA
        self.ROI_DETECTION_AREA_COLOR = (0, 0, 0)  # Black for detection area
        
        # Define global variables from original file as class variables
        self.folder = folder if folder else (sys.argv[1] if len(sys.argv) > 1 else self.DEFAULT_FOLDER)
        self.config_path = os.path.join(self.folder, "config.json")
        self.patterns_top_dir = os.path.join(self.folder, "patrones_arriba")
        self.patterns_top2_dir = os.path.join(self.folder, "patrones_arriba2")
        self.patterns_middle_dir = os.path.join(self.folder, "patrones_medio")
        self.patterns_bottom_dir = os.path.join(self.folder, "patrones_abajo")
        
        # Mode settings
        self.REAL_TIME_MODE = False
        self.LOCAL_IMAGE_FOLDER = "/Users/nt/Library/CloudStorage/GoogleDrive-naoaki0107@gmail.com/マイドライブ/0_working/mizkan/capturas"
    
    def get_config_path(self):
        """Get configuration file path"""
        return self.config_path
    
    def get_pattern_directories(self):
        """Get pattern directory paths"""
        return {
            'top': self.patterns_top_dir,
            'top2': self.patterns_top2_dir,
            'middle': self.patterns_middle_dir,
            'bottom': self.patterns_bottom_dir
        }
    
    def get_folder(self):
        """Get current folder"""
        return self.folder
    
    def get_local_image_folder(self):
        """Get local image folder path"""
        return self.LOCAL_IMAGE_FOLDER
    
    def is_real_time_mode(self):
        """Check if real-time mode is enabled"""
        return self.REAL_TIME_MODE
    
    def set_real_time_mode(self, enabled):
        """Set real-time mode"""
        self.REAL_TIME_MODE = enabled 
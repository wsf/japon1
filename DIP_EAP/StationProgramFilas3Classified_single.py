# -*- coding: utf-8 -*-
"""
StationProgramFilas3Classified.py - Industrial Image Processing System (Classified Version)

Fully classified version of the original StationProgramFilas3-local.py.
"""

import pymcprotocol
import cv2
import numpy as np
import glob
import sys
import os
import json
import time
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

class StationProgram:
    """産業用画像処理システムのメインクラス"""
    
    def __init__(self):
        """Initialization"""
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
        self.folder = sys.argv[1] if len(sys.argv) > 1 else self.DEFAULT_FOLDER
        self.config_path = os.path.join(self.folder, "config.json")
        self.patterns_top_dir = os.path.join(self.folder, "patrones_arriba")
        self.patterns_top2_dir = os.path.join(self.folder, "patrones_arriba2")
        self.patterns_middle_dir = os.path.join(self.folder, "patrones_medio")
        self.patterns_bottom_dir = os.path.join(self.folder, "patrones_abajo")
        
        # Mode settings
        self.REAL_TIME_MODE = False
        self.LOCAL_IMAGE_FOLDER = "/Users/nt/Library/CloudStorage/GoogleDrive-naoaki0107@gmail.com/マイドライブ/0_working/mizkan/capturas"
        
        # PLC related
        self.mc = None
        self.plc_connected = False
        
        # Image related
        self.patterns = self.IMAGE_EXTENSIONS
        self.image_list = []
        self.camera_capture = None
        self.current_image_index = 0
        self.local_images = []
        self.ok_to_process = True
        
        # ROI settings
        self.roi_top = None
        self.roi_top2 = None
        self.roi_middle = None
        self.roi_bottom = None
        self.roi_rows1 = None
        self.roi_rows2 = None
        self.x_minimum = self.X_MINIMO_DEFAULT
        self.max_distance_x = self.MAX_DISTANCIA_X_DEFAULT
        self.validation_threshold = self.UMBRAL_VALIDACION_DEFAULT
        self.max_1x = None
        self.max_2x = None
        self.max_3x = None
        
        # Global variables (same as original file)
        self.zone_max_delta = ("none", self.DELTA_DEFAULT)
        self.delta_max_mm = self.DELTA_DEFAULT
        self.zone_min_delta = ("none", self.DELTA_DEFAULT)
        self.delta_min_mm = self.DELTA_DEFAULT
        
        # Main loop variables
        self.i = self.I_DEFAULT
        self.interval_ms = self.INTERVALO_MS_DEFAULT
        self.last_time = time.time()
        self.automatic = False
        
        # Reference images
        self.reference_rows1 = {}
        self.reference_rows2 = {}
        self.roi_rows_x1 = {}
        self.roi_rows_x2 = {}
    
    def connect_plc(self):
        """PLC connection (same logic as original file)"""
        try:
            self.mc = pymcprotocol.Type3E()
            self.mc.connect(self.PLC_IP_ADDRESS, self.PLC_PORT)
            self.plc_connected = True
            print("✅ PLC connection successful")
        except Exception as e:
            print(f"⚠️ PLC connection failed: {e}")
            print("🔄 Skipping PLC communication and running in local mode")
            self.plc_connected = False
    
    def int32_to_words(self, n: int) -> list[int]:
        """Convert 32-bit signed integer to two WORDs (low, high) (same as original file)"""
        n &= self.BIT32_MASK  # 32-bit two's complement
        return [n & self.BIT16_MASK, (n >> self.BIT_SHIFT_16) & self.BIT16_MASK]
    
    def search_image_files(self):
        """Search for image files (same logic as original file)"""
        # Use LOCAL_IMAGE_FOLDER for local mode, otherwise use specified folder
        if not self.REAL_TIME_MODE:
            # Local mode: search for images from LOCAL_IMAGE_FOLDER
            for pattern in self.patterns:
                self.image_list.extend(glob.glob(os.path.join(self.LOCAL_IMAGE_FOLDER, pattern)))
            self.image_list.sort()
            
            if not self.image_list:
                print(f"❌ No image files found in {self.LOCAL_IMAGE_FOLDER}")
                print(f"📁 Verified path: {self.LOCAL_IMAGE_FOLDER}")
                print("💡 In local mode, please place image files in LOCAL_IMAGE_FOLDER")
                sys.exit(self.EXIT_ERROR)
        else:
            # Real-time mode: search for images from specified folder (for patterns)
            for pattern in self.patterns:
                self.image_list.extend(glob.glob(os.path.join(self.folder, pattern)))
            self.image_list.sort()
            
            if not self.image_list:
                print(f"❌ No se encontraron imágenes en {self.folder}")
                sys.exit(self.EXIT_ERROR)
    
    def initialize_camera_or_local(self):
        """Initialize camera or local images (same logic as original file)"""
        if self.REAL_TIME_MODE:
            # Real-time mode: IP camera connection
            username = self.CAMERA_USERNAME
            password = self.CAMERA_PASSWORD
            channel = self.CAMERA_STREAM
            camera_ip_url = f"rtsp://{username}:{password}@{self.CAMERA_IP_ADDRESS}:{self.RTSP_PORT}/{channel}"

            self.camera_capture = cv2.VideoCapture(camera_ip_url)
            #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            if not self.camera_capture.isOpened():
                print("❌ No se pudo abrir la cámara IP.")
                sys.exit(self.EXIT_ERROR)

            # Capture an image to initialize
            ret, frame = self.camera_capture.read()
            if not ret:
                print("❌ Could not capture image from camera.")
                sys.exit(self.EXIT_ERROR)
            
            print("🔴 Real-time mode: Getting images from IP camera")
        else:
            # Local mode: load image files from folder
            print(f"🟢 Local mode: Loading images from {self.LOCAL_IMAGE_FOLDER}")
            
            # Get list of local image files (already stored in image_list)
            self.local_images = self.image_list
            
            print(f"📁 Number of loaded images: {len(self.local_images)}")
            for img_path in self.local_images:
                print(f"  - {os.path.basename(img_path)}")
            
            # Dummy cap object (for local mode)
            self.camera_capture = None
            self.current_image_index = 0
            self.ok_to_process = True  # Process first image in local mode
    
    def load_or_setup_roi_settings(self):
        """Load ROI settings from config file or setup interactively if not exists"""
        if os.path.exists(self.config_path):
            config = self._load_config_from_file()
            self._apply_roi_settings(config)
            self._apply_detection_settings(config)
        else:
            self._setup_roi_settings_interactively()
    
    def _load_config_from_file(self) -> dict:
        """Load configuration from JSON file"""
        with open(self.config_path, "r") as f:
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
        self.x_minimum = config.get("x_minimo", self.X_MINIMO_DEFAULT)
        self.max_distance_x = config.get("max_distancia_x", self.MAX_DISTANCIA_X_DEFAULT)
        self.validation_threshold = config.get("umbral_validacion", self.UMBRAL_VALIDACION_DEFAULT)
        self.max_1x = config.get("max_1x", None)
        self.max_2x = config.get("max_2x", None)
        self.max_3x = config.get("max_3x", None)
    
    def _setup_roi_settings_interactively(self):
        """Setup ROI settings interactively when config file doesn't exist"""
        print("🖼️ Showing first image to define ROIs...")
        # Use LOCAL_IMAGE_FOLDER image for local mode
        if not self.REAL_TIME_MODE:
            img = cv2.imread(self.local_images[0])
        else:
            img = cv2.imread(self.image_list[0])
        
        if img is None:
            print("❌ Could not load initial image.")
            sys.exit(self.EXIT_ERROR)

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
            setattr(self, attr_name, cv2.selectROI(window_title, img, self.ROI_SELECT_FROM_CENTER, self.ROI_SELECT_FIXED_ASPECT))

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
        with open(self.config_path, "w") as f:
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
            color = self.COLOR_WHITE
        if font_scale is None:
            font_scale = self.FONT_SCALE_DEFAULT
        if thickness is None:
            thickness = self.LINE_THICKNESS_DEFAULT
        cv2.putText(img, text, (self.TEXT_X_OFFSET, y_position), self.FONT_FACE, font_scale, color, thickness)
    
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
                                           self.CLAHE_CLIP_LIMIT, 
                                           self.CLAHE_TILE_GRID_SIZE,
                                           self.GAUSSIAN_BLUR_KERNEL_SIZE, 
                                           self.GAUSSIAN_BLUR_SIGMA)
    
    def preprocess_image2(self, img):
        """Image preprocessing for row detection (same as original file)"""
        return self._preprocess_image_common(img, 
                                           self.CLAHE_CLIP_LIMIT_2, 
                                           self.CLAHE_TILE_GRID_SIZE_2,
                                           self.GAUSSIAN_BLUR_KERNEL_SIZE_2, 
                                           self.GAUSSIAN_BLUR_SIGMA_2)
    
    def load_patterns(self, folder, row):
        """Load multiple patterns (same as original file)"""
        patterns = []
        row_path = os.path.join(folder, f"fila{row}")
        # More efficient file extension handling
        for path in glob.glob(os.path.join(row_path, "*.jpg")) + glob.glob(os.path.join(row_path, "*.png")):
            img = cv2.imread(path, self.GRAYSCALE_READ_MODE)
            if img is not None:
                img_proc = self.preprocess_image(img)
                name = os.path.basename(path)
                patterns.append((img_proc, name))
        return patterns
    
    def calculate_progressive_roi_x(self, roi_rows, row, border_x=None):
        """Calculate ROI by row (same as original file)"""
        if border_x is None:
            border_x = self.BORDE_X_DEFAULT
            
        x, y, w, h = roi_rows
        usable_width = w - 2 * border_x
        step = (usable_width / self.NUM_FILAS)

        row_roi_width = int(step * self.ROI_WIDTH_MULTIPLIER)
        start_x = int(x + border_x + (step * (row - self.ROI_OFFSET)))

        return (start_x, y, row_roi_width, h)

    def _initialize_row_detection_parameters(self, roi_rows, border_x):
        """Initialize parameters for row detection"""
        x, y, w, h = roi_rows
        usable_width = w - 2 * border_x
        step = usable_width / self.NUM_SECTORS
        sector_width = int(step * self.SECTOR_WIDTH_MULTIPLIER)
        pattern_width = int(step * self.PATTERN_WIDTH_MULTIPLIER)
        return RowDetectionParams(x, y, w, h, step, sector_width, pattern_width)

    def _validate_pattern_against_f9(self, current_roi, f9_references, pattern_start_x, pattern_width, h, empty_threshold):
        """Validate detected pattern against F9 (no row) patterns"""
        is_valid = True
        for f9_pattern, _ in f9_references:
            f9_pattern_crop = f9_pattern[self.CROP_TOP_OFFSET:h-self.CROP_BOTTOM_OFFSET, max(0, pattern_start_x):pattern_start_x + pattern_width]
            if (current_roi.shape[0] >= f9_pattern_crop.shape[0] and current_roi.shape[1] >= f9_pattern_crop.shape[1]):
                res_f9 = cv2.matchTemplate(current_roi, f9_pattern_crop, cv2.TM_CCOEFF_NORMED)
                _, score_f9, _, _ = cv2.minMaxLoc(res_f9)
                if score_f9 > empty_threshold:
                    is_valid = False
                    break
        return is_valid

    def _process_pattern_matching(self, current_roi, current_pattern, row, threshold, f9_references, pattern_start_x, pattern_width, h, x, y):
        """Process pattern matching for a single pattern"""
        res = cv2.matchTemplate(current_roi, current_pattern, self.TEMPLATE_MATCHING_METHOD)
        _, score, _, _ = cv2.minMaxLoc(res)
        
        if row <= 2:
            final_threshold = self.FINAL_THRESHOLD_ROW_2
            empty_threshold = self.EMPTY_THRESHOLD_ROW_2
        else:
            final_threshold = threshold
            empty_threshold = self.EMPTY_THRESHOLD
            
        if score > final_threshold:  # Validation threshold
            # Validate against F9 patterns (no row)
            is_valid = self._validate_pattern_against_f9(current_roi, f9_references, pattern_start_x, pattern_width, h, empty_threshold)
            return is_valid, score, pattern_width
        return False, score, pattern_width

    def detect_rows_by_sectors(self, gray, roi_rows, references_by_rows, threshold=0.70, border_x=10):
        """
        Progressively evaluates if there are rows present from each sector (1 to 8).
        Returns the lowest row (closest to top) where detection occurs.
        """
        empty_threshold = self.EMPTY_THRESHOLD
        params = self._initialize_row_detection_parameters(roi_rows, border_x)
        x, y, w, h = params.x, params.y, params.w, params.h
        step, sector_width, pattern_width = params.step, params.sector_width, params.pattern_width

        best_row_detected = self.DEFAULT_ROW_COUNT  # Default: row 9 = no rows
        best_score = -1
        best_name = self.DEFAULT_NAME
        best_roi = (0, 0, 0, 0)

        # F9 references = no row
        f9_references = references_by_rows.get(self.DEFAULT_ROW_COUNT, [])

        for row in range(self.NUM_SECTORS, 0, -1):  # from 8 to 1
            #print("row:" + str(row))
            sector_start_x = int(border_x + (step * (row - self.ROW_OFFSET_1 - self.ROW_OFFSET_2)))
            pattern_start_x = int(border_x + (step * (row - self.ROW_OFFSET_3 - self.ROW_OFFSET_2)))
            current_roi = gray[0:h, max(0,sector_start_x):sector_start_x+sector_width]

            if row not in references_by_rows:
                #print("no references")
                continue

            for pattern, name in references_by_rows[row]:
                current_pattern = pattern[self.CROP_TOP_OFFSET:h-self.CROP_BOTTOM_OFFSET, max(0,pattern_start_x):pattern_start_x+pattern_width]

                if current_roi.shape[0] < current_pattern.shape[0] or current_roi.shape[1] < current_pattern.shape[1]:
                    print("Size error")
                    continue

                is_valid, score, pattern_width = self._process_pattern_matching(current_roi, current_pattern, row, threshold, f9_references, pattern_start_x, pattern_width, h, x, y)
                
                if is_valid and row < best_row_detected:
                    best_row_detected = row
                    best_score = score
                    best_name = name
                    best_roi = (max(0,pattern_start_x)+x, y,pattern_width,h)
                    #print(best_row_detected)
                    #print(best_score)
                    #print(best_name)
                break  # already detected a row in this zone

        return best_row_detected, best_score, best_name, best_roi

    def load_references_by_rows(self, base_folder, roi_rows):
        """Load row references (same as original file)"""
        references = {}
        roi_rows_final = {}
        for i in range(1, self.DEFAULT_ROW_COUNT + 1):
            folder = os.path.join(base_folder, f"F{i}")
            refs = []
            #roi_rows_x = calculate_progressive_roi_x(roi_rows,i)
            roi_rows_x = roi_rows
            for img_path in glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")):
                img = cv2.imread(img_path, 0)
                if img is not None:
                    ref_img = self.crop_roi_region(img, roi_rows_x)
                    #refs.append(ref_img)
                    img_proc = self.preprocess_image2(ref_img)
                    name = os.path.basename(img_path)
                    refs.append((img_proc, name))
            if refs:
                references[i] = refs
                roi_rows_final[i] = roi_rows_x
        return references, roi_rows_final

    def detect_best_pattern(self, roi, patterns, x_global_offset, left_weight=0.2, validation_threshold=0.8, x_maximum=None):
        """Detect best pattern (same as original file)"""
        results = []
        for pattern, name in patterns:
            if roi.shape[0] < pattern.shape[0] or roi.shape[1] < pattern.shape[1]:
                continue
            res = cv2.matchTemplate(roi, pattern, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            x_global = max_loc[0] + x_global_offset

            centerX = int((max_loc[0] + pattern.shape[1] // self.CENTER_DIVISOR) + x_global_offset)

            # Global position filters
            if centerX < self.x_minimum:
                continue
            if (x_maximum is not None) and (centerX > x_maximum):
                continue
            if (x_maximum is not None) and (centerX < (x_maximum - self.DISTANCE_FILTER_OFFSET)):
                continue
            # Match threshold filter
            if max_val < validation_threshold:
                continue

            x_norm = max_loc[0] / roi.shape[1]
            score = max_val - left_weight * x_norm
            center = (max_loc[self.ARRAY_FIRST_INDEX] + pattern.shape[1] // self.CENTER_DIVISOR, max_loc[self.ARRAY_SECOND_INDEX] + pattern.shape[0] // self.CENTER_DIVISOR)
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

    def _initialize_line_detection_parameters(self, number_of_rows, roi_rows_x):
        """Initialize parameters for line detection"""
        # Get ROI coordinates directly (no unnecessary tuple conversion)
        roi_top = self.roi_top
        roi_top2 = self.roi_top2
        roi_bottom = self.roi_bottom
        roi_middle = self.roi_middle

        # Get distance limits with safe array access
        distx_max1 = self.max_1x[number_of_rows - 1] if self.max_1x else None
        distx_max2 = self.max_2x[number_of_rows - 1] if self.max_2x else None
        distx_max3 = self.max_3x[number_of_rows - 1] if self.max_3x else None

        return LineDetectionParams(
            roi_top=roi_top,
            roi_top2=roi_top2,
            roi_bottom=roi_bottom,
            roi_middle=roi_middle,
            distance_limits=(distx_max1, distx_max2, distx_max3)
        )

    def _load_patterns_for_line_detection(self, number_of_rows):
        """Load patterns for line detection"""
        top_patterns = self.load_patterns(self.patterns_top_dir, number_of_rows)
        top2_patterns = self.load_patterns(self.patterns_top2_dir, number_of_rows)
        middle_patterns = self.load_patterns(self.patterns_middle_dir, number_of_rows)
        bottom_patterns = self.load_patterns(self.patterns_bottom_dir, number_of_rows)

        if not top_patterns or not top2_patterns or not bottom_patterns or not middle_patterns:
            return None, None, None, None

        return top_patterns, top2_patterns, middle_patterns, bottom_patterns

    def _detect_patterns_in_rois(self, gray, detection_params):
        """Detect patterns in ROIs"""
        # Extract ROI coordinates with meaningful names
        roi_coords = detection_params.roi_coordinates
        
        # Extract patterns and parameters
        top_patterns, top2_patterns, middle_patterns, bottom_patterns = detection_params.patterns
        validation_threshold = detection_params.validation_threshold
        distx_max1, distx_max2, distx_max3 = detection_params.distance_limits
        
        # Extract ROIs from image using structured approach
        roi1 = self.extract_roi_using_rectangle(gray, roi_coords.top)
        roi12 = self.extract_roi_using_rectangle(gray, roi_coords.top2)
        roi2 = self.extract_roi_using_rectangle(gray, roi_coords.bottom)
        roi3 = self.extract_roi_using_rectangle(gray, roi_coords.middle)

        # Detect patterns
        results1 = self.detect_best_pattern(roi1, top_patterns, roi_coords.top.x, self.LEFT_WEIGHT, validation_threshold, distx_max1)
        if not results1:
            results1 = self.detect_best_pattern(roi12, top2_patterns, roi_coords.top2.x, self.LEFT_WEIGHT, validation_threshold, distx_max1)

        results2 = self.detect_best_pattern(roi2, bottom_patterns, roi_coords.bottom.x, self.LEFT_WEIGHT, validation_threshold, distx_max3)
        results3 = self.detect_best_pattern(roi3, middle_patterns, roi_coords.middle.x, self.LEFT_WEIGHT, validation_threshold/self.CENTER_DIVISOR, distx_max2)

        return results1, results2, results3

    def _generate_valid_combos(self, results1, results2, x1, x2):
        """Generate valid combinations from top and bottom results"""
        combos = []
        for res1 in results1:
            for res2 in results2:
                x_distance = abs((x1 + res1[0][0]) - (x2 + res2[0][0]))
                if x_distance <= self.max_distance_x:
                    combo_score = res1[2] + res2[2]
                    combos.append((res1[0], res1[1], res2[0], res2[1], combo_score, (res1[3], res2[3])))

        if not combos:
            raise Exception("No valid combos found")

        return combos

    def _find_best_middle_point(self, results3, x3, y3, pt1, pt2):
        """Find the best middle point from results3"""
        ideal_middle = ((pt1[0] + pt2[0]) // self.CENTER_DIVISOR, (pt1[1] + pt2[1]) // self.CENTER_DIVISOR)

        best_loc3 = None
        best_score_loc3 = -np.inf

        for res3 in results3:
            pt3 = (x3 + res3[0][0], y3 + res3[0][1])
            distance = np.linalg.norm(np.array(pt3) - np.array(ideal_middle))
            if distance > self.DISTANCE_THRESHOLD:
                continue
            dist_norm = distance / self.DISTANCE_NORMALIZATION
            score = res3[1] - self.DISTANCE_WEIGHT * dist_norm
            if score > best_score_loc3:
                best_score_loc3 = score
                best_loc3 = res3

        return best_loc3

    def _draw_detection_lines(self, img, best_loc3, pt1, pt2, x3, y3):
        """Draw detection lines on the image"""
        if best_loc3:
            pt3 = (x3 + best_loc3[0][0], y3 + best_loc3[0][1])
            detection_success_color = self.COLOR_RED
            cv2.line(img, pt1, pt3, detection_success_color, self.LINE_THICKNESS_THICK)
            cv2.line(img, pt3, pt2, detection_success_color, self.LINE_THICKNESS_THICK)
            middle_name = best_loc3[3]
        else:
            detection_fallback_color = self.COLOR_MAGENTA
            cv2.line(img, pt1, pt2, detection_fallback_color, self.LINE_THICKNESS_THICK)
            middle_name = "N/A"
        
        return middle_name

    def _display_detection_results(self, img, number_of_rows):
        """Display detection results on image"""
        print(f"📏 Maximum deviation: {self.delta_max_mm:.2f} mm in zone: {self.zone_max_delta[0]} (Row {number_of_rows})")
        self._draw_text(img, f"Delta max: {self.delta_max_mm:.2f} mm - {self.zone_max_delta[0]}", self.TEXT_Y_DELTA_MAX, self.COLOR_CYAN, self.FONT_SCALE_LARGE, self.LINE_THICKNESS_DEFAULT)
        print(f"📏 Minimum deviation: {self.delta_min_mm:.2f} mm in zone: {self.zone_min_delta[0]} (Row {number_of_rows})")
        self._draw_text(img, f"Delta min: {self.delta_min_mm:.2f} mm - {self.zone_min_delta[0]}", self.TEXT_Y_DELTA_MIN, self.COLOR_CYAN, self.FONT_SCALE_LARGE, self.LINE_THICKNESS_DEFAULT)

    def detect_and_draw_lines(self, img, gray, number_of_rows, roi_rows_x, validation_threshold, fallback=True):
        """
        Attempts to detect points and draw lines for a row. If it fails and fallback is enabled, tries with the next row.
        Returns True if successful, False if not.
        """
        self.zone_max_delta = ("none", 0.0)
        self.delta_max_mm = 0.0
        self.zone_min_delta = ("none", 0.0)
        self.delta_min_mm = 0.0

        try:
            # Initialize parameters
            line_params = self._initialize_line_detection_parameters(number_of_rows, roi_rows_x)
            distx_max1, distx_max2, distx_max3 = line_params.distance_limits

            # Load patterns
            top_patterns, top2_patterns, middle_patterns, bottom_patterns = self._load_patterns_for_line_detection(number_of_rows)
            if top_patterns is None:
                return -1

            # Prepare detection parameters
            roi_coords = ROICoordinates(
                top=line_params.roi_top,
                top2=line_params.roi_top2,
                bottom=line_params.roi_bottom,
                middle=line_params.roi_middle
            )
            detection_params = PatternDetectionParams(
                roi_coordinates=roi_coords,
                patterns=(top_patterns, top2_patterns, middle_patterns, bottom_patterns),
                validation_threshold=validation_threshold,
                distance_limits=(distx_max1, distx_max2, distx_max3)
            )
            
            # Detect patterns in ROIs
            results1, results2, results3 = self._detect_patterns_in_rois(gray, detection_params)

            # Generate valid combos
            combos = self._generate_valid_combos(results1, results2, roi_coords.top.x, roi_coords.bottom.x)

            # Find best combo
            best_combo = max(combos, key=lambda c: c[4])
            loc1, val1, loc2, val2, combo_score, combo_names = best_combo

            pt1 = (roi_coords.top.x + loc1[0], roi_coords.top.y + loc1[1])
            pt2 = (roi_coords.bottom.x + loc2[0], roi_coords.bottom.y + loc2[1])

            # Find best middle point
            best_loc3 = self._find_best_middle_point(results3, roi_coords.middle.x, roi_coords.middle.y, pt1, pt2)

            # Draw detection lines
            middle_name = self._draw_detection_lines(img, best_loc3, pt1, pt2, roi_coords.middle.x, roi_coords.middle.y)

            top_name, bottom_name = combo_names

            # Display results
            self._display_detection_results(img, number_of_rows)

            return number_of_rows

        except Exception as e:
            if fallback and number_of_rows < self.NUM_SECTORS:
                print(f"⚠️ Failed row {number_of_rows}, trying row {number_of_rows + 1}")
                return self.detect_and_draw_lines(img, gray, number_of_rows + 1, roi_rows_x, validation_threshold, False)

            return -1

    def _initialize_main_loop(self):
        """Initialize main loop components - 完全に同じロジック"""
        # Initialize references
        self.reference_rows1, self.roi_rows_x1 = self.load_references_by_rows(self.folder, self.roi_rows1)
        self.reference_rows2, self.roi_rows_x2 = self.load_references_by_rows(self.folder, self.roi_rows2)

        # Create ROI rectangles with meaningful names
        top_roi = ROIRectangle(*self.roi_top)
        top2_roi = ROIRectangle(*self.roi_top2)
        bottom_roi = ROIRectangle(*self.roi_bottom)
        middle_roi = ROIRectangle(*self.roi_middle)
        rows1_roi = ROIRectangle(*self.roi_rows1)
        rows2_roi = ROIRectangle(*self.roi_rows2)
        detection_area_roi = ROIRectangle(0, 0, 0, 0)

        # Group all ROIs into a structured container
        roi_coords = MainLoopROIs(
            top=top_roi,
            top2=top2_roi,
            bottom=bottom_roi,
            middle=middle_roi,
            rows1=rows1_roi,
            rows2=rows2_roi,
            detection_area=detection_area_roi
        )

        cv2.namedWindow(self.RESULT_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.RESULT_WINDOW_NAME, self.WINDOW_WIDTH, self.WINDOW_HEIGHT)

        return roi_coords

    def _handle_key_input_and_timing(self):
        """Handle key input and timing - 完全に同じロジック"""
        key = cv2.waitKey(self.KEY_WAIT_TIME_MS) & self.BIT_MASK_8BIT
        if key == ord('q'):
            return True  # Exit signal
            
        # Local mode space key processing
        if not self.REAL_TIME_MODE and key == ord(' '):
            self.current_image_index = (self.current_image_index + 1) % len(self.local_images)
            print(f"📷 Moving to next image: {os.path.basename(self.local_images[self.current_image_index])} ({self.current_image_index + 1}/{len(self.local_images)})")
            self.ok_to_process = True  # Process image when changed
        
        # Real-time mode camera reading
        if self.REAL_TIME_MODE:
            _ = self.camera_capture.read()

        current_time = time.time()
        if (current_time - self.last_time) * self.SECONDS_TO_MILLISECONDS >= self.interval_ms:
            self.last_time = current_time
            self.ok_to_process = True
            
        return False  # Continue signal

    def _read_plc_and_check_processing(self):
        """Read PLC and check processing conditions - 完全に同じロジック"""
        try:
            d28 = self.PLC_DEFAULT_VALUE  # Default value
            if self.plc_connected:
                try:
                    d28 = self.mc.batchread_wordunits(headdevice=self.PLC_DEVICE_D28, readsize=1)[self.ARRAY_FIRST_INDEX]
                except Exception as e:
                    print(f"⚠️ PLC read error: {e}")
                    d28 = self.PLC_DEFAULT_VALUE  # Use default value on error
            
            if (d28 == self.PLC_DEFAULT_VALUE) or (self.ok_to_process and self.automatic):
                print(f"🔍 Inspection request ({self.PLC_DEVICE_D28} = {self.PLC_DEFAULT_VALUE})")
                return True
            return False
        except Exception as e:
            print(f"⚠️ PLC processing error: {e}")
            return False

    def _acquire_image(self):
        """Acquire image from camera or local files - 完全に同じロジック"""
        # Get image (real-time/local)
        if self.REAL_TIME_MODE:
            ret, img = self.camera_capture.read()
            if not ret:
                print("❌ Could not capture image.")
                return None
        else:
            try:
                img_path = self.local_images[self.current_image_index]
                img = cv2.imread(img_path)
                if img is None:
                    print(f"❌ Failed to load image: {img_path}")
                    return None
                print(f"📷 Processing: {os.path.basename(img_path)}")
            except Exception as e:
                print(f"❌ Local image access error: {e}")
                return None
        return img

    def _process_image_and_detection(self, img):
        """Process image and perform detection - 完全に同じロジック"""
        # Image preprocessing
        gray = self.preprocess_image(img)
        gray2 = self.preprocess_image2(img)
        img_filas1 = self.crop_roi_region(gray2, self.roi_rows1)
        img_filas2 = self.crop_roi_region(gray2, self.roi_rows2)
        
        # Row detection
        number_of_rows1, row_score1, row_pattern1, row_roi1 = self.detect_rows_by_sectors(img_filas1, self.roi_rows1, self.reference_rows1)
        number_of_rows2, row_score2, row_pattern2, row_roi2 = self.detect_rows_by_sectors(img_filas2, self.roi_rows2, self.reference_rows2)

        if number_of_rows1 < number_of_rows2:
            number_of_rows = number_of_rows1
            row_score = row_score1
            row_pattern = row_pattern1
            roi_rows_x = self.roi_rows_x1
        else:
            number_of_rows = number_of_rows2
            row_score = row_score2
            row_pattern = row_pattern2
            roi_rows_x = self.roi_rows_x2

        # Line detection and position measurement
        if 2 < number_of_rows < self.DEFAULT_ROW_COUNT:
            corrected_row = self.detect_and_draw_lines(img, gray, number_of_rows, roi_rows_x, self.validation_threshold)
            if corrected_row > 0:
                number_of_rows = corrected_row
            else:
                print(f"❌ Could not draw lines even with fallback from row {number_of_rows}")

        # Initialize display variables
        top_name = "N/A"
        middle_name = "N/A"
        bottom_name = "N/A"
        val1 = val2 = val3 = -1

        # Create meaningfully grouped data objects
        main_result = MainDetectionResult(number_of_rows, row_score, row_pattern, roi_rows_x)
        roi_info = ROIDetectionInfo(ROIRectangle(*row_roi1), ROIRectangle(*row_roi2))
        detailed_results = DetailedRowResults(
            RowDetectionResult(number_of_rows1, row_score1, row_pattern1),
            RowDetectionResult(number_of_rows2, row_score2, row_pattern2)
        )
        display_info = DisplayInfo(top_name, middle_name, bottom_name, val1, val2, val3)
        
        # Return meaningfully grouped results
        return DetectionResults(main_result, roi_info, detailed_results, display_info)





    def _process_results_and_display(self, img, roi_coords, detection_results: DetectionResults):
        """Process results and display on image with meaningfully grouped data - 完全に同じロジック"""
        # Result interpretation using data class methods
        result_text1 = detection_results.detailed_results.row1_results.get_result_text(self.DEFAULT_ROW_COUNT)
        result_text2 = detection_results.detailed_results.row2_results.get_result_text(self.DEFAULT_ROW_COUNT)
        result_text = detection_results.main_result.get_result_text(self.DEFAULT_ROW_COUNT)

        # PLC result transmission using data class methods
        if self.plc_connected:
            try:
                if detection_results.main_result.is_valid_detection(self.DEFAULT_ROW_COUNT):
                    send_value = int(round(-self.delta_min_mm * self.MM_TO_HUNDREDTHS))  # 1/100 mm unit
                    words = self.int32_to_words(send_value)
                    self.mc.batchwrite_wordunits(headdevice=self.PLC_DEVICE_D29, values=words)
                    self.mc.batchwrite_wordunits(headdevice=self.PLC_DEVICE_D14, values=[detection_results.main_result.number_of_rows])
                    self.mc.batchwrite_wordunits(headdevice=self.PLC_DEVICE_D28, values=[self.PLC_SUCCESS_VALUE])
                else:
                    words = self.int32_to_words(0)
                    self.mc.batchwrite_wordunits(headdevice=self.PLC_DEVICE_D29, values=words)
                    self.mc.batchwrite_wordunits(headdevice=self.PLC_DEVICE_D14, values=[0])
                    self.mc.batchwrite_wordunits(headdevice=self.PLC_DEVICE_D28, values=[self.PLC_ERROR_VALUE])
            except Exception as e:
                print(f"⚠️ PLC write error: {e}")
        else:
            # Local display when PLC not connected
            if detection_results.main_result.is_valid_detection(self.DEFAULT_ROW_COUNT):
                send_value = int(round(-self.delta_min_mm * self.MM_TO_HUNDREDTHS))
                print(f"📊 Local result: rows={detection_results.main_result.number_of_rows}, deviation={send_value/self.MM_TO_HUNDREDTHS:.2f}mm")
            else:
                print(f"📊 Local result: rows=0, deviation=0.00mm")

        # Display on image
        mode_text = "REAL-TIME" if self.REAL_TIME_MODE else "LOCAL"
        self._draw_text(img, f"Mode: {mode_text}", self.TEXT_Y_MODE, self.COLOR_WHITE, self.FONT_SCALE_DEFAULT, self.LINE_THICKNESS_DEFAULT)
        
        if not self.REAL_TIME_MODE:
            filename = os.path.basename(self.local_images[self.current_image_index])
            self._draw_text(img, f"File: {filename}", self.TEXT_Y_FILENAME, self.COLOR_WHITE, self.FONT_SCALE_SMALL, self.LINE_THICKNESS_DEFAULT)
            self._draw_text(img, f"Space: Next Image", self.TEXT_Y_INFO, self.COLOR_YELLOW, self.FONT_SCALE_SMALL, self.LINE_THICKNESS_DEFAULT)
        
        self._draw_text(img, detection_results.detailed_results.row1_results.get_display_text(self.DEFAULT_ROW_COUNT),
                       self.TEXT_Y_ROW1, self.COLOR_GREEN, self.FONT_SCALE_LARGE, self.LINE_THICKNESS_DEFAULT)
        self._draw_text(img, detection_results.detailed_results.row2_results.get_display_text(self.DEFAULT_ROW_COUNT),
                       self.TEXT_Y_ROW2, self.COLOR_GREEN, self.FONT_SCALE_LARGE, self.LINE_THICKNESS_DEFAULT)
        
        h_img, w_img = img.shape[:2]
        (text_w, text_h), _ = cv2.getTextSize(result_text, self.FONT_FACE, self.FONT_SCALE_EXTRA_LARGE, self.LINE_THICKNESS_VERY_THICK)
        x_text = (w_img - text_w) // self.CENTER_DIVISOR
        cv2.putText(img, f"{result_text}",
                    (x_text, self.TEXT_Y_CENTER), self.FONT_FACE, self.FONT_SCALE_EXTRA_LARGE, self.COLOR_GREEN, self.LINE_THICKNESS_VERY_THICK)
        
        self._draw_text(img, detection_results.display_info.get_probability_text("top"), 
                       self.TEXT_Y_PROB_TOP, self.COLOR_YELLOW, self.FONT_SCALE_LARGE, self.LINE_THICKNESS_DEFAULT)
        self._draw_text(img, detection_results.display_info.get_probability_text("middle"), 
                       self.TEXT_Y_PROB_MIDDLE, self.COLOR_YELLOW, self.FONT_SCALE_LARGE, self.LINE_THICKNESS_DEFAULT)
        self._draw_text(img, detection_results.display_info.get_probability_text("bottom"), 
                       self.TEXT_Y_PROB_BOTTOM, self.COLOR_YELLOW, self.FONT_SCALE_LARGE, self.LINE_THICKNESS_DEFAULT)

        # ROI rectangle display
        self.draw_roi_rectangle_on_image(img, roi_coords.top, self.ROI_TOP_COLOR, self.LINE_THICKNESS_DEFAULT)  # Top ROI (blue)
        self.draw_roi_rectangle_on_image(img, roi_coords.top2, self.ROI_TOP_ALT_COLOR, self.LINE_THICKNESS_DEFAULT)  # Alternative top ROI (orange)
        self.draw_roi_rectangle_on_image(img, roi_coords.middle, self.ROI_MIDDLE_COLOR, self.LINE_THICKNESS_DEFAULT)  # Middle ROI (yellow)
        self.draw_roi_rectangle_on_image(img, roi_coords.bottom, self.ROI_BOTTOM_COLOR, self.LINE_THICKNESS_DEFAULT)  # Bottom ROI (green)
        roi1_coords = detection_results.roi_info.get_roi1_coordinates()
        roi2_coords = detection_results.roi_info.get_roi2_coordinates()
        cv2.rectangle(img, (roi1_coords[0], roi1_coords[1]), (roi1_coords[2], roi1_coords[3]), 
                      self.ROI_DETECTION_COLOR, self.LINE_THICKNESS_THICK)  # Row detection ROI1 (purple)
        cv2.rectangle(img, (roi2_coords[0], roi2_coords[1]), (roi2_coords[2], roi2_coords[3]), 
                      self.ROI_DETECTION_COLOR, self.LINE_THICKNESS_THICK)  # Row detection ROI2 (purple)
        self.draw_roi_rectangle_on_image(img, roi_coords.detection_area, self.ROI_DETECTION_AREA_COLOR, self.LINE_THICKNESS_THICK)  # Detection area (black)

        cv2.imshow(self.RESULT_WINDOW_NAME, img)

    def run_main_loop(self):
        """Main processing loop (same as original file)"""
        roi_coords = self._initialize_main_loop()

        while True:
            if self._handle_key_input_and_timing():
                break

            should_process = self._read_plc_and_check_processing()
            if should_process:
                img = self._acquire_image()
                if img is None:
                    continue

                detection_results = self._process_image_and_detection(img)
                if detection_results is None:
                    continue

                self._process_results_and_display(img, roi_coords, detection_results)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    program = StationProgram()
    program.connect_plc()
    program.search_image_files()
    program.initialize_camera_or_local()
    program.load_or_setup_roi_settings()
    program.run_main_loop() 
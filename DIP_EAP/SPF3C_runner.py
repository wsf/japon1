# -*- coding: utf-8 -*-
"""
StationProgramFilas3Classified.py - Industrial Image Processing System (Classified Version)

Fully classified version of the original StationProgramFilas3-local.py.
"""

import cv2
import glob
import sys
import os
import time
# Data structures are now handled by individual component modules
from SPF3C_config_manager import ConfigManager
from SPF3C_plc_communicator import PLCCommunicator
from SPF3C_image_processor import ImageProcessor
from SPF3C_roi_manager import ROIManager
from SPF3C_detection_engine import DetectionEngine
from SPF3C_display_manager import DisplayManager
from SPF3C_main_controller import MainController




class StationProgram:
    """産業用画像処理システムのメインクラス"""
    
    def __init__(self):
        """Initialization"""
        # Initialize configuration manager
        self.config = ConfigManager()
        
        # Create aliases for backward compatibility
        self._create_config_aliases()
        
        # PLC related
        self.plc_communicator = PLCCommunicator(self.config)
        
        # Image processing
        self.image_processor = ImageProcessor(self.config)
        
        # ROI management
        self.roi_manager = ROIManager(self.config)
        
        # Detection engine
        self.detection_engine = DetectionEngine(self.config, self.image_processor, self.roi_manager)
        
        # Display manager
        self.display_manager = DisplayManager(self.config, self.image_processor)
        
        # Main controller
        self.main_controller = MainController(self.config, self.plc_communicator, self.image_processor, self.roi_manager, self.detection_engine, self.display_manager)
        
        # Image related
        self.patterns = self.IMAGE_EXTENSIONS
        self.image_list = []
        self.camera_capture = None
        self.local_images = []
        
        # Create ROI aliases for backward compatibility
        self._update_roi_aliases()
        
        # Initialize configuration aliases
        self._create_config_aliases()
        
        # Main loop variables
        self.interval_ms = self.INTERVALO_MS_DEFAULT
        self.automatic = False
        

    

    
    def _create_config_aliases(self):
        """Create only the aliases that are actually used in this file"""
        # Camera settings (used in initialize_camera_or_local)
        self.CAMERA_IP_ADDRESS = self.config.CAMERA_IP_ADDRESS
        self.RTSP_PORT = self.config.RTSP_PORT
        self.CAMERA_USERNAME = self.config.CAMERA_USERNAME
        self.CAMERA_PASSWORD = self.config.CAMERA_PASSWORD
        self.CAMERA_STREAM = self.config.CAMERA_STREAM
        
        # Default values (used in __init__)
        self.DELTA_DEFAULT = self.config.DELTA_DEFAULT
        self.I_DEFAULT = self.config.I_DEFAULT
        self.INTERVALO_MS_DEFAULT = self.config.INTERVALO_MS_DEFAULT
        
        # File extensions (used in search_image_files)
        self.IMAGE_EXTENSIONS = self.config.IMAGE_EXTENSIONS
        
        # Exit codes (used in search_image_files and initialize_camera_or_local)
        self.EXIT_ERROR = self.config.EXIT_ERROR
        
        # Define global variables from original file as class variables
        self.folder = self.config.get_folder()
        
        # Mode settings
        self.REAL_TIME_MODE = self.config.is_real_time_mode()
        self.LOCAL_IMAGE_FOLDER = self.config.get_local_image_folder()
    
    def connect_plc(self):
        """PLC connection (same logic as original file)"""
        self.plc_communicator.connect_plc()
    

    
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
    
    def load_or_setup_roi_settings(self):
        """Load ROI settings from config file or setup interactively if not exists"""
        self.roi_manager.load_or_setup_roi_settings()
        # Update aliases after loading settings
        self._update_roi_aliases()
    
    def _update_roi_aliases(self):
        """Update ROI aliases after settings change"""
        # Only the ROI settings that are used in run_main_loop
        self.roi_rows1 = self.roi_manager.roi_rows1
        self.roi_rows2 = self.roi_manager.roi_rows2
    

    def run_main_loop(self):
        """Main processing loop using main controller"""
        # Set state variables in main controller before running
        self.main_controller.set_main_program_state(
            self.local_images,
            self.camera_capture,
            self.REAL_TIME_MODE,
            self.automatic,
            self.interval_ms
        )
        self.main_controller.run_main_loop(self.folder, self.roi_rows1, self.roi_rows2)


if __name__ == "__main__":
    program = StationProgram()
    program.connect_plc()
    program.search_image_files()
    program.initialize_camera_or_local()
    program.load_or_setup_roi_settings()
    program.run_main_loop() 
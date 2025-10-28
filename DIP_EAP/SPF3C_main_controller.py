# -*- coding: utf-8 -*-
"""
SPF3C_main_controller.py - Main controller for industrial image processing system

Contains the main loop control and component integration logic.
"""

import cv2
import time
import os
from SPF3C_data_structures import MainDetectionResult, ROIDetectionInfo, DetailedRowResults, DisplayInfo, DetectionResults, RowDetectionResult, ROIRectangle


class MainController:
    """Main controller class for the industrial image processing system"""
    
    def __init__(self, config_manager, plc_communicator, image_processor, roi_manager, detection_engine, display_manager):
        """Initialize main controller"""
        self.config = config_manager
        self.plc_communicator = plc_communicator
        self.image_processor = image_processor
        self.roi_manager = roi_manager
        self.detection_engine = detection_engine
        self.display_manager = display_manager
        
        # Main loop state variables
        self.reference_rows1 = None
        self.reference_rows2 = None
        self.roi_rows_x1 = None
        self.roi_rows_x2 = None
        self.last_time = time.time()
        self.ok_to_process = True
        self.current_image_index = 0
        self.local_images = []
        self.camera_capture = None
        self.REAL_TIME_MODE = False
        self.automatic = True
        self.interval_ms = 1000
        self.validation_threshold = 0.8
        
        # Global variables for detection results (same as original file)
        self.zone_max_delta = ("none", self.config.DELTA_DEFAULT)
        self.delta_max_mm = self.config.DELTA_DEFAULT
        self.zone_min_delta = ("none", self.config.DELTA_DEFAULT)
        self.delta_min_mm = self.config.DELTA_DEFAULT
        

    
    def set_main_program_state(self, local_images, camera_capture, real_time_mode, automatic, interval_ms):
        """Set state variables from main program"""
        self.local_images = local_images
        self.camera_capture = camera_capture
        self.REAL_TIME_MODE = real_time_mode
        self.automatic = automatic
        self.interval_ms = interval_ms
    
    def initialize_main_loop(self, folder, roi_rows1, roi_rows2):
        """Initialize main loop components"""
        # Initialize references
        self.reference_rows1, self.roi_rows_x1 = self.detection_engine.load_references_by_rows(folder, roi_rows1)
        self.reference_rows2, self.roi_rows_x2 = self.detection_engine.load_references_by_rows(folder, roi_rows2)

        # Create ROI rectangles with meaningful names
        from SPF3C_data_structures import ROIRectangle, MainLoopROIs
        top_roi = ROIRectangle(*self.roi_manager.roi_top)
        top2_roi = ROIRectangle(*self.roi_manager.roi_top2)
        bottom_roi = ROIRectangle(*self.roi_manager.roi_bottom)
        middle_roi = ROIRectangle(*self.roi_manager.roi_middle)
        rows1_roi = ROIRectangle(*roi_rows1)
        rows2_roi = ROIRectangle(*roi_rows2)
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

        self.display_manager.create_display_window(self.config.RESULT_WINDOW_NAME)
        cv2.resizeWindow(self.config.RESULT_WINDOW_NAME, self.config.WINDOW_WIDTH, self.config.WINDOW_HEIGHT)

        return roi_coords
    
    def handle_key_input_and_timing(self):
        """Handle key input and timing"""
        key = cv2.waitKey(self.config.KEY_WAIT_TIME_MS) & self.config.BIT_MASK_8BIT
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
        if (current_time - self.last_time) * self.config.SECONDS_TO_MILLISECONDS >= self.interval_ms:
            self.last_time = current_time
            self.ok_to_process = True
            
        return False  # Continue signal
    
    def read_plc_and_check_processing(self):
        """Read PLC and check processing conditions"""
        try:
            d28 = self.plc_communicator.read_plc_data()
            
            if (d28 == self.config.PLC_DEFAULT_VALUE) or (self.ok_to_process and self.automatic):
                print(f"🔍 Inspection request ({self.config.PLC_DEVICE_D28} = {self.config.PLC_DEFAULT_VALUE})")
                return True
            return False
        except Exception as e:
            print(f"⚠️ PLC processing error: {e}")
            return False
    
    def acquire_image(self):
        """Acquire image from camera or local files"""
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
    
    def process_image_and_detection(self, img):
        """Process image and perform detection"""
        # Image preprocessing
        gray = self.image_processor.preprocess_image(img)
        gray2 = self.image_processor.preprocess_image2(img)
        img_filas1 = self.image_processor.crop_roi_region(gray2, self.roi_manager.roi_rows1)
        img_filas2 = self.image_processor.crop_roi_region(gray2, self.roi_manager.roi_rows2)
        
        # Row detection
        number_of_rows1, row_score1, row_pattern1, row_roi1 = self.detection_engine.detect_rows_by_sectors(img_filas1, self.roi_manager.roi_rows1, self.reference_rows1)
        number_of_rows2, row_score2, row_pattern2, row_roi2 = self.detection_engine.detect_rows_by_sectors(img_filas2, self.roi_manager.roi_rows2, self.reference_rows2)

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
        if 2 < number_of_rows < self.config.DEFAULT_ROW_COUNT:
            detection_result = self.detection_engine.detect_and_draw_lines(img, gray, number_of_rows, roi_rows_x, self.validation_threshold)
            if isinstance(detection_result, dict) and detection_result.get('number_of_rows', -1) > 0:
                number_of_rows = detection_result['number_of_rows']
                # Update detection result variables from detection engine
                self.zone_max_delta = self.detection_engine.zone_max_delta
                self.delta_max_mm = self.detection_engine.delta_max_mm
                self.zone_min_delta = self.detection_engine.zone_min_delta
                self.delta_min_mm = self.detection_engine.delta_min_mm
                # Update display values
                top_name = detection_result.get('top_name', 'N/A')
                middle_name = detection_result.get('middle_name', 'N/A')
                bottom_name = detection_result.get('bottom_name', 'N/A')
                val1 = detection_result.get('val1', -1)
                val2 = detection_result.get('val2', -1)
                val3 = detection_result.get('val3', -1)
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

        return DetectionResults(main_result, roi_info, detailed_results, display_info)
    
    def process_results_and_display(self, img, roi_coords, detection_results: DetectionResults):
        """Process results and display on image"""
        # Result interpretation using data class methods
        result_text1 = detection_results.detailed_results.row1_results.get_result_text(self.config.DEFAULT_ROW_COUNT)
        result_text2 = detection_results.detailed_results.row2_results.get_result_text(self.config.DEFAULT_ROW_COUNT)
        result_text = detection_results.main_result.get_result_text(self.config.DEFAULT_ROW_COUNT)

        # PLC result transmission using data class methods
        if self.plc_communicator.plc_connected:
            try:
                if detection_results.main_result.is_valid_detection(self.config.DEFAULT_ROW_COUNT):
                    send_value = int(round(-self.delta_min_mm * self.config.MM_TO_HUNDREDTHS))  # 1/100 mm unit
                    self.plc_communicator.write_plc_data(send_value, detection_results.main_result.number_of_rows, True)
                else:
                    self.plc_communicator.write_plc_data(0, 0, False)
            except Exception as e:
                print(f"⚠️ PLC write error: {e}")
        else:
            # Local display when PLC not connected
            if detection_results.main_result.is_valid_detection(self.config.DEFAULT_ROW_COUNT):
                send_value = int(round(-self.delta_min_mm * self.config.MM_TO_HUNDREDTHS))
                print(f"📊 Local result: rows={detection_results.main_result.number_of_rows}, deviation={send_value/self.config.MM_TO_HUNDREDTHS:.2f}mm")
            else:
                print(f"📊 Local result: rows=0, deviation=0.00mm")

        # Display on image
        mode_text = "REAL-TIME" if self.REAL_TIME_MODE else "LOCAL"
        self.display_manager._draw_text(img, f"Mode: {mode_text}", self.config.TEXT_Y_MODE, self.config.COLOR_WHITE, self.config.FONT_SCALE_DEFAULT, self.config.LINE_THICKNESS_DEFAULT)
        
        if not self.REAL_TIME_MODE:
            filename = os.path.basename(self.local_images[self.current_image_index])
            self.display_manager._draw_text(img, f"File: {filename}", self.config.TEXT_Y_FILENAME, self.config.COLOR_WHITE, self.config.FONT_SCALE_SMALL, self.config.LINE_THICKNESS_DEFAULT)
            self.display_manager._draw_text(img, f"Space: Next Image", self.config.TEXT_Y_INFO, self.config.COLOR_YELLOW, self.config.FONT_SCALE_SMALL, self.config.LINE_THICKNESS_DEFAULT)
        
        self.display_manager._draw_text(img, detection_results.detailed_results.row1_results.get_display_text(self.config.DEFAULT_ROW_COUNT),
                       self.config.TEXT_Y_ROW1, self.config.COLOR_GREEN, self.config.FONT_SCALE_LARGE, self.config.LINE_THICKNESS_DEFAULT)
        self.display_manager._draw_text(img, detection_results.detailed_results.row2_results.get_display_text(self.config.DEFAULT_ROW_COUNT),
                       self.config.TEXT_Y_ROW2, self.config.COLOR_GREEN, self.config.FONT_SCALE_LARGE, self.config.LINE_THICKNESS_DEFAULT)
        
        h_img, w_img = img.shape[:2]
        (text_w, text_h), _ = cv2.getTextSize(result_text, self.config.FONT_FACE, self.config.FONT_SCALE_EXTRA_LARGE, self.config.LINE_THICKNESS_VERY_THICK)
        x_text = (w_img - text_w) // self.config.CENTER_DIVISOR
        cv2.putText(img, f"{result_text}",
                    (x_text, self.config.TEXT_Y_CENTER), self.config.FONT_FACE, self.config.FONT_SCALE_EXTRA_LARGE, self.config.COLOR_GREEN, self.config.LINE_THICKNESS_VERY_THICK)
        
        self.display_manager._draw_text(img, detection_results.display_info.get_probability_text("top"), 
                       self.config.TEXT_Y_PROB_TOP, self.config.COLOR_YELLOW, self.config.FONT_SCALE_LARGE, self.config.LINE_THICKNESS_DEFAULT)
        self.display_manager._draw_text(img, detection_results.display_info.get_probability_text("middle"), 
                       self.config.TEXT_Y_PROB_MIDDLE, self.config.COLOR_YELLOW, self.config.FONT_SCALE_LARGE, self.config.LINE_THICKNESS_DEFAULT)
        self.display_manager._draw_text(img, detection_results.display_info.get_probability_text("bottom"), 
                       self.config.TEXT_Y_PROB_BOTTOM, self.config.COLOR_YELLOW, self.config.FONT_SCALE_LARGE, self.config.LINE_THICKNESS_DEFAULT)

        # ROI rectangle display
        self.display_manager.draw_roi_rectangle_on_image(img, roi_coords.top, self.config.ROI_TOP_COLOR, self.config.LINE_THICKNESS_DEFAULT)  # Top ROI (blue)
        self.display_manager.draw_roi_rectangle_on_image(img, roi_coords.top2, self.config.ROI_TOP_ALT_COLOR, self.config.LINE_THICKNESS_DEFAULT)  # Alternative top ROI (orange)
        self.display_manager.draw_roi_rectangle_on_image(img, roi_coords.middle, self.config.ROI_MIDDLE_COLOR, self.config.LINE_THICKNESS_DEFAULT)  # Middle ROI (yellow)
        self.display_manager.draw_roi_rectangle_on_image(img, roi_coords.bottom, self.config.ROI_BOTTOM_COLOR, self.config.LINE_THICKNESS_DEFAULT)  # Bottom ROI (green)
        roi1_coords = detection_results.roi_info.get_roi1_coordinates()
        roi2_coords = detection_results.roi_info.get_roi2_coordinates()
        cv2.rectangle(img, (roi1_coords[0], roi1_coords[1]), (roi1_coords[2], roi1_coords[3]), 
                      self.config.ROI_DETECTION_COLOR, self.config.LINE_THICKNESS_THICK)  # Row detection ROI1 (purple)
        cv2.rectangle(img, (roi2_coords[0], roi2_coords[1]), (roi2_coords[2], roi2_coords[3]), 
                      self.config.ROI_DETECTION_COLOR, self.config.LINE_THICKNESS_THICK)  # Row detection ROI2 (purple)
        self.display_manager.draw_roi_rectangle_on_image(img, roi_coords.detection_area, self.config.ROI_DETECTION_AREA_COLOR, self.config.LINE_THICKNESS_THICK)  # Detection area (black)

        self.display_manager.show_image(img, self.config.RESULT_WINDOW_NAME)
    
    def run_main_loop(self, folder, roi_rows1, roi_rows2):
        """Main processing loop"""
        roi_coords = self.initialize_main_loop(folder, roi_rows1, roi_rows2)

        while True:
            if self.handle_key_input_and_timing():
                break

            should_process = self.read_plc_and_check_processing()
            if should_process:
                img = self.acquire_image()
                if img is None:
                    continue

                detection_results = self.process_image_and_detection(img)
                if detection_results is None:
                    continue

                self.process_results_and_display(img, roi_coords, detection_results)

        self.display_manager.close_all_windows() 
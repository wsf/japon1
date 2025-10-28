# -*- coding: utf-8 -*-
"""
SPF3C_detection_engine.py - Detection engine for industrial image processing system

Contains all detection logic including row detection, line detection, and pattern matching.
"""

import cv2
import numpy as np
import glob
import os
from SPF3C_data_structures import RowDetectionParams


class DetectionEngine:
    """Detection engine class for the industrial image processing system"""
    
    def __init__(self, config_manager, image_processor, roi_manager):
        """Initialize detection engine"""
        self.config = config_manager
        self.image_processor = image_processor
        self.roi_manager = roi_manager
        
        # Global variables for detection results (same as original file)
        self.zone_max_delta = ("none", self.config.DELTA_DEFAULT)
        self.delta_max_mm = self.config.DELTA_DEFAULT
        self.zone_min_delta = ("none", self.config.DELTA_DEFAULT)
        self.delta_min_mm = self.config.DELTA_DEFAULT
    
    def _initialize_row_detection_parameters(self, roi_rows, border_x):
        """Initialize parameters for row detection"""
        x, y, w, h = roi_rows
        usable_width = w - 2 * border_x
        step = usable_width / self.config.NUM_SECTORS
        sector_width = int(step * self.config.SECTOR_WIDTH_MULTIPLIER)
        pattern_width = int(step * self.config.PATTERN_WIDTH_MULTIPLIER)
        return RowDetectionParams(x, y, w, h, step, sector_width, pattern_width)

    def _validate_pattern_against_f9(self, current_roi, f9_references, pattern_start_x, pattern_width, h, empty_threshold):
        """Validate detected pattern against F9 (no row) patterns"""
        is_valid = True
        for f9_pattern, _ in f9_references:
            f9_pattern_crop = f9_pattern[self.config.CROP_TOP_OFFSET:h-self.config.CROP_BOTTOM_OFFSET, max(0, pattern_start_x):pattern_start_x + pattern_width]
            if (current_roi.shape[0] >= f9_pattern_crop.shape[0] and current_roi.shape[1] >= f9_pattern_crop.shape[1]):
                res_f9 = cv2.matchTemplate(current_roi, f9_pattern_crop, cv2.TM_CCOEFF_NORMED)
                _, score_f9, _, _ = cv2.minMaxLoc(res_f9)
                if score_f9 > empty_threshold:
                    is_valid = False
                    break
        return is_valid

    def _process_pattern_matching(self, current_roi, current_pattern, row, threshold, f9_references, pattern_start_x, pattern_width, h, x, y):
        """Process pattern matching for a single pattern"""
        res = cv2.matchTemplate(current_roi, current_pattern, self.config.TEMPLATE_MATCHING_METHOD)
        _, score, _, _ = cv2.minMaxLoc(res)
        
        if row <= 2:
            final_threshold = self.config.FINAL_THRESHOLD_ROW_2
            empty_threshold = self.config.EMPTY_THRESHOLD_ROW_2
        else:
            final_threshold = threshold
            empty_threshold = self.config.EMPTY_THRESHOLD
            
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
        empty_threshold = self.config.EMPTY_THRESHOLD
        params = self._initialize_row_detection_parameters(roi_rows, border_x)
        x, y, w, h = params.x, params.y, params.w, params.h
        step, sector_width, pattern_width = params.step, params.sector_width, params.pattern_width

        best_row_detected = self.config.DEFAULT_ROW_COUNT  # Default: row 9 = no rows
        best_score = -1
        best_name = self.config.DEFAULT_NAME
        best_roi = (0, 0, 0, 0)

        # F9 references = no row
        f9_references = references_by_rows.get(self.config.DEFAULT_ROW_COUNT, [])

        for row in range(self.config.NUM_SECTORS, 0, -1):  # from 8 to 1
            #print("row:" + str(row))
            sector_start_x = int(border_x + (step * (row - self.config.ROW_OFFSET_1 - self.config.ROW_OFFSET_2)))
            pattern_start_x = int(border_x + (step * (row - self.config.ROW_OFFSET_3 - self.config.ROW_OFFSET_2)))
            current_roi = gray[0:h, max(0,sector_start_x):sector_start_x+sector_width]

            if row not in references_by_rows:
                #print("no references")
                continue

            for pattern, name in references_by_rows[row]:
                current_pattern = pattern[self.config.CROP_TOP_OFFSET:h-self.config.CROP_BOTTOM_OFFSET, max(0,pattern_start_x):pattern_start_x+pattern_width]

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
        for i in range(1, self.config.DEFAULT_ROW_COUNT + 1):
            folder = os.path.join(base_folder, f"F{i}")
            refs = []
            #roi_rows_x = calculate_progressive_roi_x(roi_rows,i)
            roi_rows_x = roi_rows
            for img_path in glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")):
                img = cv2.imread(img_path, 0)
                if img is not None:
                    ref_img = self.image_processor.crop_roi_region(img, roi_rows_x)
                    #refs.append(ref_img)
                    img_proc = self.image_processor.preprocess_image2(ref_img)
                    name = os.path.basename(img_path)
                    refs.append((img_proc, name))
            if refs:
                references[i] = refs
                roi_rows_final[i] = roi_rows_x
        return references, roi_rows_final

    def _initialize_line_detection_parameters(self, number_of_rows, roi_rows_x):
        """Initialize parameters for line detection"""
        # Get ROI coordinates directly (no unnecessary tuple conversion)
        roi_top = self.roi_manager.roi_top
        roi_top2 = self.roi_manager.roi_top2
        roi_bottom = self.roi_manager.roi_bottom
        roi_middle = self.roi_manager.roi_middle

        # Get distance limits with safe array access
        distx_max1 = self.roi_manager.max_1x[number_of_rows - 1] if self.roi_manager.max_1x else None
        distx_max2 = self.roi_manager.max_2x[number_of_rows - 1] if self.roi_manager.max_2x else None
        distx_max3 = self.roi_manager.max_3x[number_of_rows - 1] if self.roi_manager.max_3x else None

        from SPF3C_data_structures import LineDetectionParams
        return LineDetectionParams(
            roi_top=roi_top,
            roi_top2=roi_top2,
            roi_bottom=roi_bottom,
            roi_middle=roi_middle,
            distance_limits=(distx_max1, distx_max2, distx_max3)
        )

    def _load_patterns_for_line_detection(self, number_of_rows):
        """Load patterns for line detection"""
        top_patterns = self.image_processor.load_patterns(self.config.patterns_top_dir, number_of_rows)
        top2_patterns = self.image_processor.load_patterns(self.config.patterns_top2_dir, number_of_rows)
        middle_patterns = self.image_processor.load_patterns(self.config.patterns_middle_dir, number_of_rows)
        bottom_patterns = self.image_processor.load_patterns(self.config.patterns_bottom_dir, number_of_rows)

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
        roi1 = self.image_processor.extract_roi_using_rectangle(gray, roi_coords.top)
        roi12 = self.image_processor.extract_roi_using_rectangle(gray, roi_coords.top2)
        roi2 = self.image_processor.extract_roi_using_rectangle(gray, roi_coords.bottom)
        roi3 = self.image_processor.extract_roi_using_rectangle(gray, roi_coords.middle)

        # Detect patterns
        results1 = self.image_processor.detect_best_pattern(roi1, top_patterns, roi_coords.top.x, self.config.LEFT_WEIGHT, validation_threshold, distx_max1)
        if not results1:
            results1 = self.image_processor.detect_best_pattern(roi12, top2_patterns, roi_coords.top2.x, self.config.LEFT_WEIGHT, validation_threshold, distx_max1)

        results2 = self.image_processor.detect_best_pattern(roi2, bottom_patterns, roi_coords.bottom.x, self.config.LEFT_WEIGHT, validation_threshold, distx_max3)
        results3 = self.image_processor.detect_best_pattern(roi3, middle_patterns, roi_coords.middle.x, self.config.LEFT_WEIGHT, validation_threshold/self.config.CENTER_DIVISOR, distx_max2)

        return results1, results2, results3

    def _generate_valid_combos(self, results1, results2, x1, x2):
        """Generate valid combinations from top and bottom results"""
        combos = []
        for res1 in results1:
            for res2 in results2:
                x_distance = abs((x1 + res1[0][0]) - (x2 + res2[0][0]))
                if x_distance <= self.roi_manager.max_distance_x:
                    combo_score = res1[2] + res2[2]
                    combos.append((res1[0], res1[1], res2[0], res2[1], combo_score, (res1[3], res2[3])))

        if not combos:
            raise Exception("No valid combos found")

        return combos

    def _find_best_middle_point(self, results3, x3, y3, pt1, pt2):
        """Find the best middle point from results3"""
        ideal_middle = ((pt1[0] + pt2[0]) // self.config.CENTER_DIVISOR, (pt1[1] + pt2[1]) // self.config.CENTER_DIVISOR)

        best_loc3 = None
        best_score_loc3 = -np.inf

        for res3 in results3:
            pt3 = (x3 + res3[0][0], y3 + res3[0][1])
            distance = np.linalg.norm(np.array(pt3) - np.array(ideal_middle))
            if distance > self.config.DISTANCE_THRESHOLD:
                continue
            dist_norm = distance / self.config.DISTANCE_NORMALIZATION
            score = res3[1] - self.config.DISTANCE_WEIGHT * dist_norm
            if score > best_score_loc3:
                best_score_loc3 = score
                best_loc3 = res3

        return best_loc3

    def _draw_detection_lines(self, img, best_loc3, pt1, pt2, x3, y3):
        """Draw detection lines on the image"""
        if best_loc3:
            pt3 = (x3 + best_loc3[0][0], y3 + best_loc3[0][1])
            detection_success_color = self.config.COLOR_RED
            cv2.line(img, pt1, pt3, detection_success_color, self.config.LINE_THICKNESS_THICK)
            cv2.line(img, pt3, pt2, detection_success_color, self.config.LINE_THICKNESS_THICK)
            middle_name = best_loc3[3]
        else:
            detection_fallback_color = self.config.COLOR_MAGENTA
            cv2.line(img, pt1, pt2, detection_fallback_color, self.config.LINE_THICKNESS_THICK)
            middle_name = "N/A"
        
        return middle_name

    def _display_detection_results(self, img, number_of_rows):
        """Display detection results on image"""
        print(f"📏 Maximum deviation: {self.delta_max_mm:.2f} mm in zone: {self.zone_max_delta[0]} (Row {number_of_rows})")
        self.image_processor._draw_text(img, f"Delta max: {self.delta_max_mm:.2f} mm - {self.zone_max_delta[0]}", self.config.TEXT_Y_DELTA_MAX, self.config.COLOR_CYAN, self.config.FONT_SCALE_LARGE, self.config.LINE_THICKNESS_DEFAULT)
        print(f"📏 Minimum deviation: {self.delta_min_mm:.2f} mm in zone: {self.zone_min_delta[0]} (Row {number_of_rows})")
        self.image_processor._draw_text(img, f"Delta min: {self.delta_min_mm:.2f} mm - {self.zone_min_delta[0]}", self.config.TEXT_Y_DELTA_MIN, self.config.COLOR_CYAN, self.config.FONT_SCALE_LARGE, self.config.LINE_THICKNESS_DEFAULT)

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
            from SPF3C_data_structures import ROICoordinates, PatternDetectionParams
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

            # Get val3 from best_loc3 (same as original file)
            val3 = best_loc3[1] if best_loc3 else -1

            # Draw detection lines
            middle_name = self._draw_detection_lines(img, best_loc3, pt1, pt2, roi_coords.middle.x, roi_coords.middle.y)

            top_name, bottom_name = combo_names

            # Calculate deltas (same as original file)
            try:
                # Get configuration for ideal positions and mm per pixel
                config = self.roi_manager.get_config()
                pos_ideal = config.get("posiciones_filas", {}).get(str(number_of_rows), None)
                mm_per_pixel = config.get("mm_por_pixel", {})

                if pos_ideal:
                    # Get real positions
                    pos_real_arriba = pt1[0]
                    pos_real_abajo = pt2[0]
                    pos_real_medio = (roi_coords.middle.x + best_loc3[0][0]) if best_loc3 else None

                    # Get ideal positions from config
                    pos_ideal_arriba = pos_ideal.get("arriba", pos_real_arriba)
                    pos_ideal_medio = pos_ideal.get("medio", pos_real_medio) if pos_real_medio is not None else None
                    pos_ideal_abajo = pos_ideal.get("abajo", pos_real_abajo)

                    # Calculate deltas in mm
                    delta_arriba = (pos_ideal_arriba - pos_real_arriba) * mm_per_pixel.get("arriba", 0.0)
                    delta_abajo = (pos_ideal_abajo - pos_real_abajo) * mm_per_pixel.get("abajo", 0.0)
                    delta_medio = (
                        (pos_ideal_medio - pos_real_medio) * mm_per_pixel.get("medio", 0.0)
                        if pos_real_medio is not None and pos_ideal_medio is not None else 0.0
                    )

                    # Determine max and min deltas
                    self.zone_max_delta = max(
                        [("arriba", delta_arriba), ("medio", delta_medio), ("abajo", delta_abajo)],
                        key=lambda x: x[1]
                    )
                    self.delta_max_mm = self.zone_max_delta[1]
                    self.zone_min_delta = min(
                        [("arriba", delta_arriba), ("medio", delta_medio), ("abajo", delta_abajo)],
                        key=lambda x: x[1]
                    )
                    self.delta_min_mm = self.zone_min_delta[1]

            except Exception as e:
                print(f"⚠️ Error calculating delta: {e}")
                self.zone_max_delta = ("none", 0.0)
                self.delta_max_mm = 0.0
                self.zone_min_delta = ("none", 0.0)
                self.delta_min_mm = 0.0

            # Display results
            self._display_detection_results(img, number_of_rows)

            # Return detection results with display values (same as original file)
            return {
                'number_of_rows': number_of_rows,
                'top_name': top_name,
                'middle_name': middle_name,
                'bottom_name': bottom_name,
                'val1': val1,
                'val2': val2,
                'val3': val3
            }

        except Exception as e:
            if fallback and number_of_rows < self.config.NUM_SECTORS:
                print(f"⚠️ Failed row {number_of_rows}, trying row {number_of_rows + 1}")
                return self.detect_and_draw_lines(img, gray, number_of_rows + 1, roi_rows_x, validation_threshold, False)

            return -1 
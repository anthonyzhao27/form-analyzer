import math
from typing import Tuple, List
import numpy as np

class GaitAnalyzer:
    def __init__(self, 
                 stride_threshold: float = 0.1,
                 cadence_window: int = 30,
                 alignment_tolerance: float = 0.05,
                 forward_lean_threshold: float = 0.1):
        # Initialize analysis parameters
        self.stride_threshold = stride_threshold
        self.cadence_window = cadence_window
        self.alignment_tolerance = alignment_tolerance
        self.forward_lean_threshold = forward_lean_threshold
        
        # Initialize tracking variables for analysis
        self.stride_history = []
        self.cadence_history = []
        self.alignment_history = []
        self.lean_history = []
        self.frame_count = 0
        
        # Initialize gait state
        self.current_stride_length = 0.0
        self.current_cadence = 0.0
        self.current_alignment = 0.0
        self.current_lean = 0.0
        
    def _is_valid_coordinate(self, coord):
        return (isinstance(coord, (int, float)) and 
                not math.isnan(coord) and 
                not math.isinf(coord) and
                coord is not None)

    def calculate_stride_length(self, left_ankle: Tuple[float, float, float], right_ankle: Tuple[float, float, float]) -> float:
        
        if not left_ankle or not right_ankle:
            return 0.0
        
        if not isinstance(left_ankle, tuple) or not isinstance(right_ankle, tuple):
            return 0.0
        
        if len(left_ankle) != 3 or len(right_ankle) != 3:
            return 0.0
        
        for coord in left_ankle + right_ankle:
            if not self._is_valid_coordinate(coord):
                return 0.0

        dx = left_ankle[0] - right_ankle[0]
        dy = left_ankle[1] - right_ankle[1]
        dz = left_ankle[2] - right_ankle[2]
        stride_length = np.sqrt(dx**2 + dy**2 + dz**2)
        self.stride_history.append(stride_length)
        self.current_stride_length = stride_length
        if len(self.stride_history) > 100:  # Maintain history size
            self.stride_history.pop(0)
        return float(stride_length)
    
    def calculate_cadence(self, ankle_positions: List[Tuple[float, float, float]], frame_rate: float) -> float:
        if not ankle_positions or not frame_rate:
            return 0.0
        
        if frame_rate <= 0:
            return 0.0
        
        step_count = 0
        last_position = None
        for position in ankle_positions:
            if last_position:
                distance = np.linalg.norm(np.array(position) - np.array(last_position))
                if distance > self.stride_threshold:
                    step_count += 1
            last_position = position
        total_frames = len(ankle_positions)
        total_time_seconds = total_frames / frame_rate

        # Calculate cadence
        steps_per_second = step_count / total_time_seconds
        steps_per_minute = steps_per_second * 60
        self.cadence_history.append(steps_per_minute)
        self.current_cadence = steps_per_minute
        if len(self.cadence_history) > 100:  # Maintain history size
            self.cadence_history.pop(0)
        return float(steps_per_minute)

    def analyze_body_alignment(self, shoulder, hip, ankle):
        if not shoulder or not hip or not ankle:
            return 0.0
        
        if not isinstance(shoulder, tuple) or not isinstance(hip, tuple) or not isinstance(ankle, tuple):
            return 0.0
        
        if len(shoulder) != 3 or len(hip) != 3 or len(ankle) != 3:
            return 0.0

        for coord in shoulder + hip + ankle:
            if not self._is_valid_coordinate(coord):
                return 0.0

        # Calculate body line
        dx_shoulder = shoulder[0] - hip[0]
        dy_shoulder = shoulder[1] - hip[1]
        
        dx_hip = ankle[0] - hip[0]
        dy_hip = ankle[1] - hip[1]


        # Calculate alignment
        # Handle division by zero when x-coordinates are identical
        if abs(dx_shoulder) < 1e-10:  # Very small x-difference
            slope1 = float('inf') if dy_shoulder > 0 else float('-inf')
        else:
            slope1 = dy_shoulder / dx_shoulder
            
        if abs(dx_hip) < 1e-10:  # Very small x-difference
            slope2 = float('inf') if dy_hip > 0 else float('-inf')
        else:
            slope2 = dy_hip / dx_hip

        # Calculate angle between body line and ground
        if slope1 == float('inf') or slope1 == float('-inf'):
            angle1 = math.pi / 2 if slope1 == float('inf') else -math.pi / 2
        else:
            angle1 = math.atan(slope1)
            
        if slope2 == float('inf') or slope2 == float('-inf'):
            angle2 = math.pi / 2 if slope2 == float('inf') else -math.pi / 2
        else:
            angle2 = math.atan(slope2)
            
        angle_diff = abs(angle1 - angle2)
        
        # Special case: when both lines are vertical (same x-coordinates), 
        # they are perfectly aligned if they're in the same vertical plane
        if abs(dx_shoulder) < 1e-10 and abs(dx_hip) < 1e-10:
            angle_diff = 0.0  # Perfect alignment for vertical lines

        # Assess alignment
        max_angle = 0.5
        if angle_diff <= self.alignment_tolerance:
            alignment_score = 1.0
        else:
            alignment_score = max(0.0, 1.0 - (angle_diff / max_angle))

        # Update tracking
        self.alignment_history.append(angle_diff)
        self.current_alignment = alignment_score
        if len(self.alignment_history) > 100:  # Maintain history size
            self.alignment_history.pop(0)
        return float(alignment_score)
        
        
    def calculate_forward_lean(self, shoulder, hip):

        if not shoulder or not hip:
            return 0.0
        
        if not isinstance(shoulder, tuple) or not isinstance(hip, tuple):
            return 0.0
        
        if len(shoulder) != 3 or len(hip) != 3:
            return 0.0

        for coord in shoulder + hip:
            if not self._is_valid_coordinate(coord):
                return 0.0

        # Calculate vertical reference
        vertical_reference = hip[0]

        # Calculate horizontal offset
        shoulder_horizontal_offset = shoulder[0] - hip[0]

        # Calculate lean angle
        vertical_dist = abs(shoulder[1] - hip[1])
        lean_angle = math.atan(shoulder_horizontal_offset / vertical_dist)
        lean_angle_degrees = math.degrees(lean_angle)

        # Assess lean
        max_lean_angle = 30
        if abs(lean_angle) <= self.forward_lean_threshold:
            lean_score = 1.0
        else:
            lean_score = max(0.0, 1.0 - (abs(lean_angle) / max_lean_angle))

        # Determine lean direction
        if abs(shoulder_horizontal_offset) < 1e-10:  # Very small offset
            lean_direction = "upright"
        elif shoulder_horizontal_offset > 0:
            lean_direction = "forward"
        else:
            lean_direction = "backward"

        # Update tracking
        self.lean_history.append(lean_score)
        self.current_lean = lean_score
        if len(self.lean_history) > 100:  # Maintain history size
            self.lean_history.pop(0)
        return {"score": lean_score, "angle": lean_angle_degrees, "direction": lean_direction}
        
    def analyze_running_form(self, keypoints, frame_rate):
        if keypoints is None:
            return None
            
        required = ["left_ankle", "right_ankle", "left_hip", "right_hip", "left_shoulder", "right_shoulder"]
        for key in required:
            if key not in keypoints:
                return None
        
        left_ankle = keypoints["left_ankle"]
        right_ankle = keypoints["right_ankle"]

        left_hip = keypoints["left_hip"]
        right_hip = keypoints["right_hip"]
        left_shoulder = keypoints["left_shoulder"]
        right_shoulder = keypoints["right_shoulder"]


        shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2, (left_shoulder[2] + right_shoulder[2]) / 2)
        hip_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2, (left_hip[2] + right_hip[2]) / 2)
        ankle_center = ((left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2, (left_ankle[2] + right_ankle[2]) / 2)

        stride_length = self.calculate_stride_length(left_ankle, right_ankle)
        alignment = self.analyze_body_alignment(shoulder_center, hip_center, ankle_center)
        forward_lean = self.calculate_forward_lean(shoulder_center, hip_center)["score"]
        cadence = self.calculate_cadence([ankle_center], frame_rate)

        # Calculate overall form score
        stride_length_weight = 0.25
        body_alignment_weight = 0.30
        forward_lean_weight = 0.25
        cadence_weight = 0.20

        form_score = (stride_length_weight * stride_length + 
                     body_alignment_weight * alignment + 
                     forward_lean_weight * forward_lean + 
                     cadence_weight * cadence)

        # Determine form quality category
        if form_score >= 0.9:
            form_category = "excellent"
        elif form_score >= 0.75:
            form_category = "good"
        elif form_score >= 0.6:
            form_category = "fair"
        else:
            form_category = "poor"

        # Create comprehensive results dictionary
        results = {
            "stride_length": stride_length,
            "alignment": alignment,
            "forward_lean": forward_lean,
            "cadence": cadence,
            "form_score": form_score,
            "form_category": form_category
        }
        
        # Update frame counter only (histories already updated in helper methods)
        self.frame_count += 1

        return results
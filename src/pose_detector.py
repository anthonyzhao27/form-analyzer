import mediapipe as mp
import cv2
import numpy as np
from typing import Optional, Tuple

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, 
            model_complexity=1, 
            smooth_landmarks=True, 
            enable_segmentation=True, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )

        self.running_keypoints = []
        self.pose_detected = False
        self.landmarks = None
        self.frame_count = 0
        
    def detect_pose(self, frame: np.ndarray) -> Optional[np.ndarray]:
        # VALIDATE INPUT FRAME FIRST
        if frame is None:
            print("Error: Frame is None")
            return None
        
        if frame.size == 0:
            print("Error: Frame is empty")
            return None
        
        if len(frame.shape) != 3:
            print(f"Error: Frame must be 3D, got {len(frame.shape)}D")
            return None
        
        if frame.shape[2] != 3:
            print(f"Error: Frame must have 3 color channels (BGR), got {frame.shape[2]}")
            return None
        
        # Now proceed with normal processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)  # Store as 'results'
        
        if results.pose_landmarks:  # Check results, not self.landmarks
            self.pose_detected = True
            self.landmarks = results  # Store the full results object
            self.running_keypoints = results.pose_landmarks.landmark
            processed_frame = self._draw_landmarks(frame, results)
            self.frame_count += 1
            return processed_frame
        else:
            self.pose_detected = False
            self.landmarks = None
            self.running_keypoints = []
            return frame
    
    def get_landmark_coordinates(self, landmark_id: int) -> Optional[Tuple[float, float, float]]:
        if self.landmarks and 0 <= landmark_id <= 32:
            return self.landmarks.pose_landmarks.landmark[landmark_id].x, self.landmarks.pose_landmarks.landmark[landmark_id].y, self.landmarks.pose_landmarks.landmark[landmark_id].z
        else:
            return None
        
    def get_running_keypoints(self):
        if self.landmarks and self.landmarks.pose_landmarks:
            return {
                "left_ankle": self.get_landmark_coordinates(27),
                "right_ankle": self.get_landmark_coordinates(28),
                "left_knee": self.get_landmark_coordinates(25),
                "right_knee": self.get_landmark_coordinates(26),
                "left_hip": self.get_landmark_coordinates(23),
                "right_hip": self.get_landmark_coordinates(24),
                "left_shoulder": self.get_landmark_coordinates(11),
                "right_shoulder": self.get_landmark_coordinates(12),
                "nose": self.get_landmark_coordinates(0),
                "left_ear": self.get_landmark_coordinates(7),
                "right_ear": self.get_landmark_coordinates(8)
            }
        else:
            return None
        
    def is_pose_detected(self):
        return self.pose_detected

    def _draw_landmarks(self, frame, results):
        self.mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        return frame

    def release(self):
        try:
            if hasattr(self.pose, '_graph') and self.pose._graph is not None:
                self.pose.close()
                print("MediaPipe pose object closed")
            else:
                print("MediaPipe pose object already closed or not initialized")
        except Exception as e:
            print(f"Error closing MediaPipe pose object: {e}")
import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pose_detector import PoseDetector

class TestPoseDetector:
    
    @pytest.fixture
    def pose_detector(self):
        """Create a fresh PoseDetector instance for each test"""
        detector = PoseDetector()
        yield detector
        detector.release()
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample frame for testing"""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    @pytest.fixture
    def mock_mediapipe_results(self):
        """Mock MediaPipe pose detection results"""
        mock_results = Mock()
        
        # Create a more realistic mock landmark with all required attributes
        mock_landmark = Mock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.6
        mock_landmark.z = 0.1
        mock_landmark.visibility = 0.9
        mock_landmark.presence = 0.9
        mock_landmark.HasField = Mock(return_value=True)
        
        # Create 33 landmarks (MediaPipe pose has 33 landmarks)
        mock_results.pose_landmarks.landmark = [mock_landmark] * 33
        return mock_results
    
    def test_initialization(self, pose_detector):
        """Test PoseDetector initialization"""
        assert pose_detector.pose is not None
        assert pose_detector.running_keypoints == []
        assert pose_detector.pose_detected == False
        assert pose_detector.landmarks is None
        assert pose_detector.frame_count == 0
    
    def test_detect_pose_with_valid_frame(self, pose_detector, sample_frame):
        """Test pose detection with a valid frame"""
        with patch.object(pose_detector.pose, 'process') as mock_process:
            mock_process.return_value = Mock()
            mock_process.return_value.pose_landmarks = None
            
            result = pose_detector.detect_pose(sample_frame)
            
            assert result is not None
            assert result.shape == sample_frame.shape
            assert pose_detector.pose_detected == False
    
    def test_detect_pose_with_none_frame(self, pose_detector):
        """Test pose detection with None frame"""
        result = pose_detector.detect_pose(None)
        assert result is None
    
    def test_detect_pose_with_empty_frame(self, pose_detector):
        """Test pose detection with empty frame"""
        empty_frame = np.array([])
        result = pose_detector.detect_pose(empty_frame)
        assert result is None
    
    def test_detect_pose_with_invalid_dimensions(self, pose_detector):
        """Test pose detection with invalid frame dimensions"""
        # 2D frame
        frame_2d = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        result = pose_detector.detect_pose(frame_2d)
        assert result is None
        
        # Frame with wrong number of channels
        frame_wrong_channels = np.random.randint(0, 255, (480, 640, 4), dtype=np.uint8)
        result = pose_detector.detect_pose(frame_wrong_channels)
        assert result is None
    
    def test_detect_pose_with_landmarks(self, pose_detector, sample_frame, mock_mediapipe_results):
        """Test pose detection when landmarks are detected"""
        with patch.object(pose_detector.pose, 'process') as mock_process:
            mock_process.return_value = mock_mediapipe_results
            
            result = pose_detector.detect_pose(sample_frame)
            
            assert result is not None
            assert pose_detector.pose_detected == True
            assert pose_detector.landmarks is not None
            assert pose_detector.running_keypoints is not None
            assert pose_detector.frame_count == 1
    
    def test_get_landmark_coordinates_valid_id(self, pose_detector, mock_mediapipe_results):
        """Test getting landmark coordinates with valid landmark ID"""
        pose_detector.landmarks = mock_mediapipe_results
        pose_detector.pose_detected = True
        
        coords = pose_detector.get_landmark_coordinates(0)
        assert coords == (0.5, 0.6, 0.1)
    
    def test_get_landmark_coordinates_invalid_id(self, pose_detector):
        """Test getting landmark coordinates with invalid landmark ID"""
        coords = pose_detector.get_landmark_coordinates(100)
        assert coords is None
        
        coords = pose_detector.get_landmark_coordinates(-1)
        assert coords is None
    
    def test_get_landmark_coordinates_no_landmarks(self, pose_detector):
        """Test getting landmark coordinates when no landmarks are detected"""
        coords = pose_detector.get_landmark_coordinates(0)
        assert coords is None
    
    def test_get_running_keypoints_with_landmarks(self, pose_detector, mock_mediapipe_results):
        """Test getting running keypoints when landmarks are detected"""
        pose_detector.landmarks = mock_mediapipe_results
        pose_detector.pose_detected = True
        
        keypoints = pose_detector.get_running_keypoints()
        
        assert keypoints is not None
        assert "left_ankle" in keypoints
        assert "right_ankle" in keypoints
        assert "left_knee" in keypoints
        assert "right_knee" in keypoints
        assert "left_hip" in keypoints
        assert "right_hip" in keypoints
        assert "left_shoulder" in keypoints
        assert "right_shoulder" in keypoints
        assert "nose" in keypoints
        assert "left_ear" in keypoints
        assert "right_ear" in keypoints
        
        # Check that all keypoints have the expected coordinates
        for keypoint_name, coords in keypoints.items():
            assert coords == (0.5, 0.6, 0.1)
    
    def test_get_running_keypoints_no_landmarks(self, pose_detector):
        """Test getting running keypoints when no landmarks are detected"""
        keypoints = pose_detector.get_running_keypoints()
        assert keypoints is None
    
    def test_is_pose_detected(self, pose_detector):
        """Test pose detection status"""
        assert pose_detector.is_pose_detected() == False
        
        pose_detector.pose_detected = True
        assert pose_detector.is_pose_detected() == True
    
    def test_draw_landmarks(self, pose_detector, sample_frame, mock_mediapipe_results):
        """Test landmark drawing functionality"""
        result = pose_detector._draw_landmarks(sample_frame, mock_mediapipe_results)
        assert result is not None
        assert result.shape == sample_frame.shape
    
    def test_release(self, pose_detector):
        """Test proper cleanup of MediaPipe resources"""
        pose_detector.release()
        # The pose object should be closed after release
    
    def test_release_with_exception(self, pose_detector):
        """Test release method when an exception occurs during closing"""
        # Mock the pose object to raise an exception during close
        with patch.object(pose_detector.pose, 'close', side_effect=Exception("Test exception")):
            pose_detector.release()
        # Should handle the exception gracefully
    
    def test_integration_workflow(self, pose_detector, sample_frame, mock_mediapipe_results):
        """Test the complete workflow from detection to keypoint extraction"""
        # Mock the MediaPipe process method
        with patch.object(pose_detector.pose, 'process') as mock_process:
            mock_process.return_value = mock_mediapipe_results
            
            # Detect pose
            result_frame = pose_detector.detect_pose(sample_frame)
            assert result_frame is not None
            assert pose_detector.pose_detected == True
            
            # Get keypoints
            keypoints = pose_detector.get_running_keypoints()
            assert keypoints is not None
            
            # Check specific landmark
            nose_coords = pose_detector.get_landmark_coordinates(0)
            assert nose_coords == (0.5, 0.6, 0.1)
            
            # Check pose detection status
            assert pose_detector.is_pose_detected() == True

if __name__ == "__main__":
    pytest.main([__file__])

#!/usr/bin/env python3
"""
Simple test script for PoseDetector class
Run this directly: python test_pose_detector_simple.py
"""

import sys
import os
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pose_detector import PoseDetector

def test_basic_functionality():
    """Test basic PoseDetector functionality"""
    print("Testing PoseDetector basic functionality...")
    
    # Create detector
    detector = PoseDetector()
    print("✓ PoseDetector created successfully")
    
    # Test initialization
    assert detector.pose is not None, "MediaPipe pose object not initialized"
    assert detector.running_keypoints == [], "Running keypoints should be empty initially"
    assert detector.pose_detected == False, "Pose should not be detected initially"
    print("✓ Initialization checks passed")
    
    # Test with a sample frame
    sample_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print("✓ Sample frame created")
    
    # Test pose detection (will likely return frame without landmarks)
    result = detector.detect_pose(sample_frame)
    assert result is not None, "detect_pose should return a frame"
    assert result.shape == sample_frame.shape, "Returned frame should have same shape"
    print("✓ Pose detection with sample frame passed")
    
    # Test landmark coordinates (should return None without landmarks)
    coords = detector.get_landmark_coordinates(0)
    assert coords is None, "Should return None when no landmarks detected"
    print("✓ Landmark coordinate extraction passed")
    
    # Test running keypoints (should return None without landmarks)
    keypoints = detector.get_running_keypoints()
    assert keypoints is None, "Should return None when no landmarks detected"
    print("✓ Running keypoints extraction passed")
    
    # Test pose detection status
    assert detector.is_pose_detected() == False, "Pose should not be detected"
    print("✓ Pose detection status check passed")
    
    # Clean up
    detector.release()
    print("✓ PoseDetector released successfully")
    
    print("\n🎉 All basic tests passed!")

def test_error_handling():
    """Test error handling with invalid inputs"""
    print("\nTesting error handling...")
    
    detector = PoseDetector()
    
    # Test with None frame
    result = detector.detect_pose(None)
    assert result is None, "Should handle None frame gracefully"
    print("✓ None frame handling passed")
    
    # Test with empty frame
    empty_frame = np.array([])
    result = detector.detect_pose(empty_frame)
    assert result is None, "Should handle empty frame gracefully"
    print("✓ Empty frame handling passed")
    
    # Test with 2D frame
    frame_2d = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    result = detector.detect_pose(frame_2d)
    assert result is None, "Should handle 2D frame gracefully"
    print("✓ 2D frame handling passed")
    
    # Test with wrong number of channels
    frame_wrong_channels = np.random.randint(0, 255, (480, 640, 4), dtype=np.uint8)
    result = detector.detect_pose(frame_wrong_channels)
    assert result is None, "Should handle wrong channel count gracefully"
    print("✓ Wrong channel count handling passed")
    
    detector.release()
    print("✓ All error handling tests passed!")

def test_with_real_image():
    """Test with a real image if available"""
    print("\nTesting with real image...")
    
    # Try to create a simple test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (128, 128, 128)  # Gray image
    
    detector = PoseDetector()
    
    # This will likely not detect any pose, but should handle gracefully
    result = detector.detect_pose(test_image)
    assert result is not None, "Should return processed frame"
    print("✓ Real image test passed")
    
    detector.release()

if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_error_handling()
        test_with_real_image()
        print("\n🎯 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gait_analyzer import GaitAnalyzer

class TestGaitAnalyzer(unittest.TestCase):
    
    def setUp(self):
        """
        PSEUDOCODE for setUp:
        1. Create a GaitAnalyzer instance with default parameters
        2. Create sample coordinate data for testing:
           - left_ankle: (0.1, 0.8, 0.0)
           - right_ankle: (0.9, 0.8, 0.0)
           - left_hip: (0.2, 0.6, 0.0)
           - right_hip: (0.8, 0.6, 0.0)
           - left_shoulder: (0.2, 0.4, 0.0)
           - right_shoulder: (0.8, 0.4, 0.0)
        3. Create sample keypoints dictionary with all required keys
        4. Set up frame_rate = 30.0 for cadence calculations
        """
        # Create analyzer with default parameters
        self.analyzer = GaitAnalyzer()
        
        # Create sample coordinate data for testing
        self.left_ankle = (0.1, 0.8, 0.0)
        self.right_ankle = (0.9, 0.8, 0.0)
        self.left_hip = (0.2, 0.6, 0.0)
        self.right_hip = (0.8, 0.6, 0.0)
        self.left_shoulder = (0.2, 0.4, 0.0)
        self.right_shoulder = (0.8, 0.4, 0.0)
        
        # Create sample keypoints dictionary
        self.sample_keypoints = {
            "left_ankle": self.left_ankle,
            "right_ankle": self.right_ankle,
            "left_hip": self.left_hip,
            "right_hip": self.right_hip,
            "left_shoulder": self.left_shoulder,
            "right_shoulder": self.right_shoulder
        }
        
        # Set up frame rate
        self.frame_rate = 30.0
    
    def test_initialization(self):
        """
        PSEUDOCODE for test_initialization:
        1. Test default parameter initialization:
           - stride_threshold should be 0.1
           - cadence_window should be 30
           - alignment_tolerance should be 0.05
           - forward_lean_threshold should be 0.1
        
        2. Test custom parameter initialization:
           - Create analyzer with custom values
           - Verify all parameters are set correctly
        
        3. Test tracking variables initialization:
           - stride_history should be empty list
           - cadence_history should be empty list
           - alignment_history should be empty list
           - lean_history should be empty list
           - frame_count should be 0
        
        4. Test gait state initialization:
           - current_stride_length should be 0.0
           - current_cadence should be 0.0
           - current_alignment should be 0.0
           - current_lean should be 0.0
        """
        # Test default parameter initialization
        self.assertEqual(self.analyzer.stride_threshold, 0.1)
        self.assertEqual(self.analyzer.cadence_window, 30)
        self.assertEqual(self.analyzer.alignment_tolerance, 0.05)
        self.assertEqual(self.analyzer.forward_lean_threshold, 0.1)
        
        # Test custom parameter initialization
        custom_analyzer = GaitAnalyzer(
            stride_threshold=0.15,
            cadence_window=60,
            alignment_tolerance=0.03,
            forward_lean_threshold=0.08
        )
        self.assertEqual(custom_analyzer.stride_threshold, 0.15)
        self.assertEqual(custom_analyzer.cadence_window, 60)
        self.assertEqual(custom_analyzer.alignment_tolerance, 0.03)
        self.assertEqual(custom_analyzer.forward_lean_threshold, 0.08)
        
        # Test tracking variables initialization
        self.assertEqual(self.analyzer.stride_history, [])
        self.assertEqual(self.analyzer.cadence_history, [])
        self.assertEqual(self.analyzer.alignment_history, [])
        self.assertEqual(self.analyzer.lean_history, [])
        self.assertEqual(self.analyzer.frame_count, 0)
        
        # Test gait state initialization
        self.assertEqual(self.analyzer.current_stride_length, 0.0)
        self.assertEqual(self.analyzer.current_cadence, 0.0)
        self.assertEqual(self.analyzer.current_alignment, 0.0)
        self.assertEqual(self.analyzer.current_lean, 0.0)
    
    def test_calculate_stride_length_valid_inputs(self):
        """
        PSEUDOCODE for test_calculate_stride_length_valid_inputs:
        1. Test with valid 3D coordinates:
           - left_ankle = (0.1, 0.8, 0.0)
           - right_ankle = (0.9, 0.8, 0.0)
           - Expected stride length ≈ 0.8 (Euclidean distance)
        
        2. Test with different coordinate values:
           - left_ankle = (0.0, 0.0, 0.0)
           - right_ankle = (1.0, 1.0, 1.0)
           - Expected stride length ≈ 1.732 (sqrt(3))
        
        3. Verify history tracking:
           - stride_history should contain the calculated value
           - current_stride_length should be updated
           - History size should be maintained (max 100)
        """
        # Test with valid 3D coordinates
        stride_length = self.analyzer.calculate_stride_length(self.left_ankle, self.right_ankle)
        expected_length = np.sqrt((0.9 - 0.1)**2 + (0.8 - 0.8)**2 + (0.0 - 0.0)**2)
        self.assertAlmostEqual(stride_length, expected_length, places=6)
        
        # Verify first call history tracking
        self.assertIn(stride_length, self.analyzer.stride_history)
        self.assertEqual(self.analyzer.current_stride_length, stride_length)
        
        # Test with different coordinate values
        stride_length2 = self.analyzer.calculate_stride_length((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        expected_length2 = np.sqrt(3)  # sqrt(1² + 1² + 1²)
        self.assertAlmostEqual(stride_length2, expected_length2, places=6)
        
        # Verify second call history tracking
        self.assertIn(stride_length2, self.analyzer.stride_history)
        # After second call, current_stride_length should be updated to the latest value
        self.assertEqual(self.analyzer.current_stride_length, stride_length2)
    
    def test_calculate_stride_length_invalid_inputs(self):
        """
        PSEUDOCODE for test_calculate_stride_length_invalid_inputs:
        1. Test with None inputs:
           - left_ankle = None, right_ankle = valid_tuple
           - left_ankle = valid_tuple, right_ankle = None
           - Both should return 0.0
        
        2. Test with wrong data types:
           - left_ankle = "string", right_ankle = valid_tuple
           - left_ankle = valid_tuple, right_ankle = [1, 2, 3]
           - Both should return 0.0
        
        3. Test with wrong tuple lengths:
           - left_ankle = (1, 2), right_ankle = (1, 2, 3)
           - left_ankle = (1, 2, 3, 4), right_ankle = (1, 2, 3)
           - Both should return 0.0
        
        4. Test with invalid coordinates:
           - left_ankle = (float('inf'), 1, 1), right_ankle = (1, 1, 1)
           - left_ankle = (float('nan'), 1, 1), right_ankle = (1, 1, 1)
           - Both should return 0.0
        """
        # Test with None inputs
        self.assertEqual(self.analyzer.calculate_stride_length(None, self.right_ankle), 0.0)
        self.assertEqual(self.analyzer.calculate_stride_length(self.left_ankle, None), 0.0)
        
        # Test with wrong data types
        self.assertEqual(self.analyzer.calculate_stride_length("string", self.right_ankle), 0.0)
        self.assertEqual(self.analyzer.calculate_stride_length(self.left_ankle, [1, 2, 3]), 0.0)
        
        # Test with wrong tuple lengths
        self.assertEqual(self.analyzer.calculate_stride_length((1, 2), self.right_ankle), 0.0)
        self.assertEqual(self.analyzer.calculate_stride_length(self.left_ankle, (1, 2, 3, 4)), 0.0)
        
        # Test with invalid coordinates
        self.assertEqual(self.analyzer.calculate_stride_length((float('inf'), 1, 1), self.right_ankle), 0.0)
        self.assertEqual(self.analyzer.calculate_stride_length((float('nan'), 1, 1), self.right_ankle), 0.0)
    
    def test_calculate_cadence_valid_inputs(self):
        """
        PSEUDOCODE for test_calculate_cadence_valid_inputs:
        1. Test with single position (no steps):
           - ankle_positions = [(0.5, 0.8, 0.0)]
           - frame_rate = 30.0
           - Expected cadence = 0.0 (no movement)
        
        2. Test with two positions (one step):
           - ankle_positions = [(0.1, 0.8, 0.0), (0.9, 0.8, 0.0)]
           - frame_rate = 30.0
           - Expected cadence = 60.0 steps/minute (1 step in 0.5 seconds)
        
        3. Test with multiple positions:
           - Create list of 10 positions with alternating large movements
           - Verify step counting logic works correctly
        
        4. Test history tracking:
           - cadence_history should contain calculated values
           - current_cadence should be updated
           - History size should be maintained
        """
        # Test with single position (no steps)
        cadence1 = self.analyzer.calculate_cadence([(0.5, 0.8, 0.0)], self.frame_rate)
        self.assertEqual(cadence1, 0.0)
        
        # Test with two positions (one step)
        cadence2 = self.analyzer.calculate_cadence([(0.1, 0.8, 0.0), (0.9, 0.8, 0.0)], self.frame_rate)
        # 1 step in 2 frames = 0.5 seconds = 120 steps per minute
        expected_cadence = (1 / (2 / self.frame_rate)) * 60
        self.assertAlmostEqual(cadence2, expected_cadence, places=1)
        
        # Test with multiple positions
        positions = [(0.1, 0.8, 0.0), (0.2, 0.8, 0.0), (0.9, 0.8, 0.0), (0.1, 0.8, 0.0)]
        cadence3 = self.analyzer.calculate_cadence(positions, self.frame_rate)
        # 2 steps in 4 frames = 0.133 seconds = 90 steps per minute
        expected_cadence3 = (2 / (4 / self.frame_rate)) * 60
        self.assertAlmostEqual(cadence3, expected_cadence3, places=1)
        
        # Verify history tracking
        self.assertIn(cadence1, self.analyzer.cadence_history)
        self.assertIn(cadence2, self.analyzer.cadence_history)
        self.assertIn(cadence3, self.analyzer.cadence_history)
        self.assertEqual(self.analyzer.current_cadence, cadence3)
    
    def test_calculate_cadence_invalid_inputs(self):
        """
        PSEUDOCODE for test_calculate_cadence_invalid_inputs:
        1. Test with empty ankle_positions:
           - ankle_positions = []
           - frame_rate = 30.0
           - Should return 0.0
        
        2. Test with None inputs:
           - ankle_positions = None, frame_rate = 30.0
           - ankle_positions = valid_list, frame_rate = None
           - Both should return 0.0
        
        3. Test with invalid frame_rate:
           - frame_rate = 0.0 (should return 0.0)
           - frame_rate = -1.0 (should return 0.0)
        
        4. Test with insufficient movement:
           - ankle_positions with movements < stride_threshold
           - Should not count as steps
        """
        # Test with empty ankle_positions
        self.assertEqual(self.analyzer.calculate_cadence([], self.frame_rate), 0.0)
        
        # Test with None inputs
        self.assertEqual(self.analyzer.calculate_cadence(None, self.frame_rate), 0.0)
        self.assertEqual(self.analyzer.calculate_cadence([(0.5, 0.8, 0.0)], None), 0.0)
        
        # Test with invalid frame_rate
        self.assertEqual(self.analyzer.calculate_cadence([(0.1, 0.8, 0.0), (0.9, 0.8, 0.0)], 0.0), 0.0)
        self.assertEqual(self.analyzer.calculate_cadence([(0.1, 0.8, 0.0), (0.9, 0.8, 0.0)], -1.0), 0.0)
        
        # Test with insufficient movement (below stride_threshold)
        small_movement = [(0.1, 0.8, 0.0), (0.15, 0.8, 0.0)]  # Distance = 0.05 < 0.1 threshold
        cadence = self.analyzer.calculate_cadence(small_movement, self.frame_rate)
        self.assertEqual(cadence, 0.0)  # No steps counted
    
    def test_analyze_body_alignment_valid_inputs(self):
        """
        PSEUDOCODE for test_analyze_body_alignment_valid_inputs:
        1. Test perfect alignment (straight line):
           - shoulder = (0.5, 0.4, 0.0)
           - hip = (0.5, 0.6, 0.0)
           - ankle = (0.5, 0.8, 0.0)
           - Expected alignment_score = 1.0 (perfect)
        
        2. Test slight misalignment:
           - shoulder = (0.5, 0.4, 0.0)
           - hip = (0.5, 0.6, 0.0)
           - ankle = (0.6, 0.8, 0.0)
           - Should return score < 1.0 but > 0.0
        
        3. Test significant misalignment:
           - shoulder = (0.5, 0.4, 0.0)
           - hip = (0.5, 0.6, 0.0)
           - ankle = (0.8, 0.8, 0.0)
           - Should return lower score
        
        4. Verify history tracking:
           - alignment_history should contain angle differences
           - current_alignment should be updated
        """
        # Test perfect alignment (straight line)
        shoulder = (0.5, 0.4, 0.0)
        hip = (0.5, 0.6, 0.0)
        ankle = (0.5, 0.8, 0.0)
        alignment_score = self.analyzer.analyze_body_alignment(shoulder, hip, ankle)
        self.assertEqual(alignment_score, 1.0)  # Perfect alignment
        
        # Test slight misalignment with different slopes
        shoulder_sloped = (0.2, 0.4, 0.0)
        hip_sloped = (0.5, 0.6, 0.0)
        ankle_sloped = (0.9, 0.8, 0.0)  # Different slope than shoulder-hip line
        alignment_score2 = self.analyzer.analyze_body_alignment(shoulder_sloped, hip_sloped, ankle_sloped)
        self.assertLess(alignment_score2, 1.0)
        self.assertGreater(alignment_score2, 0.0)
        
        # Test significant misalignment
        ankle_significant = (0.8, 0.8, 0.0)
        alignment_score3 = self.analyzer.analyze_body_alignment(shoulder, hip, ankle_significant)
        self.assertLess(alignment_score3, alignment_score2)  # Lower score for worse alignment
        
        # Verify history tracking
        # History stores angle differences, not scores
        self.assertEqual(len(self.analyzer.alignment_history), 3)
        self.assertEqual(self.analyzer.current_alignment, alignment_score3)
    
    def test_analyze_body_alignment_invalid_inputs(self):
        """
        PSEUDOCODE for test_analyze_body_alignment_invalid_inputs:
        1. Test with None inputs:
           - Any of shoulder, hip, or ankle = None
           - Should return 0.0
        
        2. Test with wrong data types:
           - shoulder = "string", hip = valid_tuple, ankle = valid_tuple
           - Should return 0.0
        
        3. Test with wrong tuple lengths:
           - shoulder = (1, 2), hip = (1, 2, 3), ankle = (1, 2, 3)
           - Should return 0.0
        
        4. Test with invalid coordinates:
           - Any coordinate with inf or nan
           - Should return 0.0
        """
        # Test with None inputs
        self.assertEqual(self.analyzer.analyze_body_alignment(None, (0.5, 0.6, 0.0), (0.5, 0.8, 0.0)), 0.0)
        self.assertEqual(self.analyzer.analyze_body_alignment((0.5, 0.4, 0.0), None, (0.5, 0.8, 0.0)), 0.0)
        self.assertEqual(self.analyzer.analyze_body_alignment((0.5, 0.4, 0.0), (0.5, 0.6, 0.0), None), 0.0)
        
        # Test with wrong data types
        self.assertEqual(self.analyzer.analyze_body_alignment("string", (0.5, 0.6, 0.0), (0.5, 0.8, 0.0)), 0.0)
        
        # Test with wrong tuple lengths
        self.assertEqual(self.analyzer.analyze_body_alignment((1, 2), (0.5, 0.6, 0.0), (0.5, 0.8, 0.0)), 0.0)
        
        # Test with invalid coordinates
        self.assertEqual(self.analyzer.analyze_body_alignment((float('inf'), 0.4, 0.0), (0.5, 0.6, 0.0), (0.5, 0.8, 0.0)), 0.0)
        self.assertEqual(self.analyzer.analyze_body_alignment((float('nan'), 0.4, 0.0), (0.5, 0.6, 0.0), (0.5, 0.8, 0.0)), 0.0)
    
    def test_calculate_forward_lean_valid_inputs(self):
        """
        PSEUDOCODE for test_calculate_forward_lean_valid_inputs:
        1. Test upright posture (no lean):
           - shoulder = (0.5, 0.4, 0.0)
           - hip = (0.5, 0.6, 0.0)
           - Expected lean_score = 1.0, direction = "forward" (minimal)
        
        2. Test forward lean:
           - shoulder = (0.6, 0.4, 0.0)
           - hip = (0.5, 0.6, 0.0)
           - Expected lean_score < 1.0, direction = "forward"
        
        3. Test backward lean:
           - shoulder = (0.4, 0.4, 0.0)
           - hip = (0.5, 0.6, 0.0)
           - Expected lean_score < 1.0, direction = "backward"
        
        4. Test extreme lean:
           - shoulder = (0.8, 0.4, 0.0)
           - hip = (0.5, 0.6, 0.0)
           - Expected lean_score close to 0.0
        
        5. Verify return structure:
           - Should return dict with "score", "angle", "direction"
           - Verify all values are reasonable
        """
        # Test upright posture (no lean)
        shoulder_upright = (0.5, 0.4, 0.0)
        hip = (0.5, 0.6, 0.0)
        lean_result = self.analyzer.calculate_forward_lean(shoulder_upright, hip)
        self.assertEqual(lean_result["score"], 1.0)  # Perfect score for upright
        self.assertEqual(lean_result["direction"], "upright")  # Perfect upright posture
        
        # Test forward lean
        shoulder_forward = (0.6, 0.4, 0.0)
        lean_result2 = self.analyzer.calculate_forward_lean(shoulder_forward, hip)
        self.assertLess(lean_result2["score"], 1.0)
        self.assertEqual(lean_result2["direction"], "forward")
        
        # Test backward lean
        shoulder_backward = (0.4, 0.4, 0.0)
        lean_result3 = self.analyzer.calculate_forward_lean(shoulder_backward, hip)
        self.assertLess(lean_result3["score"], 1.0)
        self.assertEqual(lean_result3["direction"], "backward")
        
        # Test extreme lean
        shoulder_extreme = (0.8, 0.4, 0.0)
        lean_result4 = self.analyzer.calculate_forward_lean(shoulder_extreme, hip)
        self.assertLess(lean_result4["score"], lean_result2["score"])  # Lower score for more extreme lean
        
        # Verify return structure
        required_keys = ["score", "angle", "direction"]
        for key in required_keys:
            self.assertIn(key, lean_result)
        
        # Verify values are reasonable
        self.assertGreaterEqual(lean_result["score"], 0.0)
        self.assertLessEqual(lean_result["score"], 1.0)
        self.assertIsInstance(lean_result["angle"], (int, float))
        self.assertIn(lean_result["direction"], ["forward", "backward", "upright"])
        
        # Verify history tracking
        self.assertIn(lean_result["score"], self.analyzer.lean_history)
        self.assertEqual(self.analyzer.current_lean, lean_result4["score"])
    
    def test_calculate_forward_lean_invalid_inputs(self):
        """
        PSEUDOCODE for test_calculate_forward_lean_invalid_inputs:
        1. Test with None inputs:
           - shoulder = None, hip = valid_tuple
           - shoulder = valid_tuple, hip = None
           - Both should return 0.0
        
        2. Test with wrong data types:
           - shoulder = "string", hip = valid_tuple
           - Should return 0.0
        
        3. Test with wrong tuple lengths:
           - shoulder = (1, 2), hip = (1, 2, 3)
           - Should return 0.0
        
        4. Test with invalid coordinates:
           - Any coordinate with inf or nan
           - Should return 0.0
        """
        # Test with None inputs
        self.assertEqual(self.analyzer.calculate_forward_lean(None, (0.5, 0.6, 0.0)), 0.0)
        self.assertEqual(self.analyzer.calculate_forward_lean((0.5, 0.4, 0.0), None), 0.0)
        
        # Test with wrong data types
        self.assertEqual(self.analyzer.calculate_forward_lean("string", (0.5, 0.6, 0.0)), 0.0)
        
        # Test with wrong tuple lengths
        self.assertEqual(self.analyzer.calculate_forward_lean((1, 2), (0.5, 0.6, 0.0)), 0.0)
        
        # Test with invalid coordinates
        self.assertEqual(self.analyzer.calculate_forward_lean((float('inf'), 0.4, 0.0), (0.5, 0.6, 0.0)), 0.0)
        self.assertEqual(self.analyzer.calculate_forward_lean((float('nan'), 0.4, 0.0), (0.5, 0.6, 0.0)), 0.0)
    
    def test_analyze_running_form_valid_inputs(self):
        """
        PSEUDOCODE for test_analyze_running_form_valid_inputs:
        1. Test with complete keypoints:
           - All required keys present
           - Valid coordinate data
           - Should return comprehensive results dictionary
        
        2. Verify results structure:
           - "stride_length": float value
           - "alignment": float value (0.0 to 1.0)
           - "forward_lean": float value (0.0 to 1.0)
           - "cadence": float value (steps per minute)
           - "form_score": float value (0.0 to 1.0)
           - "form_category": string ("excellent", "good", "fair", "poor")
        
        3. Test form score calculation:
           - Verify weights are applied correctly
           - Verify no division by 3 (should use full weighted sum)
        
        4. Test form categorization:
           - form_score >= 0.9 → "excellent"
           - form_score >= 0.75 → "good"
           - form_score >= 0.6 → "fair"
           - form_score < 0.6 → "poor"
        
        5. Verify frame counter update:
           - frame_count should increment
           - Histories should NOT be updated (handled by helper methods)
        """
        # Test with complete keypoints
        initial_frame_count = self.analyzer.frame_count
        results = self.analyzer.analyze_running_form(self.sample_keypoints, self.frame_rate)
        
        # Verify results structure
        required_keys = ["stride_length", "alignment", "forward_lean", "cadence", "form_score", "form_category"]
        for key in required_keys:
            self.assertIn(key, results)
        
        # Verify data types
        self.assertIsInstance(results["stride_length"], (int, float))
        self.assertIsInstance(results["alignment"], (int, float))
        self.assertIsInstance(results["forward_lean"], (int, float))
        self.assertIsInstance(results["cadence"], (int, float))
        self.assertIsInstance(results["form_score"], (int, float))
        self.assertIsInstance(results["form_category"], str)
        
        # Verify value ranges
        self.assertGreaterEqual(results["alignment"], 0.0)
        self.assertLessEqual(results["alignment"], 1.0)
        self.assertGreaterEqual(results["forward_lean"], 0.0)
        self.assertLessEqual(results["forward_lean"], 1.0)
        self.assertGreaterEqual(results["form_score"], 0.0)
        self.assertLessEqual(results["form_score"], 1.0)
        
        # Verify form categorization
        self.assertIn(results["form_category"], ["excellent", "good", "fair", "poor"])
        
        # Verify frame counter update
        self.assertEqual(self.analyzer.frame_count, initial_frame_count + 1)
        
        # Verify histories are NOT updated (handled by helper methods)
        # This is already verified in the individual method tests
    
    def test_analyze_running_form_invalid_inputs(self):
        """
        PSEUDOCODE for test_analyze_running_form_invalid_inputs:
        1. Test with missing keypoints:
           - Remove one required key
           - Should return None
        
        2. Test with empty keypoints:
           - keypoints = {}
           - Should return None
        
        3. Test with None keypoints:
           - keypoints = None
           - Should return None
        
        4. Test with invalid frame_rate:
           - frame_rate = 0.0
           - frame_rate = -1.0
           - Should handle gracefully (cadence will be 0.0)
        """
        # Test with missing keypoints
        incomplete_keypoints = self.sample_keypoints.copy()
        del incomplete_keypoints["left_ankle"]
        self.assertIsNone(self.analyzer.analyze_running_form(incomplete_keypoints, self.frame_rate))
        
        # Test with empty keypoints
        self.assertIsNone(self.analyzer.analyze_running_form({}, self.frame_rate))
        
        # Test with None keypoints
        self.assertIsNone(self.analyzer.analyze_running_form(None, self.frame_rate))
        
        # Test with invalid frame_rate (should still work, just cadence will be 0.0)
        results = self.analyzer.analyze_running_form(self.sample_keypoints, 0.0)
        self.assertIsNotNone(results)
        self.assertEqual(results["cadence"], 0.0)
        
        results2 = self.analyzer.analyze_running_form(self.sample_keypoints, -1.0)
        self.assertIsNotNone(results2)
        self.assertEqual(results2["cadence"], 0.0)
    
    def test_history_management(self):
        """
        PSEUDOCODE for test_history_management:
        1. Test history size limits:
           - Add 101 values to each history list
           - Verify only 100 values remain
           - Verify oldest values are removed first
        
        2. Test history content:
           - Verify correct values are stored
           - Verify order is maintained (FIFO)
        
        3. Test history independence:
           - stride_history should only contain stride lengths
           - cadence_history should only contain cadence values
           - alignment_history should only contain alignment scores
           - lean_history should only contain lean scores
        """
        # Test history size limits
        # Add 101 values to stride_history
        for i in range(101):
            self.analyzer.calculate_stride_length((i, 0.8, 0.0), (i+1, 0.8, 0.0))
        
        self.assertEqual(len(self.analyzer.stride_history), 100)
        # First value should be removed (oldest)
        self.assertNotIn(0.0, self.analyzer.stride_history)  # First calculated value
        self.assertIn(1.0, self.analyzer.stride_history)     # Last calculated value (distance is always 1.0)
        
        # Test history content and order
        # The first calculated value should be 1.0 (distance from (0,0.8,0) to (1,0.8,0))
        self.assertEqual(self.analyzer.stride_history[0], 1.0)  # First calculated value
        # The last calculated value should be 1.0 (distance from (100,0.8,0) to (101,0.8,0))
        self.assertEqual(self.analyzer.stride_history[-1], 1.0)  # Last calculated value
        
        # Test history independence
        # stride_history should only contain stride lengths
        for value in self.analyzer.stride_history:
            self.assertIsInstance(value, (int, float))
            self.assertGreater(value, 0.0)  # Stride lengths should be positive
    
    def test_edge_cases(self):
        """
        PSEUDOCODE for test_edge_cases:
        1. Test with extreme coordinate values:
           - Very large numbers (1e6, 1e6, 1e6)
           - Very small numbers (1e-6, 1e-6, 1e-6)
           - Should handle without crashing
        
        2. Test with zero coordinates:
           - (0.0, 0.0, 0.0) for all points
           - Should handle division by zero gracefully
        
        3. Test with identical coordinates:
           - left_ankle = right_ankle
           - shoulder = hip
           - Should return appropriate values
        
        4. Test with very high frame rates:
           - frame_rate = 1000.0
           - Should calculate cadence correctly
        
        5. Test with negative coordinates:
           - Negative values for all coordinates
           - Should calculate distances correctly
        
        6. Test with very small movements:
           - Movements below typical thresholds
           - Should handle precision correctly
        
        7. Test with extreme body positions:
           - Very large lean angles
           - Very large misalignments
           - Should produce reasonable scores
        """
        # Test with extreme coordinate values
        extreme_coords = (1e6, 1e6, 1e6)
        small_coords = (1e-6, 1e-6, 1e-6)
        
        # Should handle without crashing
        stride_length = self.analyzer.calculate_stride_length(extreme_coords, small_coords)
        self.assertIsInstance(stride_length, (int, float))
        
        # Test with zero coordinates
        zero_coords = (0.0, 0.0, 0.0)
        stride_length_zero = self.analyzer.calculate_stride_length(zero_coords, zero_coords)
        self.assertEqual(stride_length_zero, 0.0)
        
        # Test with identical coordinates
        identical_coords = (0.5, 0.5, 0.5)
        stride_length_identical = self.analyzer.calculate_stride_length(identical_coords, identical_coords)
        self.assertEqual(stride_length_identical, 0.0)
        
        # Test with very high frame rates
        high_frame_rate = 1000.0
        cadence = self.analyzer.calculate_cadence([(0.1, 0.8, 0.0), (0.9, 0.8, 0.0)], high_frame_rate)
        self.assertIsInstance(cadence, (int, float))
        self.assertGreaterEqual(cadence, 0.0)
        
        # Test with negative coordinates
        negative_coords1 = (-1.0, -1.0, 0.0)
        negative_coords2 = (1.0, 1.0, 0.0)
        stride_length_negative = self.analyzer.calculate_stride_length(negative_coords1, negative_coords2)
        expected_negative = (2**2 + 2**2)**0.5  # sqrt(4 + 4) = sqrt(8) ≈ 2.83
        self.assertAlmostEqual(stride_length_negative, expected_negative, places=2)
        
        # Test with very small movements
        small_movement1 = (0.5, 0.5, 0.0)
        small_movement2 = (0.5001, 0.5, 0.0)
        stride_length_small = self.analyzer.calculate_stride_length(small_movement1, small_movement2)
        self.assertAlmostEqual(stride_length_small, 0.0001, places=5)
        
        # Test with very large numbers
        large_coords1 = (1000, 2000, 0)
        large_coords2 = (2000, 4000, 0)
        stride_length_large = self.analyzer.calculate_stride_length(large_coords1, large_coords2)
        expected_large = (1000**2 + 2000**2)**0.5  # sqrt(1,000,000 + 4,000,000) = sqrt(5,000,000)
        self.assertAlmostEqual(stride_length_large, expected_large, places=2)
        
        # Test with extreme body positions
        # Extreme forward lean
        extreme_lean_shoulder = (0.9, 0.1, 0.0)
        extreme_lean_hip = (0.5, 0.5, 0.0)
        extreme_lean_result = self.analyzer.calculate_forward_lean(extreme_lean_shoulder, extreme_lean_hip)
        self.assertLess(extreme_lean_result["score"], 1.0)
        self.assertEqual(extreme_lean_result["direction"], "forward")
        self.assertGreater(extreme_lean_result["angle"], 30.0)  # Should be a large angle
        
        # Extreme misalignment
        extreme_align_shoulder = (0.0, 0.0, 0.0)
        extreme_align_hip = (0.5, 0.5, 0.0)
        extreme_align_ankle = (1.0, 1.0, 0.0)
        extreme_align_score = self.analyzer.analyze_body_alignment(extreme_align_shoulder, extreme_align_hip, extreme_align_ankle)
        # Should still return a reasonable score (not crash)
        self.assertIsInstance(extreme_align_score, (int, float))
        self.assertGreaterEqual(extreme_align_score, 0.0)
        self.assertLessEqual(extreme_align_score, 1.0)
        
        # Test complete running form with edge case coordinates
        edge_case_keypoints = {
            "left_ankle": (-0.5, 0.8, 0.0),
            "right_ankle": (0.5, 0.8, 0.0),
            "left_hip": (0.0, 0.6, 0.0),
            "right_hip": (0.0, 0.6, 0.0),
            "left_shoulder": (0.0, 0.4, 0.0),
            "right_shoulder": (0.0, 0.4, 0.0)
        }
        
        edge_case_results = self.analyzer.analyze_running_form(edge_case_keypoints, 60.0)
        self.assertIsNotNone(edge_case_results)
        self.assertIn("form_score", edge_case_results)
        self.assertIn("form_category", edge_case_results)
        self.assertGreaterEqual(edge_case_results["form_score"], 0.0)
        self.assertLessEqual(edge_case_results["form_score"], 1.0)
        self.assertIn(edge_case_results["form_category"], ["excellent", "good", "fair", "poor"])
    
    def test_integration_scenarios(self):
        """
        PSEUDOCODE for test_integration_scenarios:
        1. Test complete running form analysis:
           - Create realistic keypoints for good running form
           - Verify all metrics are calculated
           - Verify form score and category are reasonable
        
        2. Test poor running form:
           - Create keypoints representing poor posture
           - Verify lower scores and appropriate category
        
        3. Test improvement over time:
           - Simulate multiple frames with improving form
           - Verify scores increase over time
           - Verify history tracking works correctly
        
        4. Test real-world coordinate ranges:
           - Use coordinates in typical video analysis ranges
           - Verify calculations produce reasonable results
        """
        # Test complete running form analysis
        good_form_keypoints = {
            "left_ankle": (0.1, 0.8, 0.0),
            "right_ankle": (0.9, 0.8, 0.0),
            "left_hip": (0.5, 0.6, 0.0),
            "right_hip": (0.5, 0.6, 0.0),
            "left_shoulder": (0.5, 0.4, 0.0),
            "right_shoulder": (0.5, 0.4, 0.0)
        }
        
        good_results = self.analyzer.analyze_running_form(good_form_keypoints, self.frame_rate)
        self.assertIsNotNone(good_results)
        self.assertGreater(good_results["form_score"], 0.5)  # Should be reasonable score
        
        # Test poor running form
        poor_form_keypoints = {
            "left_ankle": (0.1, 0.8, 0.0),
            "right_ankle": (0.9, 0.8, 0.0),
            "left_hip": (0.5, 0.6, 0.0),
            "right_hip": (0.5, 0.6, 0.0),
            "left_shoulder": (0.8, 0.4, 0.0),  # Extreme forward lean
            "right_shoulder": (0.8, 0.4, 0.0)
        }
        
        poor_results = self.analyzer.analyze_running_form(poor_form_keypoints, self.frame_rate)
        self.assertIsNotNone(poor_results)
        # Poor form should generally have lower scores
        self.assertLessEqual(poor_results["form_score"], good_results["form_score"])
        
        # Test real-world coordinate ranges
        real_world_keypoints = {
            "left_ankle": (100, 400, 0),
            "right_ankle": (200, 400, 0),
            "left_hip": (150, 300, 0),
            "right_hip": (150, 300, 0),
            "left_shoulder": (150, 200, 0),
            "right_shoulder": (150, 200, 0)
        }
        
        real_world_results = self.analyzer.analyze_running_form(real_world_keypoints, self.frame_rate)
        self.assertIsNotNone(real_world_results)
        # Should produce reasonable results even with larger coordinate values

if __name__ == '__main__':
    unittest.main()

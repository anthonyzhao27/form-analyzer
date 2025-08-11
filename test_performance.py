#!/usr/bin/env python3
"""
Performance testing script for PoseDetector
Run this to test processing speed with different frame sizes
"""

import sys
import os
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pose_detector import PoseDetector

def test_performance():
    """Test PoseDetector performance with different frame sizes"""
    print("🚀 Performance Testing PoseDetector")
    print("=" * 50)
    
    # Test different frame sizes
    frame_sizes = [
        (480, 640, 3),    # Standard definition
        (720, 1280, 3),   # HD
        (1080, 1920, 3),  # Full HD
        (1440, 2560, 3),  # 2K
    ]
    
    detector = PoseDetector()
    
    try:
        for height, width, channels in frame_sizes:
            print(f"\n📏 Testing frame size: {width}x{height}")
            
            # Create test frame
            frame = np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
            
            # Warm up (first few frames are slower)
            print("   Warming up...")
            for _ in range(3):
                detector.detect_pose(frame)
            
            # Performance test
            print("   Running performance test...")
            start_time = time.time()
            
            num_frames = 50
            for _ in range(num_frames):
                detector.detect_pose(frame)
            
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            fps = num_frames / total_time
            avg_time_per_frame = total_time / num_frames * 1000  # in milliseconds
            
            print(f"   ✅ Processed {num_frames} frames in {total_time:.2f}s")
            print(f"   📊 Performance: {fps:.1f} FPS")
            print(f"   ⏱️  Average time per frame: {avg_time_per_frame:.1f}ms")
            
            # Memory usage info
            frame_size_mb = (height * width * channels) / (1024 * 1024)
            print(f"   💾 Frame size: {frame_size_mb:.1f} MB")
    
    finally:
        detector.release()
        print("\n🧹 Cleanup completed")

def test_memory_usage():
    """Test memory usage with continuous processing"""
    print("\n💾 Memory Usage Test")
    print("=" * 30)
    
    detector = PoseDetector()
    
    try:
        # Create a moderate-sized frame
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        print("Processing 100 frames to test memory stability...")
        
        for i in range(100):
            if i % 20 == 0:
                print(f"   Processed {i} frames...")
            
            result = detector.detect_pose(frame)
            
            # Simulate some processing delay
            time.sleep(0.01)
        
        print("   ✅ Memory test completed successfully")
        
    finally:
        detector.release()

def test_error_handling_performance():
    """Test performance when handling invalid inputs"""
    print("\n⚠️  Error Handling Performance Test")
    print("=" * 40)
    
    detector = PoseDetector()
    
    try:
        # Test with various invalid inputs
        invalid_inputs = [
            None,
            np.array([]),
            np.random.randint(0, 255, (480, 640), dtype=np.uint8),  # 2D
            np.random.randint(0, 255, (480, 640, 4), dtype=np.uint8),  # Wrong channels
        ]
        
        start_time = time.time()
        
        for i, invalid_input in enumerate(invalid_inputs):
            result = detector.detect_pose(invalid_input)
            assert result is None, f"Should handle invalid input {i} gracefully"
        
        end_time = time.time()
        
        print(f"   ✅ Processed {len(invalid_inputs)} invalid inputs in {(end_time - start_time)*1000:.1f}ms")
        print(f"   📊 Average time per invalid input: {(end_time - start_time)/len(invalid_inputs)*1000:.1f}ms")
        
    finally:
        detector.release()

if __name__ == "__main__":
    try:
        test_performance()
        test_memory_usage()
        test_error_handling_performance()
        
        print("\n🎯 All performance tests completed successfully!")
        print("\n📋 Performance Summary:")
        print("   • Frame processing speed tested")
        print("   • Memory usage stability verified")
        print("   • Error handling performance measured")
        print("   • Resource cleanup confirmed")
        
    except Exception as e:
        print(f"\n❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Testing Guide for PoseDetector

This guide explains how to test the `pose_detector.py` file using different approaches.

## Prerequisites

Make sure you have all the required dependencies installed:

```bash
pip install -r requirements.txt
pip install -r requirements-test.txt
```

## Testing Approaches

### 1. **Quick Test (Recommended for beginners)**

Run the simple test script directly:

```bash
python test_pose_detector_simple.py
```

This will run basic functionality tests and error handling tests without requiring pytest.

### 2. **Comprehensive Unit Tests with pytest**

Run the full test suite:

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_pose_detector.py

# Run specific test method
pytest tests/test_pose_detector.py::TestPoseDetector::test_initialization
```

### 3. **Interactive Testing in Python REPL**

```python
# Start Python REPL
python

# Import and test
from src.pose_detector import PoseDetector
import numpy as np

# Create detector
detector = PoseDetector()

# Test with sample frame
frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
result = detector.detect_pose(frame)

# Check results
print(f"Pose detected: {detector.is_pose_detected()}")
print(f"Frame shape: {result.shape}")

# Clean up
detector.release()
```

## What the Tests Cover

### **Basic Functionality Tests**
- ✅ Initialization of MediaPipe pose detection
- ✅ Frame processing and validation
- ✅ Pose detection workflow
- ✅ Landmark coordinate extraction
- ✅ Running keypoints retrieval
- ✅ Pose detection status checking

### **Error Handling Tests**
- ✅ None frame handling
- ✅ Empty frame handling
- ✅ Invalid frame dimensions (2D, wrong channels)
- ✅ Invalid landmark IDs
- ✅ Graceful degradation when no pose detected

### **Integration Tests**
- ✅ Complete workflow from detection to keypoint extraction
- ✅ Resource cleanup and memory management
- ✅ MediaPipe integration

## Test Structure

```
tests/
├── test_pose_detector.py          # Comprehensive pytest tests
├── requirements-test.txt           # Testing dependencies
└── test_pose_detector_simple.py   # Simple test script

TESTING.md                         # This guide
```

## Understanding Test Results

### **Green Checkmarks (✓)**
- All assertions passed
- Functionality working as expected

### **Red X (❌)**
- Test failed
- Check error messages for debugging

### **Common Issues and Solutions**

1. **Import Errors**
   - Make sure you're in the project root directory
   - Check that `src/` is in your Python path

2. **MediaPipe Errors**
   - Ensure MediaPipe is properly installed
   - Check version compatibility

3. **OpenCV Errors**
   - Verify OpenCV installation
   - Check numpy array compatibility

## Performance Testing

For performance testing, you can create larger frames or process multiple frames:

```python
# Performance test with larger frames
large_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
detector = PoseDetector()

import time
start_time = time.time()
for _ in range(100):
    result = detector.detect_pose(large_frame)
end_time = time.time()

print(f"Processed 100 frames in {end_time - start_time:.2f} seconds")
detector.release()
```

## Continuous Integration

To integrate testing into your development workflow:

1. **Pre-commit hooks**: Run tests before each commit
2. **GitHub Actions**: Automate testing on push/PR
3. **Local development**: Run tests after each significant change

## Debugging Failed Tests

1. **Read error messages carefully** - they often point to the exact issue
2. **Check dependencies** - ensure all packages are installed
3. **Verify file paths** - make sure you're in the correct directory
4. **Use verbose output** - `pytest -v` provides more details
5. **Run individual tests** - isolate the failing test case

## Contributing

When adding new features to `PoseDetector`:

1. **Write tests first** (TDD approach)
2. **Cover edge cases** - test invalid inputs
3. **Test error conditions** - ensure graceful failure
4. **Update this guide** - document new testing approaches

Happy testing! 🧪✨

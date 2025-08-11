# Running Form Analyzer

A computer vision application that analyzes running gait and form in real-time or from video footage.

## Features (MVP)

1. **Real-time pose detection** - Track 33 body keypoints using MediaPipe
2. **Gait analysis** - Analyze stride length, cadence, and body alignment
3. **Form feedback** - Provide real-time feedback on running technique
4. **Video processing** - Analyze pre-recorded running videos
5. **Data visualization** - Generate charts and reports of running metrics

## Project Structure

```
form-analyzer/
├── src/
│   ├── __init__.py
│   ├── pose_detector.py      # Core pose detection logic
│   ├── gait_analyzer.py      # Running form analysis
│   ├── video_processor.py    # Video input/output handling
│   └── utils.py              # Helper functions
├── tests/
│   └── __init__.py
├── data/                     # Sample videos and output
├── requirements.txt
└── main.py                   # Main application entry point
```

## Installation

1. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Real-time Analysis
```bash
python main.py --mode realtime
```

### Video Analysis
```bash
python main.py --mode video --input path/to/video.mp4
```

## System Design

The application follows a modular architecture:

1. **Input Layer**: Camera feed or video file
2. **Pose Detection Layer**: MediaPipe for real-time pose estimation
3. **Analysis Layer**: Custom algorithms for gait analysis
4. **Output Layer**: Real-time feedback and data visualization

## Key Technologies

- **MediaPipe**: Google's ML framework for pose detection
- **OpenCV**: Computer vision operations and video processing
- **TensorFlow**: Deep learning models for enhanced pose estimation
- **NumPy/SciPy**: Numerical computations and signal processing

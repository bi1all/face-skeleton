Face Skeleton - Real-Time 3D Mesh Tracking



Vain, fast, privacy-locked face landmark skeleton. 468+ real-time facial mesh points. Zero cloud. Pure flex.



 Features

468+ Face Landmarks - Dense 3D mesh in real-time

CPU-Only - No GPU dependency, runs on potato hardware

 Privacy First - Raw frames never leave your machine

 Live Rendering - 30fps+ on standard hardware

 Landmark Export - Save mesh data for analysis



 Requirements

```bash

pip install mediapipe opencv-python numpy

```



 Usage

```bash

python face\_skeleton.py

```

# Controls

 ESC - Exit

 S - Save landmark data to file

 SPACE - Pause/Resume


 How It Works

Uses MediaPipe Tasks Face Landmark Detection to extract 468 3D points from your face in real-time. Renders as connected mesh overlay on video feed.


 Output

Landmarks saved as JSON with x, y, z coordinates for each point. Use for ML training, animation rigging, or just staring at yourself.

Built with MediaPipe. No data sent anywhere.



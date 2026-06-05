# Face Skeleton - Real-Time 3D Mesh Tracking & Expressions



Vain, fast, privacy-locked face landmark skeleton. 468+ real-time facial mesh points with live micro-expression blendshape tracking. Zero cloud. Pure flex.



 Features

* **468+ Face Landmarks** - Dense 3D mesh in real-time, stabilized with Exponential Moving Average (EMA) smoothing for jitter-free tracking.
* **Live Micro-Expressions (Blendshapes)** - Real-time UI rendering over 50+ unique facial micro-expressions (e.g., eye blink, mouth smile, jaw open) similar to Apple's Face ID capabilities.
* **Depth-Mapped 3D Visuals** - High-tech visual connections dynamically colored from deep-blue to bright-cyan based on real-time Z-axis depth.
* **CPU-Only** - No GPU dependency, runs gracefully on standard hardware.
* **Privacy First** - Raw frames never leave your machine.
* **Pause & Save** - Freeze frame functionality with spacebar to easily review your mesh and save the stabilized 3D coordinates.


 Requirements

```bash

pip install mediapipe opencv-python numpy

```



 Usage

```bash

python face_skeleton.py

```

# Controls

* **ESC** - Exit the application
* **SPACE** - Pause/Resume the camera feed (useful for freezing a facial expression to examine the tracking or save it)
* **S** - Save landmark data to file (`face_landmarks.txt`)


 How It Works

Uses MediaPipe Tasks Face Landmark Detection to extract 468 3D points and blendshape coefficients from your face in real-time. Renders as a connected mesh overlay mapped to depth on the video feed.


 Output

Landmarks saved as CSV (`id, x, y, z`) for each point. Use for ML training, animation rigging, avatar control, or just staring at yourself.

Built with MediaPipe. No data sent anywhere.



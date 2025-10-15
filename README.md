# Hand Gesture Volume Control (Streamlit)

A **Windows-based Streamlit app** that controls system volume using hand gestures via a webcam.

## Features

- Hand detection using **MediaPipe**
- Pinch thumb & index → adjust volume (0–99%)
- Open hand → max volume
- Closed hand → mute
- Animated vertical volume bar
- Text feedback: Volume percentage and gesture
- Start/Stop webcam buttons

## Requirements

- Python 3.8 – 3.12 (Windows)
- Streamlit
- OpenCV
- Mediapipe
- NumPy
- Pycaw
- PyWin32

## Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/hand-volume-control.git
cd hand-volume-control

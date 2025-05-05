# Elegoo Smart Car ML Control

This project controls an Elegoo Smart Robot Car using a machine learning model and a WiFi camera. It uses image classification to make movement decisions based on real-time camera input.

## Features
- TensorFlow/Keras image classification
- Real-time control via TCP socket
- Custom depthwise layer fix for model compatibility
- Heartbeat and reconnect logic

## Requirements
- Python 3.8+
- Elegoo Smart Robot Car V4.0
- ESP32-CAM streaming MJPEG
- TensorFlow 2.x
- PIL (Pillow), numpy

## Setup

1. Place your Keras model and labels in the `/model` directory.
2. Update `MODEL_PATH`, `LABELS_PATH`, and `ESP_CAMERA_IP` in `main.py`.
3. Run:
```bash
pip install -r requirements.txt
python src/main.py
```

## Files
- `main.py`: Entry point
- `image_utils.py`: Handles image processing
- `network_utils.py`: Reliable communication logic
- `version_compat_test.py`: Compatibility checks for custom layers

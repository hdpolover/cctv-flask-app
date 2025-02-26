# CCTV People Detection App

This Flask-based web application monitors a CCTV camera feed and detects people entering and exiting a room. The app uses pre-trained object detection models (like Faster R-CNN or YOLO) to track and count the number of people in the room.

## Features
- Real-time video feed processing to detect people
- Detection using a pre-trained model (Faster R-CNN/YOLO)
- Drawing a door line to track movement across the line (indicating entry/exit)
- Counts how many people entered and exited the room
- GPU acceleration support via CUDA

## Tech Stack
- Python 3.x
- Flask (Web Framework)
- OpenCV (Video Processing)
- PyTorch (Machine Learning)
- TorchVision (Pre-trained Models for Object Detection)
- CUDA (for GPU acceleration)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/cctv-flask-app.git
cd cctv-flask-app
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # For Mac/Linux
.venv\Scripts\activate     # For Windows
```

### 3. Install required dependencies
```bash
pip install -r requirements.txt
```

### 4. Install PyTorch with CUDA support (recommended for performance)
For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CPU only (slower detection):
```bash
pip install torch torchvision torchaudio
```

## Configuration

### 1. Camera Setup
Make sure the CCTV camera is connected and its stream is accessible. In the Flask app, the camera feed is pulled using OpenCV.

Edit the following line in `app.py` to set your camera source:

```python
camera = cv2.VideoCapture('your_camera_url_or_device')  # E.g., "0" for a local camera
```

### 2. Model Setup
The app uses a pre-trained Faster R-CNN model or YOLO for people detection. By default, it loads the Faster R-CNN model trained on the CCTV dataset.

To use the YOLOv5 model instead of Faster R-CNN, modify the detection code to use `detect_people_yolo()`.

### 3. GPU Configuration
The application automatically detects if a CUDA-compatible GPU is available:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

To check if CUDA is properly configured:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Run the Application

To start the Flask app, run the following command:

```bash
flask run
```

For development with debug mode:
```bash
flask run --debug
```

This will start a local server at `http://127.0.0.1:5000/`.

## Usage
- Open your browser and navigate to `http://127.0.0.1:5000/` to view the real-time CCTV feed.
- The app will detect people entering and exiting the room, and draw bounding boxes around detected people.
- The door line can be drawn on the video feed to track when people cross it (indicating entry/exit).
- Real-time counters display how many people entered and exited.

## Detection Logic
- **Faster R-CNN Model**: A pre-trained Faster R-CNN model is used for detecting people in the video feed.
- **Crossing Detection**: A door line is drawn on the video frame. If a detected person crosses this line, it is counted as either entering or exiting the room.
- **Tracking Algorithm**: Centroid tracking is used to maintain person identity across frames.

## Performance Optimization
- **GPU Acceleration**: Using CUDA can significantly improve detection speed.
- **Frame Skipping**: If running on slower hardware, try adjusting the frame skip rate in the configuration.
- **Resolution Scaling**: Lower resolution processing can improve performance.

## Troubleshooting
- **Laggy Video Feed**: If the video feed is laggy, try reducing the frame resolution, skipping frames, or using a lighter object detection model like YOLOv5.
- **CUDA Out of Memory**: Reduce batch size or input resolution if encountering GPU memory issues.
- **Model Performance**: If using a CPU, object detection models may be slower. Consider using a GPU if available.
- **Error with Camera Feed**: Ensure the camera is properly connected and the stream URL is correct.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository, make improvements, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

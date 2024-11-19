# CCTV People Detection App

This Flask-based web application monitors a CCTV camera feed and detects people entering and exiting a room. The app uses pre-trained object detection models (like Faster R-CNN or YOLO) to track and count the number of people in the room.

## Features
- Real-time video feed processing to detect people
- Detection using a pre-trained model (Faster R-CNN/YOLO)
- Drawing a door line to track movement across the line (indicating entry/exit)
- Counts how many people entered and exited the room

## Tech Stack
- Python 3.x
- Flask (Web Framework)
- OpenCV (Video Processing)
- PyTorch (Machine Learning)
- TorchVision (Pre-trained Models for Object Detection)

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

### 4. Install other dependencies for object detection (if necessary)
```bash
pip install torch torchvision
```

## Configuration

### 1. Camera Setup
Make sure the CCTV camera is connected and its stream is accessible. In the Flask app, the camera feed is pulled using OpenCV.

Edit the following line in `app.py` to set your camera source:

```python
camera = cv2.VideoCapture('your_camera_url_or_device')  # E.g., "0" for a local camera
```

### 2. Model Setup
The app uses a pre-trained Faster R-CNN model or YOLO for people detection. By default, it loads the Faster R-CNN model trained on the COCO dataset.

To use the YOLOv5 model instead of Faster R-CNN, modify the detection code to use `detect_people_yolo()`.

## Run the Application

To start the Flask app, run the following command:

```bash
flask run
```

This will start a local server at `http://127.0.0.1:5000/`.

## Usage
- Open your browser and navigate to `http://127.0.0.1:5000/` to view the real-time CCTV feed.
- The app will detect people entering and exiting the room, and draw bounding boxes around detected people.
- The door line can be drawn on the video feed to track when people cross it (indicating entry/exit).

## Detection Logic
- **Faster R-CNN Model**: A pre-trained Faster R-CNN model is used for detecting people in the video feed.
- **Crossing Detection**: A door line is drawn on the video frame. If a detected person crosses this line, it is counted as either entering or exiting the room.

## Troubleshooting
- **Laggy Video Feed**: If the video feed is laggy, try reducing the frame resolution, skipping frames, or using a lighter object detection model like YOLOv5.
- **Model Performance**: If using a CPU, object detection models may be slower. Consider using a GPU if available.
- **Error with Camera Feed**: Ensure the camera is properly connected and the stream URL is correct.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository, make improvements, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


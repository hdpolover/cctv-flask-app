import cv2
import time
import base64
import torch
from flask_socketio import SocketIO
from detection_model import DetectionModel

class VideoStream:
    def __init__(self, video_path, frame_rate=30, resolution=(640, 480)):
        self.video_path = video_path
        self.frame_rate = frame_rate
        self.resolution = resolution
        self.cap = cv2.VideoCapture(video_path)
        self.prev_time = 0
        self.detection_model = DetectionModel()

    def update_settings(self, video_path, frame_rate, resolution):
        self.video_path = video_path
        self.frame_rate = frame_rate
        self.resolution = resolution
        self.cap.release()
        self.cap = cv2.VideoCapture(video_path)

    def capture_frames(self, socketio):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            resized_frame = cv2.resize(frame, self.resolution)
            
            try:
                people_boxes, movement = self.detection_model.detect_people(resized_frame)
            except NotImplementedError as e:
                print(f"CUDA operation not supported, falling back to CPU: {e}")
                self.detection_model.device = torch.device("cpu")
                self.detection_model.model.to(self.detection_model.device)
                people_boxes, movement = self.detection_model.detect_people(resized_frame)
                
            for i, box in enumerate(people_boxes):
                # Draw a green rectangle around the detected person
                cv2.rectangle(resized_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                # Add label to the box
                label = f"Person {i + 1}"
                cv2.putText(resized_frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', resized_frame)
            frame = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('video_frame', frame)
            time.sleep(1.0 / self.frame_rate)
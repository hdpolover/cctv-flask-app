import cv2
import torch
import time
from torchvision import models, transforms, ops
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

class DetectionModel:
    def __init__(self, device):
        self.device = device
        self.model = self.load_model()
        self.model.to(self.device)
        self.score_threshold = 0.8
        self.iou_threshold = 0.3

    def load_model(self):
        """Load the Faster R-CNN model pre-trained on COCO dataset."""
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        model.eval()
        return model

    def preprocess_image(self, image):
        """Preprocess image for Faster R-CNN."""
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        return transform(image)

    def detect_people(self, image):
        """Detect people using Faster R-CNN."""
        image_tensor = self.preprocess_image(image).to(self.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model([image_tensor])
        inference_time = time.time() - start_time

        boxes = outputs[0]['boxes']
        scores = outputs[0]['scores']
        labels = outputs[0]['labels']

        # Filter out non-person detections and low-confidence scores
        person_indices = (labels == 1) & (scores >= self.score_threshold)
        boxes = boxes[person_indices]
        scores = scores[person_indices]

        # Apply NMS
        keep_indices = ops.nms(boxes, scores, self.iou_threshold)
        people_boxes = boxes[keep_indices].cpu().numpy().astype(int)

        return people_boxes, inference_time

def main():
    # Initialize video capture with a static video file
    cap = cv2.VideoCapture("static/videos/demo1.mp4")
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Initialize models for CPU and GPU
    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_device = cpu_device
    detection_model = DetectionModel(current_device)

    print("Press 'c' to switch to CPU, 'g' to switch to GPU, and 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from video file.")
            break

        # Resize frame
        resized_frame = cv2.resize(frame, (640, 480))

        # Detect people
        people_boxes, inference_time = detection_model.detect_people(resized_frame)
        for box in people_boxes:
            cv2.rectangle(resized_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(resized_frame, f"Device: {'GPU' if current_device.type == 'cuda' else 'CPU'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(resized_frame, f"Time: {inference_time:.4f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Video', resized_frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            current_device = cpu_device
            detection_model = DetectionModel(current_device)
            print("Switched to CPU")
        elif key == ord('g') and torch.cuda.is_available():
            current_device = gpu_device
            detection_model = DetectionModel(current_device)
            print("Switched to GPU")

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import torch
from torchvision import models, transforms, ops
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import numpy as np
import time

class DetectionModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"CUDA available: {torch.cuda.is_available()}")
        self.model = self.load_model()
        self.model.to(self.device)
        self.previous_centers = {}  # Store previous positions
        self.track_id = 0  # Unique ID for each tracked person
        self.score_threshold = 0.8
        self.iou_threshold = 0.3
        self.tracking_threshold = 50  # Maximum pixel distance for tracking
        self.left_to_right = 0
        self.right_to_left = 0

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

    def get_box_center(self, box):
        """Calculate center point of bounding box."""
        return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

    def track_movement(self, current_boxes, frame_width):
        """Track movement direction of detected people."""
        center_line = frame_width // 2
        current_centers = {}
        movement_count = {"left_to_right": 0, "right_to_left": 0}

        # Calculate centers for current boxes
        for box in current_boxes:
            center_x, center_y = self.get_box_center(box)
            min_dist = float('inf')
            matched_id = None

            # Try to match with previous centers
            for track_id, prev_center in self.previous_centers.items():
                dist = np.sqrt((center_x - prev_center[0])**2 + (center_y - prev_center[1])**2)
                if dist < min_dist and dist < self.tracking_threshold:
                    min_dist = dist
                    matched_id = track_id

            # If no match found, create new track
            if matched_id is None:
                matched_id = self.track_id
                self.track_id += 1

            current_centers[matched_id] = (center_x, center_y)

            # Check for line crossing
            if matched_id in self.previous_centers:
                prev_x = self.previous_centers[matched_id][0]
                # Left to right movement
                if prev_x < center_line and center_x >= center_line:
                    movement_count["left_to_right"] += 1
                    self.left_to_right += 1
                # Right to left movement
                elif prev_x >= center_line and center_x < center_line:
                    movement_count["right_to_left"] += 1
                    self.right_to_left += 1

        # Update previous centers
        self.previous_centers = current_centers
        return movement_count

    def detect_people(self, image, score_threshold=None, iou_threshold=None):
        """Detect people using Faster R-CNN."""
        if score_threshold is None:
            score_threshold = self.score_threshold
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
            
        start_time = time.time()
        image_tensor = self.preprocess_image(image).to(self.device)
        
        with torch.no_grad():
            outputs = self.model([image_tensor])
        
        inference_time = time.time() - start_time

        boxes = outputs[0]['boxes']
        scores = outputs[0]['scores']
        labels = outputs[0]['labels']

        # Filter out non-person detections and low-confidence scores
        person_indices = (labels == 1) & (scores >= score_threshold)
        boxes = boxes[person_indices]
        scores = scores[person_indices]

        # Apply NMS
        keep_indices = ops.nms(boxes, scores, iou_threshold)
        people_boxes = boxes[keep_indices].cpu().numpy().astype(int)

        # Track movements
        movement = self.track_movement(people_boxes, image.shape[1])

        return people_boxes, movement
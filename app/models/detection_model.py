"""
People detection model for identifying and tracking people in video frames
"""
import torch
from torchvision import models, transforms, ops
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import numpy as np
import time
import logging

from flask import current_app

# Configure logging
logger = logging.getLogger(__name__)

class DetectionModel:
    """Model for detecting and tracking people in video frames"""
    
    def __init__(self, config=None):
        """Initialize the detection model.
        
        Args:
            config: Configuration dictionary with detection parameters
        """
        # Use provided config or get from Flask app config if available
        if config is None and current_app:
            config = current_app.config
        else:
            config = {}
            
        # Set up device (CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing detection model on {self.device} (CUDA available: {torch.cuda.is_available()})")
        
        # Load model
        self.model = self.load_model()
        self.model.to(self.device)
        
        # Detection parameters
        self.score_threshold = config.get('SCORE_THRESHOLD', 0.8)
        self.iou_threshold = config.get('IOU_THRESHOLD', 0.3)
        self.tracking_threshold = config.get('TRACKING_THRESHOLD', 50)
        
        # Tracking state
        self.previous_centers = {}  # Store previous positions
        self.track_id = 0  # Unique ID for each tracked person
        self.left_to_right = 0
        self.right_to_left = 0
        
        # Door detection parameters
        self.door_defined = False
        self.door_area = None
        self.inside_direction = "right"  # can be "left" or "right"

    def load_model(self):
        """Load the Faster R-CNN model pre-trained on COCO dataset.
        
        Returns:
            Loaded PyTorch model
        """
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        model.eval()
        return model

    def set_door_area(self, x1, y1, x2, y2):
        """Define the door area in the frame.
        
        Args:
            x1: Left coordinate of door area
            y1: Top coordinate of door area
            x2: Right coordinate of door area
            y2: Bottom coordinate of door area
            
        Returns:
            True if door area was set successfully
        """
        self.door_area = (x1, y1, x2, y2)
        self.door_defined = True
        # Reset counters when door area is changed
        self.left_to_right = 0
        self.right_to_left = 0
        self.previous_centers = {}
        logger.info(f"Door area set to: {self.door_area}")
        return True

    def set_inside_direction(self, direction):
        """Set which direction is considered 'inside' (left/right).
        
        Args:
            direction: String indicating inside direction ("left" or "right")
            
        Returns:
            True if valid direction was set, False otherwise
        """
        if direction in ["left", "right"]:
            self.inside_direction = direction
            logger.info(f"Inside direction set to: {direction}")
            return True
        return False

    def preprocess_image(self, image):
        """Preprocess image for Faster R-CNN.
        
        Args:
            image: OpenCV image frame
            
        Returns:
            Preprocessed tensor for model input
        """
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        return transform(image)

    def get_box_center(self, box):
        """Calculate center point of bounding box.
        
        Args:
            box: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            (center_x, center_y) coordinates
        """
        return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

    def is_crossing_door(self, prev_center, current_center):
        """Determine if a person is crossing the door area.
        
        Args:
            prev_center: Previous center position (x, y)
            current_center: Current center position (x, y)
            
        Returns:
            (is_crossing, direction) tuple where direction is 
            "left_to_right" or "right_to_left" if crossing
        """
        if not self.door_defined or prev_center is None or self.door_area is None:
            return False, None
            
        x1, y1, x2, y2 = self.door_area
        door_center_x = (x1 + x2) / 2
        
        # Calculate if the person was on the left/right of door before and after
        prev_x = prev_center[0]
        curr_x = current_center[0]
        
        # Check if person crossed the door center line
        if (prev_x < door_center_x and curr_x >= door_center_x):
            return True, "left_to_right"
        elif (prev_x >= door_center_x and curr_x < door_center_x):
            return True, "right_to_left"
        
        return False, None

    def is_in_door_area(self, center):
        """Check if a point is within the door area.
        
        Args:
            center: (x, y) position
            
        Returns:
            True if point is inside the door area
        """
        if not self.door_defined or self.door_area is None:
            return False
            
        x, y = center
        x1, y1, x2, y2 = self.door_area
        return (x1 <= x <= x2) and (y1 <= y <= y2)

    def track_movement(self, current_boxes, frame_width):
        """Track movement direction of detected people.
        
        Args:
            current_boxes: List of current bounding boxes
            frame_width: Width of the frame for center line detection
            
        Returns:
            Dictionary with movement counts
        """
        if not self.door_defined:
            # Fall back to center line detection if door not defined
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
        else:
            # Use door area detection
            current_centers = {}
            movement_count = {"left_to_right": 0, "right_to_left": 0}

            # Calculate centers for current boxes
            for box in current_boxes:
                center = self.get_box_center(box)
                center_x, center_y = center
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

                current_centers[matched_id] = center

                # Check if person is in or near door area and track movement
                if matched_id in self.previous_centers:
                    crossed, direction = self.is_crossing_door(
                        self.previous_centers[matched_id], center
                    )
                    
                    if crossed:
                        movement_count[direction] += 1
                        if direction == "left_to_right":
                            self.left_to_right += 1
                            logger.debug(f"Person {matched_id} moved left to right through door")
                        else:
                            self.right_to_left += 1
                            logger.debug(f"Person {matched_id} moved right to left through door")

        # Update previous centers
        self.previous_centers = current_centers
        return movement_count
        
    def get_entry_exit_count(self):
        """Return the number of entries and exits based on the inside direction.
        
        Returns:
            (entries, exits) tuple with counts
        """
        if self.inside_direction == "right":
            entries = self.left_to_right
            exits = self.right_to_left
        else:
            entries = self.right_to_left
            exits = self.left_to_right
            
        return entries, exits
        
    def reset_counters(self):
        """Reset all movement counters and tracking state."""
        self.left_to_right = 0
        self.right_to_left = 0
        self.previous_centers = {}
        self.track_id = 0
        logger.info("Movement counters have been reset")

    def detect_people(self, image, score_threshold=None, iou_threshold=None):
        """Detect people using Faster R-CNN.
        
        Args:
            image: Input image frame
            score_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
            
        Returns:
            (people_boxes, movement_data) tuple
        """
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
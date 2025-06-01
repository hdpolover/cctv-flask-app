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
            
        # Check CUDA availability and set memory management
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            # Set CUDA to use TensorCores if available
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Get processing device preference from config
        self.use_gpu = config.get('USE_GPU', True)
        
        # Set up device (CPU/GPU)
        if self.use_gpu and self.cuda_available:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        logger.info(f"Initializing detection model on {self.device} (CUDA available: {self.cuda_available})")
        
        # Load model and ensure it's in eval mode
        self.model = self.load_model()
        self.model.to(self.device)
        self.model.eval()  # Ensure model is in eval mode
        
        # Create transform pipeline on device
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
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
        self.inside_direction = "right"  # can be "left", "right", "up", or "down"
        # Track movements in all directions
        self.left_to_right = 0
        self.right_to_left = 0
        self.top_to_bottom = 0
        self.bottom_to_top = 0

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
        """Set which direction is considered 'inside'.
        
        Args:
            direction: String indicating inside direction ("left", "right", "up", or "down")
            
        Returns:
            True if valid direction was set, False otherwise
        """
        if direction in ["left", "right", "up", "down"]:
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
        
        Args:            box: Bounding box coordinates [x1, y1, x2, y2]
            
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
            "left_to_right", "right_to_left", "top_to_bottom", or "bottom_to_top" if crossing
        """
        if not self.door_defined or prev_center is None or self.door_area is None:
            return False, None
            
        x1, y1, x2, y2 = self.door_area
        door_width = x2 - x1
        door_height = y2 - y1
        
        # Calculate if the person was on either side of door before and after
        prev_x, prev_y = prev_center
        curr_x, curr_y = current_center
        
        # Determine door orientation based on aspect ratio
        is_vertical_door = door_height > door_width
        
        if is_vertical_door:
            # For vertical doors, use zones instead of center line
            # Create detection zones with buffer for more reliable detection
            zone_width = door_width * 0.3  # 30% of door width for each zone
            left_zone_right = x1 + zone_width
            right_zone_left = x2 - zone_width
            
            # Check if both positions are within the door height range
            if (y1 <= prev_y <= y2) or (y1 <= curr_y <= y2):
                # Left to right crossing: was in left zone, now in right zone or beyond
                if (prev_x <= left_zone_right and curr_x >= right_zone_left):
                    return True, "left_to_right"
                # Right to left crossing: was in right zone, now in left zone or beyond
                elif (prev_x >= right_zone_left and curr_x <= left_zone_right):
                    return True, "right_to_left"
        else:
            # For horizontal doors, use zones instead of center line
            zone_height = door_height * 0.3  # 30% of door height for each zone
            top_zone_bottom = y1 + zone_height
            bottom_zone_top = y2 - zone_height
            
            # Check if both positions are within the door width range
            if (x1 <= prev_x <= x2) or (x1 <= curr_x <= x2):
                # Top to bottom crossing: was in top zone, now in bottom zone or beyond
                if (prev_y <= top_zone_bottom and curr_y >= bottom_zone_top):
                    return True, "top_to_bottom"
                # Bottom to top crossing: was in bottom zone, now in top zone or beyond
                elif (prev_y >= bottom_zone_top and curr_y <= top_zone_bottom):
                    return True, "bottom_to_top"
        
        # Fallback to center line detection for edge cases
        door_center_x = (x1 + x2) / 2
        door_center_y = (y1 + y2) / 2
        
        # Check if person crossed the horizontal center line (with door area constraint)
        if (y1 <= prev_y <= y2) and (y1 <= curr_y <= y2):
            if (prev_x < door_center_x and curr_x >= door_center_x):
                return True, "left_to_right"
            elif (prev_x >= door_center_x and curr_x < door_center_x):
                return True, "right_to_left"
        
        # Check if person crossed the vertical center line (with door area constraint)
        if (x1 <= prev_x <= x2) and (x1 <= curr_x <= x2):
            if (prev_y < door_center_y and curr_y >= door_center_y):
                return True, "top_to_bottom"
            elif (prev_y >= door_center_y and curr_y < door_center_y):
                return True, "bottom_to_top"
        
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
        else:            # Use door area detection
            current_centers = {}
            movement_count = {
                "left_to_right": 0, 
                "right_to_left": 0,
                "top_to_bottom": 0,
                "bottom_to_top": 0
            }

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
                        elif direction == "right_to_left":
                            self.right_to_left += 1
                            logger.debug(f"Person {matched_id} moved right to left through door")
                        elif direction == "top_to_bottom":
                            self.top_to_bottom += 1
                            logger.debug(f"Person {matched_id} moved top to bottom through door")
                        elif direction == "bottom_to_top":
                            self.bottom_to_top += 1
                            logger.debug(f"Person {matched_id} moved bottom to top through door")

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
        elif self.inside_direction == "left":
            entries = self.right_to_left
            exits = self.left_to_right
        elif self.inside_direction == "down":
            entries = self.top_to_bottom
            exits = self.bottom_to_top
        elif self.inside_direction == "up":
            entries = self.bottom_to_top
            exits = self.top_to_bottom
        else:
            # Default fallback
            entries = self.left_to_right
            exits = self.right_to_left
            
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
         
        # Detailed timing for performance analysis
        timing = {}
        
        # Timing: Start of entire detection process
        start_time = time.time()        # Timing: Preprocessing
        preprocess_start = time.time()
        # Convert image to tensor and move to correct device in one operation
        image_tensor = self.transform(image).contiguous().to(self.device, non_blocking=True)
        timing['preprocess'] = time.time() - preprocess_start
        
        # Timing: Model inference (this is the GPU/CPU intensive part)
        inference_start = time.time()
        with torch.cuda.amp.autocast(enabled=self.cuda_available):
            with torch.no_grad():
                outputs = self.model([image_tensor])
        if self.cuda_available:
            torch.cuda.synchronize()  # Ensure GPU operations are complete
        inference_time = time.time() - inference_start
        timing['inference'] = inference_time
        
        # Timing: Post-processing
        postprocess_start = time.time()
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
        timing['postprocess'] = time.time() - postprocess_start

        # Timing: Tracking
        tracking_start = time.time()
        movement = self.track_movement(people_boxes, image.shape[1])
        timing['tracking'] = time.time() - tracking_start
        
        # Timing: Total detection time
        total_time = time.time() - start_time
        timing['total'] = total_time
        
        # Log detailed timing information at debug level
        logger.debug(
            f"Detection timing: "
            f"Total={timing['total']:.3f}s, "
            f"Preprocess={timing['preprocess']:.3f}s, "
            f"Inference={timing['inference']:.3f}s ({100*timing['inference']/timing['total']:.1f}%), "
            f"Postprocess={timing['postprocess']:.3f}s, "
            f"Tracking={timing['tracking']:.3f}s, "
            f"Device={self.device}"
        )
        
        # Store timing info as an attribute so it can be accessed by video service
        self.last_timing = timing

        return people_boxes, movement
    
    def set_processing_device(self, use_gpu):
        """Switch between CPU and GPU for processing.
        
        Args:
            use_gpu: Boolean indicating whether to use GPU acceleration
            
        Returns:
            Dict with status and device information
        """
        self.use_gpu = use_gpu
        
        # Determine device based on preference and availability
        if use_gpu and self.cuda_available:
            new_device = torch.device("cuda")
            # Enable CUDA optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            new_device = torch.device("cpu")
            # Reset CUDA optimizations
            torch.backends.cudnn.benchmark = False
        
        # Only reload model if device changed
        if new_device != self.device:
            # Log the change
            logger.info(f"Switching processing device from {self.device} to {new_device}")
            
            # Clear CUDA cache if switching away from GPU
            if str(self.device) == "cuda" and torch.cuda.is_available():
                logger.info("Clearing CUDA cache")
                torch.cuda.empty_cache()
                
            self.device = new_device
            
            # Reload the model entirely for clean state
            logger.info(f"Reloading model for {self.device}")
            self.model = self.load_model()
            self.model.to(self.device)
            self.model.eval()  # Ensure model is in eval mode
            logger.info(f"Model reloaded and moved to {self.device}")
            
            # Perform a warm-up inference to initialize device-specific optimizations
            if str(self.device) == "cuda":
                logger.info("Performing warm-up inference on GPU")
                dummy_input = torch.zeros(1, 3, 640, 480).to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        self.model([dummy_input])
                torch.cuda.synchronize()
        
        return {
            "success": True,
            "device": str(self.device),
            "cuda_available": self.cuda_available
        }
    
    def get_door_area(self):
        """Get the current door area coordinates.
        
        Returns:
            Tuple of (x1, y1, x2, y2) or None if not defined
        """
        if self.door_defined and self.door_area:
            return self.door_area
        return None
    
    def get_inside_direction(self):
        """Get the current inside direction setting.
        
        Returns:
            String indicating inside direction ("left", "right", "up", or "down")
        """
        return self.inside_direction
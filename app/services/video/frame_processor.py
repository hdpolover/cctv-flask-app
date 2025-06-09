"""
Video frame processing utilities.
"""
import cv2
import time
import logging
import numpy as np
from app.services.video.ui_utils import UIUtils

# Configure logging
logger = logging.getLogger(__name__)

class FrameProcessor:
    """Processor for video frames with detection and annotations."""
    
    def __init__(self, detection_model, resolution=(640, 480), frame_rate=30):
        """Initialize the frame processor.
        
        Args:
            detection_model: The object detection model
            resolution: Resolution as (width, height) tuple
            frame_rate: Target frame rate for processing
        """
        self.detection_model = detection_model
        self.resolution = resolution
        self.frame_rate = frame_rate
        
        # FPS calculation
        self.fps = 0
        self.frame_times = []
        self.max_frame_samples = 30  # Number of frames to average for FPS
        self.last_processed_time = 0
        
    def process_frame(self, frame):
        """Process a single frame with people detection.
        
        Args:
            frame: Input video frame
            
        Returns:
            Processed frame with annotations
        """
        if frame is None:
            return None
            
        try:
            # Start frame processing time measurement
            frame_start_time = time.time()
            
            # Check if we should process this frame
            current_time = time.time()
            if current_time - self.last_processed_time < 1.0 / self.frame_rate:
                # Always add timestamp even when skipping detection
                frame = UIUtils.add_timestamp(frame)
                return frame  # Skip processing if we're ahead of schedule
            
            # Detect people in the frame
            people_boxes, movement = self.detection_model.detect_people(frame)
            
            # Calculate FPS
            frame_end_time = time.time()
            process_time = frame_end_time - frame_start_time
            self.last_processed_time = current_time
            
            # Update FPS calculation
            self.frame_times.append(process_time)
            if len(self.frame_times) > self.max_frame_samples:
                self.frame_times.pop(0)  # Remove oldest frame time
            
            # Calculate average FPS from frame times
            if self.frame_times:
                avg_process_time = sum(self.frame_times) / len(self.frame_times)
                self.fps = 1.0 / avg_process_time if avg_process_time > 0 else 0
            
            # Draw detection boxes
            for box in people_boxes:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # If door area is defined, draw it
            if self.detection_model.door_defined and self.detection_model.door_area:
                x1, y1, x2, y2 = self.detection_model.door_area
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Draw door center line
                door_center_x = int((x1 + x2) / 2)
                door_center_y = int((y1 + y2) / 2)
                
                # Draw vertical center line
                cv2.line(frame, (door_center_x, y1), (door_center_x, y2), (255, 0, 0), 2)
                
                # Draw horizontal center line
                cv2.line(frame, (x1, door_center_y), (x2, door_center_y), (255, 0, 0), 2)
                  # Label inside/outside directions based on selected inside direction
                if self.detection_model.inside_direction == "right":
                    cv2.putText(frame, "Luar", (x1 - 80, door_center_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, "Dalam", (x2 + 10, door_center_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                elif self.detection_model.inside_direction == "left":
                    cv2.putText(frame, "Dalam", (x1 - 80, door_center_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, "Luar", (x2 + 10, door_center_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                elif self.detection_model.inside_direction == "down":
                    cv2.putText(frame, "Luar", (door_center_x - 30, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, "Dalam", (door_center_x - 30, y2 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                elif self.detection_model.inside_direction == "up":
                    cv2.putText(frame, "Dalam", (door_center_x - 30, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, "Luar", (door_center_x - 30, y2 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                # Draw center line if no door defined (fallback)
                height, width = frame.shape[:2]
                cv2.line(frame, (width//2, 0), (width//2, height), (255, 0, 0), 2)
            
            # Get entry/exit count but don't display on frame
            entries, exits = self.detection_model.get_entry_exit_count()
            people_in_room = max(0, entries - exits)
            
            # Add processing mode and FPS information - one line at the bottom, in Indonesian
            height = frame.shape[0]
            # Draw a semi-transparent background for better readability
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, height-30), (400, height-5), (0, 0, 0), -1)
            alpha = 0.6  # Transparency factor
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Add all performance info in one line in Indonesian, smaller font
            device_type = "GPU" if self.detection_model.device.type == "cuda" else "CPU"
            info_text = f"Pemrosesan: {device_type} | FPS: {self.fps:.1f}"
            
            # Add detailed timing information if available
            if hasattr(self.detection_model, 'last_timing'):
                timing = self.detection_model.last_timing
                inference_ms = timing.get('inference', 0) * 1000
                total_ms = timing.get('total', 0) * 1000
                
                info_text += f" | Inferensi: {inference_ms:.1f}ms | Total: {total_ms:.1f}ms"
                
                if timing.get('total', 0) > 0:
                    inference_percent = 100 * timing.get('inference', 0) / timing.get('total', 1)
                    info_text += f" | Persentase Inferensi: {inference_percent:.1f}%"
            
            # Display all info in one line with smaller font
            cv2.putText(frame, info_text, (10, height-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                
            # Add timestamp to the frame
            frame = UIUtils.add_timestamp(frame)
                
            return frame
            
        except Exception as e:
            logger.exception(f"Error processing frame: {e}")
            return frame
            
    def create_error_frame(self, consecutive_failures):
        """Create an error frame to display when video source is unavailable.
        
        Args:
            consecutive_failures: Number of consecutive connection failures
            
        Returns:
            Frame with error message
        """
        # Create a black frame
        frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        # Add error message
        cv2.putText(frame, "Video Source Unavailable", (int(self.resolution[0]/2) - 150, int(self.resolution[1]/2) - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # Add reconnection message
        cv2.putText(frame, f"Reconnecting... (Attempt {consecutive_failures})", 
                   (int(self.resolution[0]/2) - 180, int(self.resolution[1]/2) + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)
        
        # Add timestamp using the add_timestamp method for consistency
        frame = UIUtils.add_timestamp(frame)
        
        return frame
    
    def create_test_pattern_frame(self):
        """Create a test pattern frame for debugging video stream issues.
        
        Returns:
            Frame with test pattern (color bars)
        """
        # Create a frame with color bars for debugging
        frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        # Create vertical color bars
        bar_width = self.resolution[0] // 8
        colors = [
            (255, 255, 255),  # White
            (255, 255, 0),    # Yellow
            (0, 255, 255),    # Cyan
            (0, 255, 0),      # Green
            (255, 0, 255),    # Magenta
            (255, 0, 0),      # Red
            (0, 0, 255),      # Blue
            (0, 0, 0)         # Black
        ]
        
        for i, color in enumerate(colors):
            x1 = i * bar_width
            x2 = min((i + 1) * bar_width, self.resolution[0])
            frame[:, x1:x2] = color
        
        # Add text overlay
        cv2.putText(frame, "RTSP Stream Test Pattern", 
                   (int(self.resolution[0]/2) - 150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, "If you see this, video processing is working", 
                   (int(self.resolution[0]/2) - 200, self.resolution[1] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add timestamp
        frame = UIUtils.add_timestamp(frame)
        
        return frame

    def get_door_area(self):
        """Get the current door area coordinates.
        
        Returns:
            Tuple of (x1, y1, x2, y2) or None if not defined
        """
        if self.detection_model and hasattr(self.detection_model, 'door_area'):
            return self.detection_model.door_area
        return None
        
    def get_inside_direction(self):
        """Get the current inside direction.
        
        Returns:
            String indicating inside direction ("left", "right", "up", or "down")
        """
        if self.detection_model and hasattr(self.detection_model, 'inside_direction'):
            return self.detection_model.inside_direction
        return "right"  # Default value
    
    def get_people_count(self):
        """Get the current number of people in room.
        
        Returns:
            Integer count of people currently in the room
        """
        if self.detection_model:
            entries, exits = self.detection_model.get_entry_exit_count()
            return max(0, entries - exits)
        return 0

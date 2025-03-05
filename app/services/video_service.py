"""
Video streaming service for capturing and processing video frames
"""
import cv2
import time
import base64
import threading
import logging
from flask import current_app

# Configure logging
logger = logging.getLogger(__name__)

class VideoService:
    """Service for handling video capture and processing"""
    
    def __init__(self, detection_model, socketio, video_path=0, frame_rate=30, resolution=(640, 480)):
        """Initialize the video streaming service.
        
        Args:
            detection_model: Detection model instance for people detection
            socketio: SocketIO instance for real-time updates
            video_path: Path to video file or camera index/URL
            frame_rate: Target frame rate for processing
            resolution: Resolution as (width, height) tuple
        """
        self.detection_model = detection_model
        self.socketio = socketio
        self.video_path = video_path
        self.frame_rate = frame_rate
        self.resolution = resolution
        
        # Initialize camera
        self.cap = self._initialize_capture()
        
        # Threading control
        self.is_running = False
        self.thread = None
        
        # Frame cache
        self.current_frame = None
        self.last_processed_time = 0
    
    def _initialize_capture(self):
        """Initialize the video capture object.
        
        Returns:
            OpenCV VideoCapture object
        """
        try:
            # Check if path is a number (camera index)
            if isinstance(self.video_path, str) and self.video_path.isdigit():
                self.video_path = int(self.video_path)
                
            cap = cv2.VideoCapture(self.video_path)
            
            # Check if camera opened successfully
            if not cap.isOpened():
                logger.error(f"Failed to open video source: {self.video_path}")
                # Try to use default camera as fallback
                if self.video_path != 0:
                    logger.info("Attempting to open default camera instead")
                    cap = cv2.VideoCapture(0)
            
            logger.info(f"Video capture initialized with source: {self.video_path}")
            return cap
        
        except Exception as e:
            logger.exception(f"Error initializing video capture: {e}")
            return cv2.VideoCapture(0)  # Fallback to default camera
    
    def update_settings(self, video_path=None, frame_rate=None, resolution=None):
        """Update video capture settings.
        
        Args:
            video_path: New video path or camera index/URL
            frame_rate: New frame rate
            resolution: New resolution as (width, height) tuple
        """
        restart_capture = False
        
        if video_path is not None and video_path != self.video_path:
            self.video_path = video_path
            restart_capture = True
            
        if frame_rate is not None:
            self.frame_rate = frame_rate
            
        if resolution is not None:
            self.resolution = resolution
        
        # If video source changed, reinitialize the capture
        if restart_capture:
            self.cap.release()
            self.cap = self._initialize_capture()
            
        logger.info(f"Video settings updated: path={self.video_path}, "
                    f"frame_rate={self.frame_rate}, resolution={self.resolution}")
    
    def get_frame(self):
        """Get current video frame.
        
        Returns:
            Current video frame or None if not available
        """
        if self.current_frame is not None:
            return self.current_frame
            
        success, frame = self.cap.read()
        if not success:
            logger.warning("Failed to read frame from video source")
            return None
        
        # Resize frame to target resolution
        return cv2.resize(frame, self.resolution)
    
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
            # Detect people in the frame
            people_boxes, movement = self.detection_model.detect_people(frame)
            
            # Draw detection boxes
            for box in people_boxes:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # If door area is defined, draw it
            if self.detection_model.door_defined and self.detection_model.door_area:
                x1, y1, x2, y2 = self.detection_model.door_area
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Draw door center line
                door_center_x = int((x1 + x2) / 2)
                cv2.line(frame, (door_center_x, y1), (door_center_x, y2), (255, 0, 0), 2)
                
                # Label inside/outside directions
                if self.detection_model.inside_direction == "right":
                    cv2.putText(frame, "Outside", (x1 - 80, y1 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, "Inside", (x2 + 10, y1 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    cv2.putText(frame, "Inside", (x1 - 80, y1 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, "Outside", (x2 + 10, y1 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                # Draw center line if no door defined (fallback)
                height, width = frame.shape[:2]
                cv2.line(frame, (width//2, 0), (width//2, height), (255, 0, 0), 2)
            
            # Add counter information to the video frame
            entries, exits = self.detection_model.get_entry_exit_count()
            people_in_room = max(0, entries - exits)
            
            cv2.putText(frame, f"Entries: {entries}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Exits: {exits}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"People in room: {people_in_room}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
            return frame
            
        except Exception as e:
            logger.exception(f"Error processing frame: {e}")
            return frame
    
    def get_jpeg_frame(self):
        """Get current frame as JPEG bytes.
        
        Returns:
            JPEG-encoded frame as bytes or None if no frame available
        """
        frame = self.get_frame()
        if frame is None:
            return None
            
        processed_frame = self.process_frame(frame)
        if processed_frame is None:
            return None
            
        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            logger.error("Failed to encode frame as JPEG")
            return None
            
        return buffer.tobytes()
    
    def generate_frames(self):
        """Generate a sequence of frames for HTTP streaming.
        
        Yields:
            JPEG frames for multipart HTTP response
        """
        while True:
            # Limit frame rate to target FPS
            current_time = time.time()
            time_elapsed = current_time - self.last_processed_time
            
            if time_elapsed < (1.0 / self.frame_rate):
                time.sleep((1.0 / self.frame_rate) - time_elapsed)
            
            self.last_processed_time = time.time()
            
            # Get and process frame
            frame_bytes = self.get_jpeg_frame()
            
            if frame_bytes is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # If frame capture failed, wait briefly before trying again
                time.sleep(0.1)
    
    def start_capture_thread(self):
        """Start background thread for continuous frame capture.
        
        Returns:
            True if thread started successfully
        """
        if self.is_running:
            logger.warning("Video capture thread is already running")
            return False
            
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_frames)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Video capture thread started")
        return True
    
    def stop_capture_thread(self):
        """Stop the background capture thread.
        
        Returns:
            True if thread stopped successfully
        """
        if not self.is_running:
            logger.warning("Video capture thread is not running")
            return False
            
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
            
        logger.info("Video capture thread stopped")
        return True
    
    def _capture_frames(self):
        """Background thread function for continuous frame capture."""
        logger.info("Started continuous frame capture")
        
        while self.is_running:
            try:
                # Read a frame
                success, frame = self.cap.read()
                
                if not success:
                    logger.warning("Failed to read frame from video source")
                    # Attempt to reconnect if using a camera
                    if isinstance(self.video_path, int) or (isinstance(self.video_path, str) and self.video_path.isdigit()):
                        self.cap.release()
                        self.cap = self._initialize_capture()
                    time.sleep(1.0)
                    continue
                
                # Resize and store current frame
                resized_frame = cv2.resize(frame, self.resolution)
                self.current_frame = resized_frame
                
                # Process frame for socketio broadcast (optional)
                if self.socketio:
                    processed_frame = self.process_frame(resized_frame.copy())
                    ret, buffer = cv2.imencode('.jpg', processed_frame)
                    if ret:
                        frame_encoded = base64.b64encode(buffer).decode('utf-8')
                        self.socketio.emit('video_frame', frame_encoded)
                
                # Sleep to maintain frame rate
                time.sleep(1.0 / self.frame_rate)
                
            except Exception as e:
                logger.exception(f"Error in capture thread: {e}")
                time.sleep(1.0)
    
    def release(self):
        """Release resources when service is no longer needed."""
        self.stop_capture_thread()
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        logger.info("Video service resources released")
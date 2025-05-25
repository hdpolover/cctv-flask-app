"""
Video streaming service for capturing and processing video frames
"""
import cv2
import time
import base64
import threading
import logging
import os
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
        
        # FPS calculation
        self.fps = 0
        self.frame_times = []
        self.max_frame_samples = 30  # Number of frames to average for FPS
    
    def _initialize_capture(self):
        """Initialize the video capture object.
        
        Returns:
            OpenCV VideoCapture object
        """
        try:
            # Initialize cap to None to avoid reference before assignment issues
            cap = None
            
            # Convert string digit path to integer
            if isinstance(self.video_path, str) and self.video_path.isdigit():
                self.video_path = int(self.video_path)
            
            # Initialize capture - use DirectShow on Windows for better performance
            if isinstance(self.video_path, int):
                cap = cv2.VideoCapture(self.video_path, cv2.CAP_DSHOW)
            else:  # Handle file paths or URLs
                cap = cv2.VideoCapture(self.video_path)
            
            # Check if camera opened successfully
            if not cap.isOpened():
                logger.error(f"Failed to open video source: {self.video_path}")
                # Try to use default camera as fallback
                if self.video_path != 0:
                    logger.info("Attempting to open default camera instead")
                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        raise RuntimeError("Failed to open default camera as fallback")
            
            # Set camera buffer size to minimum to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set capture properties for performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, self.frame_rate)
            
            logger.info(f"Video capture initialized with source: {self.video_path}")
            return cap
        
        except Exception as e:
            logger.exception(f"Error initializing video capture: {e}")
            # Create a fresh capture as fallback
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
            
        # Check if we need to skip frames to catch up
        current_time = time.time()
        time_since_last = current_time - self.last_processed_time
        frames_to_skip = int(time_since_last * self.frame_rate) - 1
        
        # Skip frames if we're falling behind
        if frames_to_skip > 0:
            for _ in range(frames_to_skip):
                self.cap.grab()  # Just grab frame, don't decode
            logger.debug(f"Skipped {frames_to_skip} frames to catch up")
        
        success, frame = self.cap.read()
        if not success:
            logger.warning("Failed to read frame from video source")
            return None
        
        # Use CUDA-enabled resize if available
        try:
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_frame = cv2.cuda.resize(gpu_frame, self.resolution)
                frame = gpu_frame.download()
            else:
                frame = cv2.resize(frame, self.resolution)
        except Exception as e:
            frame = cv2.resize(frame, self.resolution)
            
        self.last_processed_time = current_time
        return frame
    
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
                    cv2.putText(frame, "Outside", (x1 - 80, door_center_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, "Inside", (x2 + 10, door_center_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                elif self.detection_model.inside_direction == "left":
                    cv2.putText(frame, "Inside", (x1 - 80, door_center_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, "Outside", (x2 + 10, door_center_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                elif self.detection_model.inside_direction == "down":
                    cv2.putText(frame, "Outside", (door_center_x - 30, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, "Inside", (door_center_x - 30, y2 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                elif self.detection_model.inside_direction == "up":
                    cv2.putText(frame, "Inside", (door_center_x - 30, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, "Outside", (door_center_x - 30, y2 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
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
              # Add processing mode and FPS information
            height = frame.shape[0]
            # Draw a semi-transparent background for better readability
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, height-130), (340, height-5), (0, 0, 0), -1)
            alpha = 0.6  # Transparency factor
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Add processing mode text
            device_type = "GPU" if self.detection_model.device.type == "cuda" else "CPU"
            cv2.putText(frame, f"Processing: {device_type}", (10, height-100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Add FPS counter
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, height-75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Add detailed timing information if available
            if hasattr(self.detection_model, 'last_timing'):
                timing = self.detection_model.last_timing
                
                # Display inference time (model execution)
                inference_ms = timing.get('inference', 0) * 1000
                cv2.putText(frame, f"Inference: {inference_ms:.1f}ms", (10, height-50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Display total time
                total_ms = timing.get('total', 0) * 1000
                cv2.putText(frame, f"Total: {total_ms:.1f}ms", (10, height-25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Display inference percentage
                if timing.get('total', 0) > 0:
                    inference_percent = 100 * timing.get('inference', 0) / timing.get('total', 1)
                    cv2.putText(frame, f"Inference: {inference_percent:.1f}%", (210, height-50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
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
                # Start timing for FPS calculation
                frame_start_time = time.time()
                
                # Read a frame
                success, frame = self.cap.read()
                
                if not success:
                    logger.warning("Failed to read frame from video source")
                    
                    # Check if this is a file source (not a camera/device)
                    is_file = not (isinstance(self.video_path, int) or 
                                (isinstance(self.video_path, str) and self.video_path.isdigit()))
                    
                    # For video files: reopen the file to restart playback
                    if is_file and isinstance(self.video_path, str) and os.path.exists(self.video_path):
                        logger.info(f"Restarting video file: {self.video_path}")
                        self.cap.release()
                        self.cap = cv2.VideoCapture(self.video_path)
                        if not self.cap.isOpened():
                            logger.error(f"Failed to reopen video file: {self.video_path}")
                    # For cameras: attempt to reconnect
                    else:
                        logger.info(f"Attempting to reconnect to camera: {self.video_path}")
                        self.cap.release()
                        self.cap = self._initialize_capture()
                        
                    time.sleep(0.5)  # Brief delay before retry
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
                
                # Calculate and update FPS
                frame_end_time = time.time()
                process_time = frame_end_time - frame_start_time
                self.frame_times.append(process_time)
                if len(self.frame_times) > self.max_frame_samples:
                    self.frame_times.pop(0)  # Remove oldest frame time
                
                # Calculate average FPS from frame times
                if self.frame_times:
                    avg_process_time = sum(self.frame_times) / len(self.frame_times)
                    self.fps = 1.0 / avg_process_time if avg_process_time > 0 else 0
                
                # Sleep to maintain frame rate
                time.sleep(max(0, 1.0 / self.frame_rate - process_time))
                
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
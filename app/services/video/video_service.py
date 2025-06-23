"""
Video streaming service for capturing and processing video frames
"""
import cv2
import time
import base64
import threading
import logging
import os
import numpy as np
from flask import current_app

from app.services.video.video_capture import VideoCaptureManager
from app.services.video.frame_processor import FrameProcessor
from app.services.video.health_monitor import HealthMonitor
from app.services.video.rtsp_monitor import RTSPStreamMonitor
from app.services.video.ui_utils import UIUtils

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
        self.resolution = resolution        # Initialize subsystems
        self.capture_manager = VideoCaptureManager(video_path, frame_rate, resolution)
        self.health_monitor = HealthMonitor()
        self.frame_processor = FrameProcessor(detection_model, resolution, frame_rate)
        
        # Initialize health monitor with first successful read
        if self.capture_manager.cap and self.capture_manager.cap.isOpened():
            self.health_monitor.update_on_success()
        
        # Initialize RTSP stream monitor for automatic reconnection
        self.rtsp_monitor = RTSPStreamMonitor(self, check_interval=15.0)
        if self.capture_manager.is_rtsp:
            self.rtsp_monitor.start()
            logger.info("Started RTSP stream monitor for automatic reconnection")
        
        # Link components to main service
        self.cap = self.capture_manager.cap
        self.is_file = self.capture_manager.is_file
        self.is_rtsp = self.capture_manager.is_rtsp
        self.is_camera = self.capture_manager.is_camera
        
        # Track fallback status
        self.original_video_path = video_path  # Keep track of original source
        self.is_fallback_active = False
        self.fallback_reason = None
        
        # Threading control
        self.is_running = False
        self.thread = None
        
        # Frame cache
        self.current_frame = None
        self.last_processed_time = 0
        
        # FPS calculation from frame processor
        self.fps = 0
    
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
            self.frame_processor.frame_rate = frame_rate
            
        if resolution is not None:
            self.resolution = resolution
            self.frame_processor.resolution = resolution
        
        # If video source changed, reinitialize the capture
        if restart_capture:
            logger.info(f"Video source changed, reinitializing capture manager: {self.video_path}")
            
            # Stop capture thread if running
            was_running = self.is_running
            if was_running:
                self.stop_capture_thread()
                time.sleep(0.5)  # Give thread time to stop
            
            # Release old capture resources
            if hasattr(self, 'capture_manager') and self.capture_manager:
                self.capture_manager.release()
            
            # Reinitialize the capture manager
            self.capture_manager = VideoCaptureManager(self.video_path, self.frame_rate, self.resolution)
            
            # Update references
            self.cap = self.capture_manager.cap
            self.is_file = self.capture_manager.is_file
            self.is_rtsp = self.capture_manager.is_rtsp
            self.is_camera = self.capture_manager.is_camera
            
            # Reset health monitor
            self.health_monitor = HealthMonitor()
            
            # Restart capture thread if it was running
            if was_running:
                self.start_capture_thread()
            
        logger.info(f"Video settings updated: path={self.video_path}, "
                   f"frame_rate={self.frame_rate}, resolution={self.resolution}")
    
    def get_frame(self):
        """Get current video frame.
        
        Returns:
            Current video frame or None if not available
        """
        # Return cached frame if available and recent (within last 2 seconds)
        current_time = time.time()
        if (self.current_frame is not None and 
            current_time - self.last_processed_time < 2.0):
            return self.current_frame
            
        # Check if capture is valid
        if not self.cap or not self.cap.isOpened():
            logger.warning("Capture not initialized or opened in get_frame")
            # Try to reopen the capture
            try:
                self.capture_manager.reopen()
                self.cap = self.capture_manager.cap
                if not self.cap or not self.cap.isOpened():
                    self.health_monitor.update_on_failure()
                    return None
            except Exception as e:
                logger.error(f"Failed to reopen capture in get_frame: {e}")
                self.health_monitor.update_on_failure()
                return None
            
        try:
            # For direct frame reading, don't skip frames to avoid timing issues
            # Only skip frames if we have a recent timestamp
            if self.last_processed_time > 0:
                time_since_last = current_time - self.last_processed_time
                frames_to_skip = int(time_since_last * self.frame_rate) - 1
                
                # Skip frames if we're falling behind (but not for video files)
                if frames_to_skip > 0 and not self.is_file:
                    for _ in range(min(frames_to_skip, 3)):  # Reduced max skipped frames
                        self.cap.grab()  # Just grab frame, don't decode
                    logger.debug(f"Skipped {min(frames_to_skip, 3)} frames to catch up")
            
            # Read frame with timeout protection
            success, frame = self.cap.read()
            
            if not success or frame is None or frame.size == 0:
                logger.warning("Failed to read frame from video source in get_frame")
                self.health_monitor.update_on_failure()
                
                # For RTSP streams, try a quick reconnect
                if self.is_rtsp:
                    try:
                        logger.info("Attempting quick RTSP reconnect...")
                        self.capture_manager.reopen()
                        self.cap = self.capture_manager.cap
                        if self.cap and self.cap.isOpened():
                            success, frame = self.cap.read()
                            if success and frame is not None and frame.size > 0:
                                logger.info("Quick RTSP reconnect successful")
                            else:
                                return None
                        else:
                            return None
                    except Exception as e:
                        logger.error(f"Quick RTSP reconnect failed: {e}")
                        return None
                else:
                    return None
            
            # Update health monitoring on successful frame read
            self.health_monitor.update_on_success()
            
            # Use GPU-accelerated resize if available
            try:
                if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    gpu_frame = cv2.cuda_GpuMat()
                    gpu_frame.upload(frame)
                    gpu_frame = cv2.cuda.resize(gpu_frame, self.resolution)
                    frame = gpu_frame.download()
                else:
                    frame = cv2.resize(frame, self.resolution)
            except Exception as e:
                logger.warning(f"GPU resize failed, falling back to CPU: {e}")
                frame = cv2.resize(frame, self.resolution)
                
            self.last_processed_time = current_time
            self.current_frame = frame  # Update the cached frame
            return frame
            
        except Exception as e:
            logger.exception(f"Error in get_frame: {e}")
            self.health_monitor.update_on_failure()
            return None
    
    def process_frame(self, frame):
        """Process a single frame with people detection.
        
        Args:
            frame: Input video frame
            
        Returns:
            Processed frame with annotations
        """
        if frame is None:
            return None
        
        # Use the frame processor to process the frame
        return self.frame_processor.process_frame(frame)
    
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
        last_error_time = 0
        error_message_cooldown = 5.0  # seconds
        
        while True:
            try:
                # Limit frame rate to target FPS
                current_time = time.time()
                time_elapsed = current_time - self.last_processed_time
                
                if time_elapsed < (1.0 / self.frame_rate):
                    time.sleep((1.0 / self.frame_rate) - time_elapsed)
                
                self.last_processed_time = time.time()
                
                # Check connection health
                health_info = self.check_connection_health()
                if not health_info["healthy"]:
                    # If connection is unhealthy, wait briefly and try to recover
                    if current_time - last_error_time > error_message_cooldown:
                        logger.warning("Connection unhealthy during streaming, attempting to recover...")
                        last_error_time = current_time
                    
                    # Try to reconnect if needed
                    if self.health_monitor.should_reconnect():
                        self.capture_manager.reopen()
                        self.cap = self.capture_manager.cap
                        time.sleep(0.5)
                  # Get and process frame
                frame_bytes = self.get_jpeg_frame()
                
                if frame_bytes is not None:
                    # Reset health monitoring on successful frame
                    self.health_monitor.update_on_success()
                    
                    # Yield the frame for streaming
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # If frame capture failed, wait briefly before trying again
                    time.sleep(0.1)
                    
                    # After multiple failures, yield an error frame or test pattern
                    if self.health_monitor.consecutive_failures > 3:
                        # For RTSP streams, create a test pattern to help debug
                        if self.is_rtsp:
                            test_frame = self.frame_processor.create_test_pattern_frame()
                            ret, buffer = cv2.imencode('.jpg', test_frame)
                            if ret:
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                        else:
                            # Create a black frame with error message
                            error_frame = self.frame_processor.create_error_frame(self.health_monitor.consecutive_failures)
                            ret, buffer = cv2.imencode('.jpg', error_frame)
                            if ret:
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            except Exception as e:
                if current_time - last_error_time > error_message_cooldown:
                    logger.exception(f"Error in generate_frames: {e}")
                    last_error_time = current_time
                time.sleep(0.5)
    
    def generate_raw_frames(self):
        """Generate a sequence of raw frames without detection for camera settings.
        
        Yields:
            JPEG frames for multipart HTTP response
        """
        last_error_time = 0
        error_message_cooldown = 5.0  # seconds
        consecutive_frame_failures = 0
        max_consecutive_failures = 10
        demo_fallback_attempted = False
        
        logger.info("Starting raw frame generation for camera preview")
        
        while True:  # Infinite loop for continuous streaming
            try:
                # Limit frame rate to target FPS (but be more generous for previews)
                current_time = time.time()
                
                # For camera preview, use a more conservative frame rate
                preview_fps = min(self.frame_rate, 15)  # Max 15 FPS for preview
                time_elapsed = current_time - self.last_processed_time
                
                if time_elapsed < (1.0 / preview_fps):
                    time.sleep(max(0.1, (1.0 / preview_fps) - time_elapsed))
                
                # Check if we should attempt fallback to webcam
                if (consecutive_frame_failures > max_consecutive_failures and 
                    not demo_fallback_attempted and 
                    self.is_camera):
                    
                    logger.warning("Camera appears to be unavailable, attempting fallback to webcam...")
                    try:
                        # Switch to webcam as fallback
                        fallback_source = 0  # Default webcam
                        logger.info(f"Switching to webcam fallback: {fallback_source}")
                        
                        # Update video service to use webcam
                        self.video_path = fallback_source
                        self.capture_manager.release()
                        time.sleep(1.0)
                        
                        # Create new capture manager with webcam
                        self.capture_manager = VideoCaptureManager(
                            fallback_source, self.frame_rate, self.resolution
                        )
                        self.cap = self.capture_manager.cap
                        
                        # Update source type flags
                        self.is_file = self.capture_manager.is_file
                        self.is_rtsp = self.capture_manager.is_rtsp
                        self.is_camera = self.capture_manager.is_camera
                        
                        # Reset counters
                        consecutive_frame_failures = 0
                        self.health_monitor.consecutive_failures = 0
                        demo_fallback_attempted = True
                        
                        logger.info("Successfully switched to webcam fallback")
                    except Exception as fallback_error:
                        logger.error(f"Failed to switch to webcam fallback: {fallback_error}")
                
                # Check connection health less frequently for raw frames and only for live sources
                if not self.is_file:  # Skip health checks for video files
                    try:
                        health_info = self.check_connection_health()
                        if not health_info["healthy"]:
                            # If connection is unhealthy, wait briefly and try to recover
                            if current_time - last_error_time > error_message_cooldown:
                                logger.warning("Connection unhealthy during raw streaming, attempting to recover...")
                                last_error_time = current_time
                            
                            # Try to reconnect if needed (but only for cameras/RTSP, not files)
                            if self.health_monitor.should_reconnect():
                                logger.info("Attempting to reopen capture for raw frames...")
                                self.capture_manager.reopen()
                                self.cap = self.capture_manager.cap
                                time.sleep(0.5)
                    except Exception as health_error:
                        logger.warning(f"Health check failed: {health_error}")
                
                # Get raw frame without detection processing
                frame = self.get_frame()
                
                if frame is not None:
                    # Reset health monitoring and failure counters on successful frame
                    self.health_monitor.update_on_success()
                    consecutive_frame_failures = 0
                    
                    # Convert to JPEG without detection processing
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        # Yield the frame for streaming
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    else:
                        logger.warning("Failed to encode frame as JPEG")
                        consecutive_frame_failures += 1
                else:
                    # If frame capture failed, increment failure counter
                    consecutive_frame_failures += 1
                    
                    # Wait briefly before trying again
                    time.sleep(0.2)
                    
                    # After multiple failures, yield an error frame or test pattern
                    if consecutive_frame_failures > 5:
                        try:
                            if hasattr(self.frame_processor, 'create_error_frame'):
                                # Create a more informative error frame for raw preview
                                if demo_fallback_attempted:
                                    message = f"Demo Video Issue (Failures: {consecutive_frame_failures})"
                                else:
                                    message = f"Camera Unavailable (Failures: {consecutive_frame_failures})"
                                
                                error_frame = self.frame_processor.create_error_frame(
                                    consecutive_frame_failures, 
                                    message=message
                                )
                            else:
                                # Create a simple error frame if method doesn't exist
                                error_frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
                                
                                if demo_fallback_attempted:
                                    cv2.putText(error_frame, "Demo Video Issue", 
                                              (10, self.resolution[1]//2), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                else:
                                    cv2.putText(error_frame, "Camera Unavailable", 
                                              (10, self.resolution[1]//2), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                    cv2.putText(error_frame, "Check camera connection", 
                                              (10, self.resolution[1]//2 + 40), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                
                                cv2.putText(error_frame, f"Failures: {consecutive_frame_failures}", 
                                          (10, self.resolution[1]//2 + 80), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                            
                            ret, buffer = cv2.imencode('.jpg', error_frame)
                            if ret:
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                        except Exception as error_frame_ex:
                            logger.error(f"Failed to create error frame: {error_frame_ex}")
                            # Yield a minimal error response
                            error_msg = b"Camera unavailable"
                            yield (b'--frame\r\n'
                                   b'Content-Type: text/plain\r\n\r\n' + error_msg + b'\r\n')
            
            except Exception as e:
                current_time = time.time()
                if current_time - last_error_time > error_message_cooldown:
                    logger.exception(f"Error in generate_raw_frames: {e}")
                    last_error_time = current_time
                consecutive_frame_failures += 1
                time.sleep(0.5)
                
                # Yield an error frame even in case of exceptions
                try:
                    error_msg = f"Error: {str(e)[:50]}".encode()
                    yield (b'--frame\r\n'
                           b'Content-Type: text/plain\r\n\r\n' + error_msg + b'\r\n')
                except:
                    pass  # If even error yielding fails, continue loop

    def start_capture_thread(self):
        """Start background thread for continuous frame capture.
        
        Returns:
            True if thread started successfully
        """
        if self.is_running:
            logger.warning("Video capture thread is already running")
            return False
            
        # Make sure any existing thread is properly stopped first
        if self.thread and self.thread.is_alive():
            logger.warning("Existing thread detected, stopping it first...")
            self.stop_capture_thread()
            time.sleep(0.5)  # Brief pause to ensure thread stops
            
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
        
        # Stop RTSP monitor if running
        if hasattr(self, 'rtsp_monitor') and self.rtsp_monitor:
            self.rtsp_monitor.stop()
            
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
            
        logger.info("Video capture thread stopped")
        return True
    
    def _capture_frames(self):
        """Background thread function for continuous frame capture."""
        logger.info("Started continuous frame capture")
        reconnect_backoff = 0.5  # Initial backoff time in seconds
        max_backoff = 5.0  # Maximum backoff time
        
        # Enhanced RTSP monitoring variables
        last_rtsp_health_check = time.time()
        rtsp_health_interval = 30.0  # Check RTSP health every 30 seconds
        rtsp_timeout_threshold = 60.0  # Consider RTSP dead after 60 seconds without frames
        consecutive_read_failures = 0
        max_read_failures = 15  # Allow more failures for RTSP before giving up
        
        while self.is_running:
            try:
                # Start timing for FPS calculation
                frame_start_time = time.time()
                
                # Enhanced RTSP health monitoring
                if self.is_rtsp and (frame_start_time - last_rtsp_health_check) > rtsp_health_interval:
                    time_since_last_success = frame_start_time - self.health_monitor.last_successful_read
                    
                    # If RTSP has been silent for too long, proactively reconnect
                    if time_since_last_success > rtsp_timeout_threshold and self.health_monitor.last_successful_read > 0:
                        logger.warning(f"RTSP stream appears stalled - {time_since_last_success:.1f}s without frames, forcing reconnection")
                        
                        # Force full RTSP reconnection
                        self.capture_manager.release()
                        time.sleep(3.0)  # Give RTSP server time to clean up
                        self.capture_manager = VideoCaptureManager(self.video_path, self.frame_rate, self.resolution)
                        self.cap = self.capture_manager.cap
                        
                        if self.cap and self.cap.isOpened():
                            logger.info("Proactive RTSP reconnection successful")
                            self.is_rtsp = self.capture_manager.is_rtsp
                            self.is_camera = self.capture_manager.is_camera
                            self.health_monitor.consecutive_failures = 0
                            consecutive_read_failures = 0
                        else:
                            logger.error("Proactive RTSP reconnection failed")
                    
                    last_rtsp_health_check = frame_start_time
                
                # Check if we need to reconnect due to health issues (skip for files unless really broken)
                if self.health_monitor.should_reconnect():
                    # For video files, be much more conservative about reconnections
                    if self.is_file and self.health_monitor.consecutive_failures < 10:
                        # For files, only reconnect if we have many consecutive failures
                        pass  # Skip reconnection for files with few failures
                    else:
                        logger.warning(f"Connection appears unhealthy. Reconnecting... (attempt {self.health_monitor.consecutive_failures})")
                        self.capture_manager.reopen()
                        self.cap = self.capture_manager.cap
                        time.sleep(reconnect_backoff)
                        # Increase backoff time for next reconnection attempt (exponential backoff)
                        reconnect_backoff = min(reconnect_backoff * 1.5, max_backoff)
                        
                        # Stop trying after max attempts for non-RTSP sources
                        if not self.is_rtsp and self.health_monitor.exceeded_max_attempts():
                            logger.error(f"Failed to reconnect after {self.health_monitor.max_reconnect_attempts} attempts")
                            # If source is not webcam, switch to webcam as fallback
                            if self.video_path != 0:
                                logger.info("Switching to webcam fallback")
                                self.video_path = 0
                                self.capture_manager = VideoCaptureManager(self.video_path, self.frame_rate, self.resolution)
                                self.cap = self.capture_manager.cap
                                self.is_file = self.capture_manager.is_file
                                self.is_rtsp = self.capture_manager.is_rtsp
                                self.is_camera = self.capture_manager.is_camera
                                self.health_monitor.consecutive_failures = 0
                            else:
                                # Reset consecutive failures but keep trying
                                self.health_monitor.consecutive_failures = 0
                                time.sleep(3.0)  # Longer wait before retry cycle
                        continue
                
                # Read a frame
                if not self.cap or not self.cap.isOpened():
                    logger.error("Video capture is not open, attempting to reinitialize")
                    self.capture_manager.reopen()
                    self.cap = self.capture_manager.cap
                    self.health_monitor.update_on_failure()
                    time.sleep(reconnect_backoff)
                    continue
                    
                success, frame = self.cap.read()
                
                if not success or frame is None or frame.size == 0:
                    logger.warning("Failed to read frame from video source")
                    self.health_monitor.update_on_failure()
                    
                    # Check if video file reached the end
                    if self.is_file:
                        logger.info(f"Video file may have ended, restarting: {self.video_path}")
                        self.capture_manager.reopen()
                        self.cap = self.capture_manager.cap                    # Special handling for RTSP streams with enhanced reconnection
                    elif self.is_rtsp:
                        logger.warning(f"RTSP stream read failed, attempting enhanced reconnection: {self.video_path}")
                        
                        # For RTSP, try multiple reconnection strategies
                        reconnection_successful = False
                        
                        # Strategy 1: Simple reopen
                        self.capture_manager.reopen()
                        self.cap = self.capture_manager.cap
                        if self.cap and self.cap.isOpened():
                            # Test if we can actually read a frame
                            test_ret, test_frame = self.cap.read()
                            if test_ret and test_frame is not None and test_frame.size > 0:
                                logger.info("RTSP reconnection successful with simple reopen")
                                reconnection_successful = True
                        
                        # Strategy 2: Full recreation if simple reopen failed
                        if not reconnection_successful:
                            logger.info("Simple reopen failed, performing full RTSP recreation")
                            self.capture_manager.release()
                            time.sleep(2.0)  # Give RTSP server time to clean up
                            self.capture_manager = VideoCaptureManager(self.video_path, self.frame_rate, self.resolution)
                            self.cap = self.capture_manager.cap
                            if self.cap and self.cap.isOpened():
                                # Test frame read again
                                test_ret, test_frame = self.cap.read()
                                if test_ret and test_frame is not None and test_frame.size > 0:
                                    logger.info("RTSP reconnection successful with full recreation")
                                    reconnection_successful = True
                                    # Reset position to beginning for the test frame
                                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        
                        if not reconnection_successful:
                            logger.error("All RTSP reconnection strategies failed")
                            # Increase backoff time more aggressively for RTSP
                            reconnect_backoff = min(reconnect_backoff * 2.0, max_backoff)
                    # For cameras: attempt to reconnect
                    else:
                        logger.info(f"Attempting to reconnect to camera: {self.video_path}")
                        self.capture_manager.reopen()
                        self.cap = self.capture_manager.cap
                        
                    time.sleep(reconnect_backoff)
                    continue
                  # Reset health monitoring on successful frame read
                self.health_monitor.update_on_success()
                reconnect_backoff = 0.5  # Reset backoff time after successful read
                consecutive_read_failures = 0  # Reset read failure counter
                
                # Update RTSP monitor if active
                if hasattr(self, 'rtsp_monitor') and self.rtsp_monitor.is_running:
                    self.rtsp_monitor.update_frame_time()
                
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
                        # Log every 30 frames to avoid spam
                        if hasattr(self, '_frame_emit_counter'):
                            self._frame_emit_counter += 1
                        else:
                            self._frame_emit_counter = 1
                        
                        if self._frame_emit_counter % 30 == 0:
                            logger.info(f"Emitted video frame #{self._frame_emit_counter} via WebSocket (frame size: {len(frame_encoded)} chars)")
                    else:
                        logger.warning("Failed to encode frame for WebSocket emission")
                
                # Update FPS from frame processor
                self.fps = self.frame_processor.fps
                
                # Calculate frame processing time
                frame_end_time = time.time()
                process_time = frame_end_time - frame_start_time
                
                # Sleep to maintain frame rate
                time.sleep(max(0, 1.0 / self.frame_rate - process_time))
                
            except Exception as e:
                logger.exception(f"Error in capture thread: {e}")
                self.health_monitor.update_on_failure()
                time.sleep(reconnect_backoff)
    
    def check_connection_health(self):
        """Check the health status of the video connection.
        
        Returns:
            Dict with health status information
        """
        health_info = self.health_monitor.check_health(
            is_camera=self.is_camera,
            is_rtsp=self.is_rtsp,
            is_file=self.is_file
        )
        
        # Add FPS to health info
        health_info["fps"] = self.fps
        
        return health_info
    
    def get_video_source_info(self):
        """Get information about the current video source.
        
        Returns:
            Dict with video source information
        """
        # Get basic source info from capture manager
        source_info = self.capture_manager.get_source_info()
        
        # Add health info
        source_info["health"] = self.check_connection_health()
          # Add actual FPS
        source_info["actual_fps"] = self.fps
        
        return source_info
    
    def get_source_status(self):
        """Get detailed information about the video source status.
        
        Returns:
            dict: Status information including original source, current source, and fallback status
        """
        return {
            'original_source': self.original_video_path,
            'current_source': self.video_path,
            'is_fallback_active': self.is_fallback_active,
            'fallback_reason': self.fallback_reason,
            'is_camera': self.is_camera,
            'is_rtsp': self.is_rtsp,
            'is_file': self.is_file,
            'capture_opened': self.cap and self.cap.isOpened() if self.cap else False,
            'health_status': self.health_monitor.check_health(
                is_camera=self.is_camera,
                is_rtsp=self.is_rtsp,
                is_file=self.is_file
            ) if self.health_monitor else None
        }
    
    def activate_fallback(self, reason, fallback_path=None):
        """Activate fallback mode with proper status tracking.
        
        Args:
            reason: Reason for fallback activation
            fallback_path: Fallback source (auto-determined if None)
        """
        if not self.is_fallback_active:
            logger.warning(f"Activating fallback mode: {reason}")
            self.is_fallback_active = True
            self.fallback_reason = reason
            
            # Determine appropriate fallback source
            if fallback_path is None:
                if self.is_camera and self.video_path == 0:
                    # If current source is default camera, try demo video
                    fallback_path = 'app/static/videos/demo.mp4'
                    if not os.path.exists(fallback_path):
                        # If demo video doesn't exist, try camera 1
                        fallback_path = 1
                elif self.is_camera:
                    # If current source is another camera, try default camera
                    fallback_path = 0
                else:
                    # If current source is file/rtsp, try default camera
                    fallback_path = 0
            
            # Switch to fallback source
            old_path = self.video_path
            self.video_path = fallback_path
            
            # Update capture manager
            try:
                self.capture_manager.release()
                time.sleep(0.5)
                self.capture_manager = VideoCaptureManager(fallback_path, self.frame_rate, self.resolution)
                self.cap = self.capture_manager.cap
                self.is_file = self.capture_manager.is_file
                self.is_rtsp = self.capture_manager.is_rtsp
                self.is_camera = self.capture_manager.is_camera
                
                logger.info(f"Fallback activated: {old_path} -> {fallback_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to activate fallback: {e}")
                self.is_fallback_active = False
                self.fallback_reason = None
                return False
        return False
    
    def release(self):
        """Release resources when service is no longer needed."""
        self.stop_capture_thread()
        self.capture_manager.release()
        logger.info("Video service resources released")
        
    def _emit_counter_update(self):
        """Emit counter update event with current counts and system status."""
        if self.socketio:
            try:
                # Get entry and exit counts from detection model
                entries, exits = self.frame_processor.detection_model.get_entry_exit_count()
                people_in_room = max(0, entries - exits)
                
                # Basic counter data
                data = {
                    'people_in_room': people_in_room,
                    'entries': entries,
                    'exits': exits,
                    'fps': self.fps,
                    'resolution': f"{self.resolution[0]} x {self.resolution[1]}",
                    'frame_rate': self.frame_rate
                }
                
                # Add video source information
                if self.is_file:
                    video_source = f"File: {os.path.basename(self.video_path)}"
                elif self.is_rtsp:
                    video_source = "RTSP Stream"
                elif self.is_camera:
                    camera_id = self.video_path
                    video_source = f"Camera #{camera_id}"
                else:
                    video_source = "Unknown"
                    
                data['video_source'] = video_source
                
                # Add door area information if available
                door_area = self.frame_processor.get_door_area()
                if door_area and all(v is not None for v in door_area):
                    data['door_coordinates'] = {
                        'x1': door_area[0],
                        'y1': door_area[1],
                        'x2': door_area[2],
                        'y2': door_area[3]
                    }
                    data['inside_direction'] = self.frame_processor.get_inside_direction()
                
                # Emit the event with all data
                self.socketio.emit('counter_update', data)
                
            except Exception as e:
                logger.error(f"Error emitting counter update: {e}")
                
    def _emit_system_status(self):
        """Emit system status information."""
        if self.socketio:
            try:
                health_data = self.health_monitor.get_status()
                health_data['fps'] = self.fps  # Add current FPS to system status
                
                self.socketio.emit('system_status', health_data)
            except Exception as e:
                logger.error(f"Error emitting system status: {e}")

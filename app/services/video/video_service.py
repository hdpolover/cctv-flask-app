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
            # Reinitialize the capture manager
            self.capture_manager = VideoCaptureManager(self.video_path, self.frame_rate, self.resolution)
            
            # Update references
            self.cap = self.capture_manager.cap
            self.is_file = self.capture_manager.is_file
            self.is_rtsp = self.capture_manager.is_rtsp
            self.is_camera = self.capture_manager.is_camera
            
            # Reset health monitor
            self.health_monitor = HealthMonitor()
            
        logger.info(f"Video settings updated: path={self.video_path}, "
                   f"frame_rate={self.frame_rate}, resolution={self.resolution}")
    
    def get_frame(self):
        """Get current video frame.
        
        Returns:
            Current video frame or None if not available
        """
        # Return cached frame if available
        if self.current_frame is not None:
            return self.current_frame
            
        # Check if capture is valid
        if not self.cap or not self.cap.isOpened():
            logger.warning("Capture not initialized or opened in get_frame")
            self.health_monitor.update_on_failure()
            return None
            
        try:
            # Check if we need to skip frames to catch up
            current_time = time.time()
            time_since_last = current_time - self.last_processed_time
            frames_to_skip = int(time_since_last * self.frame_rate) - 1
            
            # Skip frames if we're falling behind (but not for video files)
            if frames_to_skip > 0 and not self.is_file:
                for _ in range(min(frames_to_skip, 5)):  # Limit max skipped frames
                    self.cap.grab()  # Just grab frame, don't decode
                logger.debug(f"Skipped {min(frames_to_skip, 5)} frames to catch up")
            
            # Read frame with timeout protection
            success, frame = self.cap.read()
            
            if not success or frame is None or frame.size == 0:
                logger.warning("Failed to read frame from video source in get_frame")
                self.health_monitor.update_on_failure()
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
                
                # Get raw frame without detection processing
                frame = self.get_frame()
                
                if frame is not None:
                    # Reset health monitoring on successful frame
                    self.health_monitor.update_on_success()
                    
                    # Convert to JPEG without detection processing
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        # Yield the frame for streaming
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                else:
                    # If frame capture failed, wait briefly before trying again
                    time.sleep(0.1)
                    
                    # After multiple failures, yield an error frame
                    if self.health_monitor.consecutive_failures > 3:
                        # Create a black frame with error message
                        error_frame = self.frame_processor.create_error_frame(self.health_monitor.consecutive_failures)
                        ret, buffer = cv2.imencode('.jpg', error_frame)
                        if ret:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            except Exception as e:
                if current_time - last_error_time > error_message_cooldown:
                    logger.exception(f"Error in generate_raw_frames: {e}")
                    last_error_time = current_time
                time.sleep(0.5)
    
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
                
                # Check if we need to reconnect due to health issues
                if self.health_monitor.should_reconnect():
                    logger.warning(f"Connection appears unhealthy. Reconnecting... (attempt {self.health_monitor.consecutive_failures})")
                    self.capture_manager.reopen()
                    self.cap = self.capture_manager.cap
                    time.sleep(reconnect_backoff)
                    # Increase backoff time for next reconnection attempt (exponential backoff)
                    reconnect_backoff = min(reconnect_backoff * 1.5, max_backoff)
                    
                    # Stop trying after max attempts for non-RTSP sources
                    if not self.is_rtsp and self.health_monitor.exceeded_max_attempts():
                        logger.error(f"Failed to reconnect after {self.health_monitor.max_reconnect_attempts} attempts")
                        # If we have a demo video file configured, switch to it as fallback
                        demo_path = os.path.join(current_app.static_folder, 'videos', 'demo.mp4')
                        if os.path.exists(demo_path) and self.video_path != demo_path:
                            logger.info(f"Switching to demo video: {demo_path}")
                            self.video_path = demo_path
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

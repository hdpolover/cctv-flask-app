"""
Simple camera preview service for camera settings page.
This service provides raw video streaming without detection processing.
"""
import cv2
import time
import threading
import logging
import os
import numpy as np
from flask import current_app

from app.services.video.video_capture import VideoCaptureManager

# Configure logging
logger = logging.getLogger(__name__)

class CameraPreviewService:
    """Lightweight service for camera preview without detection processing"""
    
    def __init__(self, video_path=0, frame_rate=15, resolution=(640, 480)):
        """Initialize the camera preview service.
        
        Args:
            video_path: Path to video file or camera index/URL
            frame_rate: Target frame rate for preview (default: 15 for efficiency)
            resolution: Resolution as (width, height) tuple
        """
        self.video_path = video_path
        self.frame_rate = min(frame_rate, 15)  # Cap at 15 FPS for efficiency
        self.resolution = resolution
        
        # Initialize capture manager
        self.capture_manager = VideoCaptureManager(video_path, self.frame_rate, resolution)
        self.cap = self.capture_manager.cap
        self.is_file = self.capture_manager.is_file
        self.is_rtsp = self.capture_manager.is_rtsp
        self.is_camera = self.capture_manager.is_camera
        
        # Simple status tracking
        self.is_opened = self.cap and self.cap.isOpened()
        self.consecutive_failures = 0
        self.last_frame_time = time.time()
        
        logger.info(f"Camera preview service initialized: {video_path}, "
                   f"FPS: {self.frame_rate}, Resolution: {resolution}")
    
    def update_settings(self, video_path=None, frame_rate=None, resolution=None):
        """Update camera preview settings.
        
        Args:
            video_path: New video path or camera index/URL
            frame_rate: New frame rate (capped at 15 for efficiency)
            resolution: New resolution as (width, height) tuple
        """
        restart_capture = False
        
        if video_path is not None and video_path != self.video_path:
            self.video_path = video_path
            restart_capture = True
            
        if frame_rate is not None:
            self.frame_rate = min(frame_rate, 15)  # Cap at 15 FPS
            
        if resolution is not None:
            self.resolution = resolution
        
        # If video source changed, reinitialize the capture
        if restart_capture:
            logger.info(f"Camera preview source changed: {self.video_path}")
            
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
            self.is_opened = self.cap and self.cap.isOpened()
            self.consecutive_failures = 0
            
        logger.info(f"Camera preview settings updated: path={self.video_path}, "
                   f"frame_rate={self.frame_rate}, resolution={self.resolution}")
    
    def get_frame(self):
        """Get current video frame without any processing.
        
        Returns:
            Raw video frame or None if not available
        """
        # Check if capture is valid
        if not self.cap or not self.cap.isOpened():
            logger.warning("Camera preview capture not opened")
            self.consecutive_failures += 1
            
            # Try to reopen once
            if self.consecutive_failures < 3:
                try:
                    self.capture_manager.reopen()
                    self.cap = self.capture_manager.cap
                    self.is_opened = self.cap and self.cap.isOpened()
                    if self.is_opened:
                        logger.info("Camera preview capture reopened successfully")
                        self.consecutive_failures = 0
                    else:
                        return None
                except Exception as e:
                    logger.error(f"Failed to reopen camera preview capture: {e}")
                    return None
            else:
                return None
            
        try:
            # Read frame
            success, frame = self.cap.read()
            
            if not success or frame is None or frame.size == 0:
                logger.warning("Failed to read frame from camera preview")
                self.consecutive_failures += 1
                
                # For video files, try to restart from beginning
                if self.is_file and self.consecutive_failures < 5:
                    try:
                        logger.info("Video file ended, restarting from beginning")
                        self.capture_manager.reopen()
                        self.cap = self.capture_manager.cap
                        self.consecutive_failures = 0
                        # Try reading again
                        success, frame = self.cap.read()
                        if success and frame is not None and frame.size > 0:
                            return cv2.resize(frame, self.resolution)
                    except Exception as e:
                        logger.error(f"Failed to restart video file: {e}")
                
                return None
            
            # Reset failure counter on success
            self.consecutive_failures = 0
            self.last_frame_time = time.time()
            
            # Resize frame to target resolution
            resized_frame = cv2.resize(frame, self.resolution)
            return resized_frame
            
        except Exception as e:
            logger.error(f"Error in camera preview get_frame: {e}")
            self.consecutive_failures += 1
            return None
    
    def generate_frames(self):
        """Generate a sequence of raw frames for HTTP streaming.
        
        Yields:
            JPEG frames for multipart HTTP response
        """
        logger.info("Starting camera preview frame generation")
        frame_count = 0
        max_frames = 1000  # Prevent infinite streaming
        last_error_time = 0
        error_cooldown = 5.0
        
        while frame_count < max_frames:
            frame_count += 1
            
            try:
                # Rate limiting
                frame_start = time.time()
                
                # Get raw frame
                frame = self.get_frame()
                
                if frame is not None:
                    # Convert to JPEG
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        # Yield the frame
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    else:
                        logger.warning("Failed to encode camera preview frame as JPEG")
                        time.sleep(0.1)
                else:
                    # Create error frame if needed
                    if self.consecutive_failures > 3:
                        error_frame = self._create_error_frame()
                        ret, buffer = cv2.imencode('.jpg', error_frame)
                        if ret:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    
                    time.sleep(0.2)  # Wait before retry
                
                # Frame rate control
                frame_time = time.time() - frame_start
                sleep_time = max(0, (1.0 / self.frame_rate) - frame_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                current_time = time.time()
                if current_time - last_error_time > error_cooldown:
                    logger.error(f"Error in camera preview generate_frames: {e}")
                    last_error_time = current_time
                time.sleep(0.5)
        
        logger.info("Camera preview frame generation stopped")
    
    def _create_error_frame(self):
        """Create a simple error frame for display."""
        frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        # Add error message
        if self.is_camera:
            message = "Camera Unavailable"
            submessage = "Check camera connection"
        elif self.is_rtsp:
            message = "RTSP Stream Error"
            submessage = "Check stream URL"
        else:
            message = "Video File Error"
            submessage = "Check file path"
        
        # Calculate text positions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Main message
        (text_width, text_height), _ = cv2.getTextSize(message, font, font_scale, thickness)
        text_x = (self.resolution[0] - text_width) // 2
        text_y = (self.resolution[1] - text_height) // 2
        cv2.putText(frame, message, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)
        
        # Submessage
        font_scale_sub = 0.5
        thickness_sub = 1
        (text_width_sub, text_height_sub), _ = cv2.getTextSize(submessage, font, font_scale_sub, thickness_sub)
        text_x_sub = (self.resolution[0] - text_width_sub) // 2
        text_y_sub = text_y + text_height + 20
        cv2.putText(frame, submessage, (text_x_sub, text_y_sub), font, font_scale_sub, (0, 255, 255), thickness_sub)
        
        # Failure count
        failure_text = f"Failures: {self.consecutive_failures}"
        text_y_fail = text_y_sub + 30
        cv2.putText(frame, failure_text, (10, text_y_fail), font, 0.4, (255, 255, 0), 1)
        
        return frame
    
    def get_status(self):
        """Get current camera preview status.
        
        Returns:
            Dict with status information
        """
        return {
            'video_path': self.video_path,
            'is_opened': self.is_opened,
            'is_camera': self.is_camera,
            'is_rtsp': self.is_rtsp,
            'is_file': self.is_file,
            'frame_rate': self.frame_rate,
            'resolution': self.resolution,
            'consecutive_failures': self.consecutive_failures,
            'last_frame_time': self.last_frame_time
        }
    
    def release(self):
        """Release camera preview resources."""
        if hasattr(self, 'capture_manager') and self.capture_manager:
            self.capture_manager.release()
        logger.info("Camera preview service released")

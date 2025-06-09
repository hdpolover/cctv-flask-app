"""
Enhanced RTSP Stream Monitor for automatic reconnection and health monitoring.
"""
import time
import threading
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)

class RTSPStreamMonitor:
    """Enhanced RTSP stream monitor with automatic reconnection."""
    
    def __init__(self, video_service, check_interval: float = 15.0):
        """Initialize the RTSP stream monitor.
        
        Args:
            video_service: Reference to the video service to monitor
            check_interval: How often to check stream health (seconds)
        """
        self.video_service = video_service
        self.check_interval = check_interval
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        
        # Health tracking
        self.last_frame_time = time.time()
        self.reconnection_count = 0
        self.last_reconnection_time = 0
        
        # Thresholds
        self.stream_timeout = 60.0  # Consider stream dead after 60 seconds
        self.reconnection_cooldown = 30.0  # Wait 30 seconds between reconnections
        self.max_reconnections_per_hour = 10
        
    def update_frame_time(self):
        """Call this when a frame is successfully processed."""
        self.last_frame_time = time.time()
        
    def start(self):
        """Start the RTSP monitoring."""
        if self.is_running:
            logger.warning("RTSP monitor is already running")
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started RTSP stream monitor with {self.check_interval}s check interval")
        
    def stop(self):
        """Stop the RTSP monitoring."""
        if not self.is_running:
            return
            
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        logger.info("Stopped RTSP stream monitor")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Only monitor if we're dealing with an RTSP stream
                if not (hasattr(self.video_service, 'is_rtsp') and self.video_service.is_rtsp):
                    time.sleep(self.check_interval)
                    continue
                
                # Check stream health
                time_since_last_frame = current_time - self.last_frame_time
                time_since_last_reconnection = current_time - self.last_reconnection_time
                
                # Determine if stream needs reconnection
                needs_reconnection = (
                    time_since_last_frame > self.stream_timeout and
                    time_since_last_reconnection > self.reconnection_cooldown and
                    self.reconnection_count < self.max_reconnections_per_hour
                )
                
                if needs_reconnection:
                    logger.warning(f"RTSP stream health check failed - {time_since_last_frame:.1f}s since last frame")
                    self._trigger_reconnection()
                
                # Reset reconnection count every hour
                if current_time - self.last_reconnection_time > 3600:
                    self.reconnection_count = 0
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in RTSP monitor: {e}")
                time.sleep(self.check_interval)
                
    def _trigger_reconnection(self):
        """Trigger an RTSP stream reconnection."""
        try:
            logger.info("Triggering RTSP stream reconnection")
            
            # Update reconnection tracking
            self.reconnection_count += 1
            self.last_reconnection_time = time.time()
            
            # Perform reconnection
            if hasattr(self.video_service, 'capture_manager') and self.video_service.capture_manager:
                # Release current connection
                self.video_service.capture_manager.release()
                time.sleep(2.0)  # Give time for cleanup
                
                # Recreate capture manager
                from app.services.video.video_capture import VideoCaptureManager
                self.video_service.capture_manager = VideoCaptureManager(
                    self.video_service.video_path,
                    self.video_service.frame_rate,
                    self.video_service.resolution
                )
                self.video_service.cap = self.video_service.capture_manager.cap
                
                # Update stream status
                if self.video_service.cap and self.video_service.cap.isOpened():
                    logger.info("RTSP stream reconnection successful")
                    self.video_service.is_rtsp = self.video_service.capture_manager.is_rtsp
                    self.video_service.is_camera = self.video_service.capture_manager.is_camera
                    
                    # Reset health monitor
                    if hasattr(self.video_service, 'health_monitor'):
                        self.video_service.health_monitor.consecutive_failures = 0
                        self.video_service.health_monitor.update_on_success()
                    
                    # Update frame time
                    self.last_frame_time = time.time()
                else:
                    logger.error("RTSP stream reconnection failed")
                    
        except Exception as e:
            logger.error(f"Error during RTSP reconnection: {e}")
            
    def get_status(self):
        """Get the current monitor status."""
        current_time = time.time()
        time_since_last_frame = current_time - self.last_frame_time
        
        return {
            "running": self.is_running,
            "last_frame_time": self.last_frame_time,
            "time_since_last_frame": time_since_last_frame,
            "stream_healthy": time_since_last_frame < self.stream_timeout,
            "reconnection_count": self.reconnection_count,
            "last_reconnection_time": self.last_reconnection_time
        }

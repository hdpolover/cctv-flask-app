"""
RTSP Stream Watchdog for continuous monitoring and reconnection.
"""
import time
import threading
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)

class RTSPWatchdog:
    """Watchdog that monitors RTSP stream health and triggers reconnections."""
    
    def __init__(self, video_service, check_interval: float = 30.0, timeout_threshold: float = 60.0):
        """Initialize the RTSP watchdog.
        
        Args:
            video_service: Reference to the video service to monitor
            check_interval: How often to check stream health (seconds)
            timeout_threshold: Consider stream dead after this many seconds without frames
        """
        self.video_service = video_service
        self.check_interval = check_interval
        self.timeout_threshold = timeout_threshold
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.last_frame_time = time.time()
        self.reconnection_callback: Optional[Callable] = None
        
    def set_reconnection_callback(self, callback: Callable):
        """Set a callback function to call when reconnection is needed."""
        self.reconnection_callback = callback
        
    def update_last_frame_time(self):
        """Update the timestamp of the last successful frame read."""
        self.last_frame_time = time.time()
        
    def start(self):
        """Start the watchdog monitoring."""
        if self.is_running:
            logger.warning("RTSP watchdog is already running")
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started RTSP watchdog with {self.check_interval}s check interval")
        
    def stop(self):
        """Stop the watchdog monitoring."""
        if not self.is_running:
            return
            
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        logger.info("Stopped RTSP watchdog")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                current_time = time.time()
                time_since_last_frame = current_time - self.last_frame_time
                
                # Check if stream appears to be stalled
                if time_since_last_frame > self.timeout_threshold:
                    logger.warning(f"RTSP stream appears stalled - {time_since_last_frame:.1f}s since last frame")
                    
                    # Only trigger reconnection if we're monitoring an RTSP stream
                    if (hasattr(self.video_service, 'is_rtsp') and self.video_service.is_rtsp):
                        logger.info("Triggering RTSP stream reconnection due to timeout")
                        
                        # Trigger reconnection
                        if self.reconnection_callback:
                            self.reconnection_callback()
                        else:
                            self._default_reconnection()
                        
                        # Reset the frame time to avoid immediate re-triggering
                        self.last_frame_time = current_time
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in RTSP watchdog: {e}")
                time.sleep(self.check_interval)
                
    def _default_reconnection(self):
        """Default reconnection logic if no callback is set."""
        try:
            if hasattr(self.video_service, 'capture_manager') and self.video_service.capture_manager:
                logger.info("Performing default RTSP reconnection")
                
                # Release current connection
                self.video_service.capture_manager.release()
                time.sleep(2.0)  # Give RTSP server time to clean up
                
                # Recreate connection
                from app.services.video.video_capture import VideoCaptureManager
                self.video_service.capture_manager = VideoCaptureManager(
                    self.video_service.video_path,
                    self.video_service.frame_rate,
                    self.video_service.resolution
                )
                self.video_service.cap = self.video_service.capture_manager.cap
                
                if self.video_service.cap and self.video_service.cap.isOpened():
                    logger.info("Default RTSP reconnection successful")
                    # Reset health monitor
                    if hasattr(self.video_service, 'health_monitor'):
                        self.video_service.health_monitor.consecutive_failures = 0
                else:
                    logger.error("Default RTSP reconnection failed")
                    
        except Exception as e:
            logger.error(f"Error during default RTSP reconnection: {e}")
            
    def get_status(self):
        """Get the current watchdog status.
        
        Returns:
            Dict with watchdog status information
        """
        current_time = time.time()
        time_since_last_frame = current_time - self.last_frame_time
        
        return {
            "running": self.is_running,
            "last_frame_time": self.last_frame_time,
            "time_since_last_frame": time_since_last_frame,
            "timeout_threshold": self.timeout_threshold,
            "check_interval": self.check_interval,
            "stream_healthy": time_since_last_frame < self.timeout_threshold
        }

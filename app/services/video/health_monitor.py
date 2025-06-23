"""
Health monitoring for video connections.
"""
import time
import logging

# Configure logging
logger = logging.getLogger(__name__)

class HealthMonitor:
    """Monitor for video connection health."""
    
    def __init__(self):
        """Initialize the health monitor."""
        self.last_successful_read = 0
        self.consecutive_failures = 0
        self.max_reconnect_attempts = 5
        self.connection_healthy = True
        
    def update_on_success(self):
        """Update health metrics after successful frame read."""
        self.consecutive_failures = 0
        self.connection_healthy = True
        self.last_successful_read = time.time()
        
    def update_on_failure(self):
        """Update health metrics after failed frame read."""
        self.consecutive_failures += 1
        self.connection_healthy = False
        
    def check_health(self, is_camera=False, is_rtsp=False, is_file=False):
        """Check the health status of the video connection.
        
        Args:
            is_camera: Whether the source is a camera
            is_rtsp: Whether the source is an RTSP stream
            is_file: Whether the source is a file
            
        Returns:
            Dict with health status information
        """
        # Calculate time since last successful read
        current_time = time.time()
        time_since_last_read = current_time - self.last_successful_read if self.last_successful_read > 0 else 0
        
        # For video files, be much more lenient - they can have long gaps between frames
        # or pauses, and don't need health monitoring like live streams
        if is_file:
            # For files, only consider unhealthy if we have multiple consecutive failures
            # Don't use timeout-based health checks for files
            connection_healthy = self.consecutive_failures < 3
        else:
            # Consider connection unhealthy if no successful read in appropriate timeframe
            if is_rtsp:
                timeout_threshold = 30.0  # RTSP streams get 30 seconds
            elif is_camera:
                timeout_threshold = 10.0  # Cameras get 10 seconds
            else:
                timeout_threshold = 5.0   # Default fallback
            
            connection_timeout = time_since_last_read > timeout_threshold and self.last_successful_read > 0
            connection_healthy = not connection_timeout and self.consecutive_failures < 5
        
        # Only update health status if it's actually changing and not for files
        if not is_file and connection_healthy != self.connection_healthy:
            self.connection_healthy = connection_healthy
            if not connection_healthy:
                logger.warning(f"Video connection timeout: {time_since_last_read:.1f}s since last successful read")
        elif is_file:
            # For files, set health based on failure count only
            self.connection_healthy = connection_healthy
        
        # Get source type
        source_type = "unknown"
        if is_camera:
            source_type = "camera"
        elif is_rtsp:
            source_type = "rtsp"
        elif is_file:
            source_type = "file"
        
        return {
            "healthy": self.connection_healthy,
            "source_type": source_type,
            "consecutive_failures": self.consecutive_failures,
            "last_successful_read": self.last_successful_read,
            "time_since_last_read": time_since_last_read
        }
        
    def should_reconnect(self):
        """Check if a reconnection attempt should be made.
        
        Returns:
            bool: True if reconnection should be attempted
        """
        # Only reconnect if we have failures and connection is actually unhealthy
        # Be more conservative about reconnecting
        return not self.connection_healthy and self.consecutive_failures >= 3
        
    def exceeded_max_attempts(self):
        """Check if maximum reconnection attempts have been exceeded.
        
        Returns:
            bool: True if max reconnection attempts exceeded
        """
        return self.consecutive_failures >= self.max_reconnect_attempts

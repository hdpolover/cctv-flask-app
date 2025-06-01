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
        # Consider connection unhealthy if no successful read in last 5 seconds
        time_since_last_read = time.time() - self.last_successful_read
        timeout_threshold = 5.0  # 5 seconds
        
        connection_timeout = time_since_last_read > timeout_threshold and self.last_successful_read > 0
        
        # Update health status
        if connection_timeout and self.connection_healthy:
            self.connection_healthy = False
            logger.warning(f"Video connection timeout: {time_since_last_read:.1f}s since last successful read")
        
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
        return not self.connection_healthy and self.consecutive_failures > 0
        
    def exceeded_max_attempts(self):
        """Check if maximum reconnection attempts have been exceeded.
        
        Returns:
            bool: True if max reconnection attempts exceeded
        """
        return self.consecutive_failures >= self.max_reconnect_attempts

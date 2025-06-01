"""
UI utilities for video frames, including timestamp and button overlays.
"""
import cv2
import time
import logging
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class UIUtils:
    """Utilities for adding UI elements to video frames."""
    
    @staticmethod
    def add_timestamp(frame):
        """Add a timestamp to the frame.
        
        This method adds a real-time updating date and time display at the top-right
        corner of each video frame and also adds a refresh button to the top-left.
        
        Args:
            frame: Input video frame
            
        Returns:
            Frame with timestamp and refresh button added
        """
        if frame is None:
            return None
        
        try:
            # Refresh button removed per request
            # frame = UIUtils.add_refresh_button(frame)
            
            # Get current date and time
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            # Get text size to calculate position
            h, w = frame.shape[:2]
            text_size = cv2.getTextSize(current_time, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Calculate right-aligned position (with padding)
            text_x = w - text_size[0] - 10
            text_y = 20
            bg_width = text_size[0] + 20
            
            # Create a semi-transparent background for the timestamp in top-right corner
            timestamp_bg = frame[0:30, text_x-10:text_x+bg_width].copy()
            cv2.rectangle(timestamp_bg, (0, 0), (bg_width, 30), (0, 0, 0), -1)
            cv2.addWeighted(timestamp_bg, 0.7, frame[0:30, text_x-10:text_x+bg_width], 0.3, 0, frame[0:30, text_x-10:text_x+bg_width])
            
            # Add timestamp text with shadow for better visibility in top-right corner
            cv2.putText(frame, current_time, (text_x+1, text_y+1), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Shadow
            cv2.putText(frame, current_time, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Text
                        
            return frame
            
        except Exception as e:
            logger.exception(f"Error adding timestamp: {e}")
            return frame
    
    @staticmethod
    def add_refresh_button(frame):
        """Add a refresh button to the frame.
        
        This method adds a refresh button icon to the top-left corner
        of each video frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            Frame with refresh button added
        """
        if frame is None:
            return None
            
        try:
            h, w = frame.shape[:2]
            
            # Define refresh button dimensions
            button_size = 30
            margin = 10
            
            # Create a semi-transparent background for the refresh button in top-left corner
            refresh_bg = frame[margin:margin+button_size, margin:margin+button_size].copy()
            cv2.rectangle(refresh_bg, (0, 0), (button_size, button_size), (0, 0, 0), -1)
            cv2.addWeighted(refresh_bg, 0.7, frame[margin:margin+button_size, margin:margin+button_size], 0.3, 0, 
                           frame[margin:margin+button_size, margin:margin+button_size])
            
            # Draw refresh icon (circular arrow)
            center_x = margin + button_size // 2
            center_y = margin + button_size // 2
            radius = button_size // 3
            
            # Draw circle
            cv2.circle(frame, (center_x, center_y), radius, (0, 255, 255), 2)
            
            # Draw arrow head
            arrow_size = radius // 2
            cv2.arrowedLine(frame, 
                          (center_x, center_y - radius - arrow_size), 
                          (center_x, center_y - radius), 
                          (0, 255, 255), 2, tipLength=0.5)
            
            # Draw circular part of the arrow (270 degrees)
            cv2.ellipse(frame, (center_x, center_y), (radius, radius), 
                      90, 0, 270, (0, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            logger.exception(f"Error adding refresh button: {e}")
            return frame
            
    @staticmethod
    def create_error_frame(resolution):
        """Create an error frame to display when video source is unavailable.
        
        Args:
            resolution: Resolution as (width, height) tuple
            consecutive_failures: Number of consecutive connection failures
            
        Returns:
            Frame with error message
        """
        # Create a black frame
        frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        
        # Add error message
        cv2.putText(frame, "Video Source Unavailable", (int(resolution[0]/2) - 150, int(resolution[1]/2) - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add reconnection message
        cv2.putText(frame, "Reconnecting...", 
                   (int(resolution[0]/2) - 180, int(resolution[1]/2) + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)
        
        return frame

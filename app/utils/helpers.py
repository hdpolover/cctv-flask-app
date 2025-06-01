"""
Helper utility functions for the CCTV application
"""
import cv2
import numpy as np
import os
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

def parse_resolution(resolution_str):
    """Parse resolution string into width and height tuple.
    
    Args:
        resolution_str: Resolution string in format "width,height"
        
    Returns:
        Tuple of (width, height) or None if invalid format
    """
    try:
        if isinstance(resolution_str, str) and ',' in resolution_str:
            width, height = map(int, resolution_str.split(','))
            return (width, height)
    except (ValueError, TypeError) as e:
        logger.error(f"Error parsing resolution: {e}")
    
    return None

def save_snapshot(frame, directory="snapshots"):
    """Save a snapshot of the current frame.
    
    Args:
        frame: OpenCV frame to save
        directory: Directory to save snapshots
        
    Returns:
        Path to saved snapshot file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        filepath = os.path.join(directory, filename)
        
        # Save image
        cv2.imwrite(filepath, frame)
        logger.info(f"Snapshot saved: {filepath}")
        
        return filepath
    except Exception as e:
        logger.exception(f"Error saving snapshot: {e}")
        return None

def draw_counter_info(frame, entries, exits, people_in_room):
    """Draw people counter information on frame.
    
    Args:
        frame: OpenCV frame to draw on
        entries: Number of entries
        exits: Number of exits
        people_in_room: Current people count
        
    Returns:
        Frame with text overlay
    """
    # Draw background boxes for better readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (200, 100), (0, 0, 0), -1)
    alpha = 0.6  # Transparency factor
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Draw counter info
    cv2.putText(frame, f"Entries: {entries}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Exits: {exits}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"People in room: {people_in_room}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def draw_door_overlay(frame, door_area, inside_direction):
    """Draw door area and directional indicators on frame.
    
    Args:
        frame: OpenCV frame to draw on
        door_area: Tuple of (x1, y1, x2, y2) defining door area
        inside_direction: String indicating which side is "inside" ("left" or "right")
        
    Returns:
        Frame with door overlay
    """
    if door_area:
        x1, y1, x2, y2 = door_area
        
        # Draw door rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw door center line
        door_center_x = int((x1 + x2) / 2)
        cv2.line(frame, (door_center_x, y1), (door_center_x, y2), (255, 0, 0), 2)
          # Draw inside/outside labels
        if inside_direction == "right":
            cv2.putText(frame, "Luar", (x1 - 80, y1 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Dalam", (x2 + 10, y1 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
              # Draw arrows indicating direction
            cv2.arrowedLine(frame, (x1 - 20, y1 + 40), (x1 + 20, y1 + 40), 
                        (0, 255, 0), 2, tipLength=0.3)
            cv2.arrowedLine(frame, (x2 + 20, y1 + 60), (x2 - 20, y1 + 60), 
                        (0, 0, 255), 2, tipLength=0.3)
        else:
            cv2.putText(frame, "Dalam", (x1 - 80, y1 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Luar", (x2 + 10, y1 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw arrows indicating direction
            cv2.arrowedLine(frame, (x1 + 20, y1 + 40), (x1 - 20, y1 + 40), 
                        (0, 255, 0), 2, tipLength=0.3)
            cv2.arrowedLine(frame, (x2 - 20, y1 + 60), (x2 + 20, y1 + 60), 
                        (0, 0, 255), 2, tipLength=0.3)
    
    return frame
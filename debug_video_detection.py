#!/usr/bin/env python3
"""Debug script to test people detection and door crossing"""

import sys
import os
sys.path.append('.')

# Force logging to INFO level to see detection messages
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')

from app.models.detection_model import DetectionModel
import cv2
import numpy as np
import time

def test_with_current_video():
    """Test detection with the current demo video"""
    print("=== Testing Detection with Demo Video ===")
    
    # Load the demo video
    video_path = 'app/static/videos/demo.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {video_path}")
        return
        
    # Create detection model
    print("\nInitializing detection model...")
    model = DetectionModel()
    
    # Set door area (from current configuration)
    print("\nSetting door area: (272, 144, 368, 336)")
    model.set_door_area(272, 144, 368, 336)
    model.set_inside_direction("down")
    
    print(f"Door defined: {model.door_defined}")
    print(f"Door area: {model.door_area}")
    print(f"Inside direction: {model.inside_direction}")
    
    frame_count = 0
    detection_count = 0
    movement_detected = 0
    
    print("\n=== Starting Video Analysis ===")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process every 5th frame to speed up testing
        if frame_count % 5 != 0:
            continue
            
        # Resize to standard resolution
        frame = cv2.resize(frame, (640, 480))
        
        try:
            # Detect people
            people_boxes, movement = model.detect_people(frame)
            
            if len(people_boxes) > 0:
                detection_count += 1
                print(f"Frame {frame_count}: Detected {len(people_boxes)} people")
                
                # Print bounding box centers
                for i, box in enumerate(people_boxes):
                    center = model.get_box_center(box)
                    print(f"  Person {i+1}: Center at {center}, Box: {box}")
            
            # Check for movement
            if movement and any(movement.values()):
                movement_detected += 1
                print(f"Frame {frame_count}: Movement detected: {movement}")
                
            # Print current counters every 30 frames
            if frame_count % 30 == 0:
                entries, exits = model.get_entry_exit_count()
                print(f"Frame {frame_count}: Current counts - Entries: {entries}, Exits: {exits}")
                print(f"  L2R: {model.left_to_right}, R2L: {model.right_to_left}")
                print(f"  T2B: {model.top_to_bottom}, B2T: {model.bottom_to_top}")
                
        except Exception as e:
            print(f"Frame {frame_count}: Detection error: {e}")
            
        # Limit testing to first 150 frames
        if frame_count >= 150:
            break
    
    cap.release()
    
    # Final summary
    print(f"\n=== Analysis Complete ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames with people detected: {detection_count}")
    print(f"Frames with movement: {movement_detected}")
    
    entries, exits = model.get_entry_exit_count()
    print(f"\nFinal counts:")
    print(f"  Entries: {entries}")
    print(f"  Exits: {exits}")
    print(f"  People in room: {max(0, entries - exits)}")
    print(f"\nDirect counters:")
    print(f"  Left to Right: {model.left_to_right}")
    print(f"  Right to Left: {model.right_to_left}")
    print(f"  Top to Bottom: {model.top_to_bottom}")
    print(f"  Bottom to Top: {model.bottom_to_top}")

if __name__ == "__main__":
    test_with_current_video()

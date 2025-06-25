#!/usr/bin/env python3
"""Test detection with optimized door area"""

import sys
import os
sys.path.append('.')

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')

from app.models.detection_model import DetectionModel
import cv2
import numpy as np

def test_with_optimized_door_area():
    """Test detection with a more reasonable door area"""
    print("=== Testing with Optimized Door Area ===")
    
    # Load the demo video
    video_path = 'app/static/videos/demo.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {video_path}")
        return
        
    # Create detection model with optimized door area
    # Based on the analysis, let's create a more focused door area that covers the main movement path
    # We'll place it around the center of movement (399, 274) but make it smaller
    
    door_x1, door_y1 = 320, 180  # Start point
    door_x2, door_y2 = 480, 360  # End point - covers width of 160, height of 180
    
    print(f"\nTesting door area: ({door_x1}, {door_y1}, {door_x2}, {door_y2})")
    print(f"Door size: {door_x2 - door_x1} x {door_y2 - door_y1}")
    
    model = DetectionModel()
    model.set_door_area(door_x1, door_y1, door_x2, door_y2)
    model.set_inside_direction("down")
    
    print(f"Door defined: {model.door_defined}")
    print(f"Inside direction: {model.inside_direction}")
    
    frame_count = 0
    detection_count = 0
    movement_detected = 0
    
    print("\n=== Starting Detection Test ===")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process every 5th frame 
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
                
                # Print centers and check if they're in door area
                for i, box in enumerate(people_boxes):
                    center = model.get_box_center(box)
                    in_door = model.is_in_door_area(center)
                    print(f"  Person {i+1}: Center at {center} {'(IN DOOR AREA)' if in_door else '(outside door)'}")
            
            # Check for movement
            if movement and any(movement.values()):
                movement_detected += 1
                print(f"Frame {frame_count}: ✅ MOVEMENT DETECTED: {movement}")
                
            # Print current counters every 30 frames
            if frame_count % 30 == 0:
                entries, exits = model.get_entry_exit_count()
                print(f"Frame {frame_count}: Counts - Entries: {entries}, Exits: {exits}, In room: {max(0, entries - exits)}")
                
        except Exception as e:
            print(f"Frame {frame_count}: Detection error: {e}")
            
        # Test more frames to catch movement
        if frame_count >= 200:
            break
    
    cap.release()
    
    # Final summary
    print(f"\n=== Test Results ===")
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
    
    if movement_detected > 0:
        print(f"\n✅ SUCCESS: Movement detection is working!")
        print(f"   Recommended door area: ({door_x1}, {door_y1}, {door_x2}, {door_y2})")
    else:
        print(f"\n❌ No movement detected. The demo video may not contain crossing movements.")
        print(f"   Try with a live camera feed or different video for better results.")

if __name__ == "__main__":
    test_with_optimized_door_area()

#!/usr/bin/env python3
"""Test with corrected inside direction"""

import sys
import os
sys.path.append('.')

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')

from app.models.detection_model import DetectionModel
import cv2

def test_with_correct_inside_direction():
    """Test with inside direction matching the movement type"""
    print("=== Testing with Correct Inside Direction ===")
    
    # Load the demo video
    video_path = 'app/static/videos/demo.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {video_path}")
        return
        
    # Create detection model with the working door area
    door_x1, door_y1 = 320, 180  
    door_x2, door_y2 = 480, 360
    
    model = DetectionModel()
    model.set_door_area(door_x1, door_y1, door_x2, door_y2)
    
    # Since we have horizontal movements (left-to-right, right-to-left),
    # we need to set the inside direction appropriately
    # If "left" means inside, then left_to_right = entry, right_to_left = exit
    # If "right" means inside, then right_to_left = entry, left_to_right = exit
    
    # Let's test both directions
    for inside_dir in ["left", "right"]:
        print(f"\n--- Testing with inside_direction = '{inside_dir}' ---")
        model.set_inside_direction(inside_dir)
        model.reset_counters()  # Reset for clean test
        
        frame_count = 0
        movement_detected = 0
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to start
        
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
                
                # Check for movement
                if movement and any(movement.values()):
                    movement_detected += 1
                    
            except Exception as e:
                print(f"Frame {frame_count}: Detection error: {e}")
                
            # Limit frames for faster testing
            if frame_count >= 200:
                break
        
        # Get results
        entries, exits = model.get_entry_exit_count()
        total_people_in_room = max(0, entries - exits)
        
        print(f"Results for inside_direction = '{inside_dir}':")
        print(f"  Direct counters: L->R: {model.left_to_right}, R->L: {model.right_to_left}")
        print(f"  Calculated: Entries: {entries}, Exits: {exits}")
        print(f"  People in room: {total_people_in_room}")
        print(f"  Movement frames detected: {movement_detected}")
        
        if entries > 0 or exits > 0:
            print(f"  ✅ SUCCESS: Entry/exit counting is working!")
            print(f"  📋 RECOMMENDATION: Use inside_direction = '{inside_dir}'")
            break
        else:
            print(f"  ❌ No entries/exits counted with this direction")
    
    cap.release()

if __name__ == "__main__":
    test_with_correct_inside_direction()

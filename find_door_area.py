#!/usr/bin/env python3
"""Find optimal door area based on people movement in demo video"""

import sys
import os
sys.path.append('.')

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')

from app.models.detection_model import DetectionModel
import cv2
import numpy as np

def find_optimal_door_area():
    """Analyze people movement to suggest optimal door area"""
    print("=== Finding Optimal Door Area ===")
    
    # Load the demo video
    video_path = 'app/static/videos/demo.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {video_path}")
        return
        
    # Create detection model
    model = DetectionModel()
    
    all_positions = []
    frame_count = 0
    
    print("\nAnalyzing people positions in video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process every 10th frame for speed
        if frame_count % 10 != 0:
            continue
            
        # Resize to standard resolution
        frame = cv2.resize(frame, (640, 480))
        
        try:
            # Detect people
            people_boxes, _ = model.detect_people(frame)
            
            for box in people_boxes:
                center = model.get_box_center(box)
                all_positions.append(center)
                print(f"Frame {frame_count}: Person at {center}")
                
        except Exception as e:
            print(f"Frame {frame_count}: Detection error: {e}")
            
        # Limit analysis to avoid too much processing
        if frame_count >= 300:
            break
    
    cap.release()
    
    if not all_positions:
        print("No people detected in video!")
        return
        
    # Analyze positions
    x_positions = [pos[0] for pos in all_positions]
    y_positions = [pos[1] for pos in all_positions]
    
    min_x, max_x = min(x_positions), max(x_positions)
    min_y, max_y = min(y_positions), max(y_positions)
    
    print(f"\n=== Position Analysis ===")
    print(f"Total positions detected: {len(all_positions)}")
    print(f"X range: {min_x} to {max_x}")
    print(f"Y range: {min_y} to {max_y}")
    
    # Calculate center of movement
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    
    print(f"Movement center: ({center_x}, {center_y})")
    
    # Suggest door area
    # Make door area cover the movement range plus some buffer
    door_width = max(96, (max_x - min_x) + 40)  # Minimum 96px width
    door_height = max(120, (max_y - min_y) + 40)  # Minimum 120px height
    
    # Center the door on the movement
    door_x1 = max(0, center_x - door_width // 2)
    door_y1 = max(0, center_y - door_height // 2)
    door_x2 = min(640, door_x1 + door_width)
    door_y2 = min(480, door_y1 + door_height)
    
    # Adjust if we hit boundaries
    if door_x2 == 640:
        door_x1 = 640 - door_width
    if door_y2 == 480:
        door_y1 = 480 - door_height
        
    print(f"\n=== Suggested Door Area ===")
    print(f"Current door area: (272, 144, 368, 336)")
    print(f"Suggested door area: ({door_x1}, {door_y1}, {door_x2}, {door_y2})")
    print(f"Suggested door size: {door_x2 - door_x1} x {door_y2 - door_y1}")
    
    # Test the suggested door area
    print(f"\n=== Testing Suggested Door Area ===")
    
    # Create a new model with suggested door area
    test_model = DetectionModel()
    test_model.set_door_area(door_x1, door_y1, door_x2, door_y2)
    test_model.set_inside_direction("down")
    
    # Test with a few sample positions
    print(f"Testing crossing detection with sample positions...")
    
    # Simulate movement across the suggested door area
    door_center_x = (door_x1 + door_x2) // 2
    door_center_y = (door_y1 + door_y2) // 2
    
    # Test left to right movement
    prev_pos = (door_center_x - 30, door_center_y)
    curr_pos = (door_center_x + 30, door_center_y)
    crossed, direction = test_model.is_crossing_door(prev_pos, curr_pos)
    print(f"  Left to right test: {prev_pos} -> {curr_pos}: crossed={crossed}, direction={direction}")
    
    # Test top to bottom movement  
    prev_pos = (door_center_x, door_center_y - 30)
    curr_pos = (door_center_x, door_center_y + 30)
    crossed, direction = test_model.is_crossing_door(prev_pos, curr_pos)
    print(f"  Top to bottom test: {prev_pos} -> {curr_pos}: crossed={crossed}, direction={direction}")
    
    print(f"\n=== Action Required ===")
    print(f"To fix the door counting issue:")
    print(f"1. Go to the Camera Settings page")
    print(f"2. Set door area coordinates to: ({door_x1}, {door_y1}, {door_x2}, {door_y2})")
    print(f"3. Or use the visual door area selector to draw a box covering the people movement area")

if __name__ == "__main__":
    find_optimal_door_area()

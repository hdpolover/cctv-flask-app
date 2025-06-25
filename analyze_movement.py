#!/usr/bin/env python3
"""Analyze movement patterns in detail"""

import sys
import os
sys.path.append('.')

import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(name)s - %(message)s')

from app.models.detection_model import DetectionModel
import cv2
import numpy as np

def analyze_movement_patterns():
    """Analyze detailed movement patterns"""
    print("=== Analyzing Movement Patterns ===")
    
    # Load the demo video
    video_path = 'app/static/videos/demo.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {video_path}")
        return
        
    # Create detection model with door area where people are detected
    door_x1, door_y1 = 320, 180  
    door_x2, door_y2 = 480, 360
    
    model = DetectionModel()
    model.set_door_area(door_x1, door_y1, door_x2, door_y2)
    model.set_inside_direction("down")
    
    # Track positions over time
    position_history = []
    frame_count = 0
    
    print(f"Door area: ({door_x1}, {door_y1}, {door_x2}, {door_y2})")
    print(f"Door center line (vertical): x = {(door_x1 + door_x2) // 2}")
    print(f"Door center line (horizontal): y = {(door_y1 + door_y2) // 2}")
    
    print("\n=== Tracking Position Changes ===")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process every 3rd frame for better tracking
        if frame_count % 3 != 0:
            continue
            
        # Resize to standard resolution
        frame = cv2.resize(frame, (640, 480))
        
        try:
            # Detect people
            people_boxes, movement = model.detect_people(frame)
            
            if len(people_boxes) > 0:
                # Store positions
                frame_positions = []
                for i, box in enumerate(people_boxes):
                    center = model.get_box_center(box)
                    in_door = model.is_in_door_area(center)
                    if in_door:
                        frame_positions.append(center)
                        
                if frame_positions:
                    position_history.append({
                        'frame': frame_count,
                        'positions': frame_positions
                    })
                    
                    # Print position changes
                    if len(position_history) > 1:
                        prev_positions = position_history[-2]['positions']
                        curr_positions = position_history[-1]['positions']
                        
                        print(f"\nFrame {frame_count}:")
                        for i, pos in enumerate(curr_positions):
                            if i < len(prev_positions):
                                prev_pos = prev_positions[i]
                                dx = pos[0] - prev_pos[0]
                                dy = pos[1] - prev_pos[1]
                                distance = np.sqrt(dx*dx + dy*dy)
                                
                                print(f"  Person {i+1}: {prev_pos} -> {pos} (dx={dx:.1f}, dy={dy:.1f}, dist={distance:.1f})")
                                
                                # Check if this would trigger a crossing
                                door_center_x = (door_x1 + door_x2) // 2
                                door_center_y = (door_y1 + door_y2) // 2
                                
                                # Check horizontal crossing
                                if prev_pos[0] < door_center_x and pos[0] > door_center_x:
                                    print(f"    >>> LEFT-TO-RIGHT CROSSING! ({prev_pos[0]} -> {pos[0]} across {door_center_x})")
                                elif prev_pos[0] > door_center_x and pos[0] < door_center_x:
                                    print(f"    >>> RIGHT-TO-LEFT CROSSING! ({prev_pos[0]} -> {pos[0]} across {door_center_x})")
                                    
                                # Check vertical crossing
                                if prev_pos[1] < door_center_y and pos[1] > door_center_y:
                                    print(f"    >>> TOP-TO-BOTTOM CROSSING! ({prev_pos[1]} -> {pos[1]} across {door_center_y})")
                                elif prev_pos[1] > door_center_y and pos[1] < door_center_y:
                                    print(f"    >>> BOTTOM-TO-TOP CROSSING! ({prev_pos[1]} -> {pos[1]} across {door_center_y})")
            
            # Check movement result
            if movement and any(movement.values()):
                print(f"Frame {frame_count}: ✅ MOVEMENT DETECTED: {movement}")
                
        except Exception as e:
            print(f"Frame {frame_count}: Detection error: {e}")
            
        # Limit to reasonable number of frames
        if frame_count >= 300:
            break
    
    cap.release()
    
    # Analyze position history
    print(f"\n=== Movement Analysis ===")
    print(f"Total frames with people in door area: {len(position_history)}")
    
    if len(position_history) > 1:
        max_movement = 0
        total_movement = 0
        movement_count = 0
        
        for i in range(1, len(position_history)):
            prev_frame = position_history[i-1]
            curr_frame = position_history[i]
            
            for j, pos in enumerate(curr_frame['positions']):
                if j < len(prev_frame['positions']):
                    prev_pos = prev_frame['positions'][j]
                    distance = np.sqrt((pos[0] - prev_pos[0])**2 + (pos[1] - prev_pos[1])**2)
                    max_movement = max(max_movement, distance)
                    total_movement += distance
                    movement_count += 1
        
        avg_movement = total_movement / movement_count if movement_count > 0 else 0
        print(f"Maximum movement between frames: {max_movement:.1f} pixels")
        print(f"Average movement between frames: {avg_movement:.1f} pixels")
        
        if max_movement < 5:
            print("⚠️  Very little movement detected - people may be mostly stationary")
            print("   This demo video might not show people crossing the door area")
            print("   The detection and tracking logic is working, but no significant crossings occur")
        elif max_movement < 20:
            print("⚠️  Small movements detected - but may not be crossing the door lines")
            print("   Consider adjusting the door area or testing with different content")
        else:
            print("✅ Significant movement detected - crossings should be registered")
    
    # Final summary
    entries, exits = model.get_entry_exit_count()
    print(f"\nFinal Results:")
    print(f"  Entries: {entries}")
    print(f"  Exits: {exits}")
    print(f"  Movement counters: L->R: {model.left_to_right}, R->L: {model.right_to_left}")
    print(f"                     T->B: {model.top_to_bottom}, B->T: {model.bottom_to_top}")

if __name__ == "__main__":
    analyze_movement_patterns()

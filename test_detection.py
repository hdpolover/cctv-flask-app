#!/usr/bin/env python3
"""Test script for detection model door crossing logic"""

import sys
import os
sys.path.append('.')

from app.models.detection_model import DetectionModel
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

def test_detection_model():
    print("=== Testing Detection Model ===")
    
    # Create detection model
    model = DetectionModel()
    
    # Set door area
    print("\n1. Setting door area...")
    model.set_door_area(272, 144, 368, 336)
    print(f"Door area: {model.door_area}")
    print(f"Door defined: {model.door_defined}")
    print(f"Inside direction: {model.inside_direction}")
    
    # Test door crossing detection
    print("\n2. Testing door crossing detection...")
    prev_center = (250, 200)  # Left of door
    current_center = (390, 200)  # Right of door
    
    crossed, direction = model.is_crossing_door(prev_center, current_center)
    print(f"Test crossing: {prev_center} -> {current_center}")
    print(f"Result: crossed={crossed}, direction={direction}")
    
    # Test movement tracking - first call
    print("\n3. Testing movement tracking...")
    fake_boxes = [[245, 180, 265, 220]]  # Person on left side
    
    print("First call (establish previous position):")
    movement1 = model.track_movement(fake_boxes, 640)
    print(f"Movement result: {movement1}")
    print(f"Previous centers: {model.previous_centers}")
    print(f"Counters - L2R: {model.left_to_right}, R2L: {model.right_to_left}")
    
    # Test movement tracking - second call (crossing)
    fake_boxes = [[385, 180, 405, 220]]  # Person on right side
    
    print("\nSecond call (should detect crossing):")
    movement2 = model.track_movement(fake_boxes, 640)
    print(f"Movement result: {movement2}")
    print(f"Previous centers: {model.previous_centers}")
    print(f"Counters - L2R: {model.left_to_right}, R2L: {model.right_to_left}")
    
    # Test get_entry_exit_count
    print("\n4. Testing entry/exit count...")
    entries, exits = model.get_entry_exit_count()
    print(f"Entries: {entries}, Exits: {exits}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_detection_model()

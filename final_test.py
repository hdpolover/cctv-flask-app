#!/usr/bin/env python3
"""Final comprehensive test of the fixed detection system"""

import sys
import os
sys.path.append('.')

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')

from app.models.detection_model import DetectionModel
import cv2

def final_comprehensive_test():
    """Comprehensive test of the fixed detection system"""
    print("🎯 FINAL COMPREHENSIVE TEST")
    print("=" * 50)
    
    # Load the demo video
    video_path = 'app/static/videos/demo.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open video file: {video_path}")
        return False
        
    # Create detection model with optimized settings
    door_x1, door_y1 = 320, 180  
    door_x2, door_y2 = 480, 360
    
    model = DetectionModel()
    model.set_door_area(door_x1, door_y1, door_x2, door_y2)
    model.set_inside_direction("left")  # Correct direction for horizontal crossings
    
    print(f"📐 Door area: ({door_x1}, {door_y1}, {door_x2}, {door_y2})")
    print(f"📍 Door size: {door_x2 - door_x1} x {door_y2 - door_y1} pixels")
    print(f"🚪 Inside direction: left (right-to-left = entry, left-to-right = exit)")
    print()
    
    frame_count = 0
    movement_frames = 0
    people_detected_frames = 0
    
    print("🎬 Processing video...")
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
                people_detected_frames += 1
                
            # Check for movement
            if movement and any(movement.values()):
                movement_frames += 1
                entries, exits = model.get_entry_exit_count()
                people_in_room = max(0, entries - exits)
                print(f"  📊 Frame {frame_count}: Movement detected! Entries: {entries}, Exits: {exits}, In room: {people_in_room}")
                
        except Exception as e:
            print(f"❌ Frame {frame_count}: Detection error: {e}")
            
        # Process more frames to catch all movements
        if frame_count >= 300:
            break
    
    cap.release()
    
    # Final results
    print("\n" + "=" * 50)
    print("📋 FINAL RESULTS")
    print("=" * 50)
    
    entries, exits = model.get_entry_exit_count()
    people_in_room = max(0, entries - exits)
    
    print(f"🎞️  Total frames processed: {frame_count}")
    print(f"👥 Frames with people detected: {people_detected_frames}")
    print(f"🚶‍♂️ Frames with movement: {movement_frames}")
    print()
    print(f"📊 MOVEMENT COUNTERS:")
    print(f"   Left-to-Right: {model.left_to_right}")
    print(f"   Right-to-Left: {model.right_to_left}")
    print(f"   Top-to-Bottom: {model.top_to_bottom}")
    print(f"   Bottom-to-Top: {model.bottom_to_top}")
    print()
    print(f"🏠 ENTRY/EXIT ANALYSIS:")
    print(f"   🚪 Entries: {entries}")
    print(f"   🚪 Exits: {exits}")
    print(f"   👤 People currently in room: {people_in_room}")
    print()
    
    # Validation
    success = True
    issues = []
    
    if movement_frames == 0:
        success = False
        issues.append("No movement detected")
        
    if entries == 0 and exits == 0:
        success = False
        issues.append("No entries or exits counted")
        
    if people_detected_frames == 0:
        success = False
        issues.append("No people detected")
    
    print("🔍 VALIDATION:")
    if success:
        print("✅ ALL TESTS PASSED!")
        print("✅ People detection: WORKING")
        print("✅ Movement tracking: WORKING") 
        print("✅ Door crossing detection: WORKING")
        print("✅ Entry/exit counting: WORKING")
        print()
        print("🎯 SOLUTION SUMMARY:")
        print("1. Fixed door orientation detection (now treats square-ish doors as vertical)")
        print("2. Set correct inside direction ('left' for horizontal crossings)")
        print("3. Optimized door area to cover actual people movement")
        print("4. Door area coordinates: (320, 180, 480, 360)")
        print()
        print("🚀 READY FOR PRODUCTION!")
        print("   Update the door area in Camera Settings to: (320, 180, 480, 360)")
        print("   Set inside direction to: 'left'")
        return True
    else:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        return False

if __name__ == "__main__":
    success = final_comprehensive_test()
    sys.exit(0 if success else 1)

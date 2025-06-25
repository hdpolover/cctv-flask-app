#!/usr/bin/env python3
"""
Comprehensive test of the door configuration helper with demo video validation
"""

import cv2
import numpy as np
from door_config_helper import set_door_config, get_current_door_config, reset_counts, get_detection_model

def test_with_demo_video():
    """Test the door configuration with the demo video to validate counting"""
    print("🎬 Testing door configuration with demo.mp4...")
    
    # Apply optimized configuration
    print("\n1. Applying optimized configuration...")
    result = set_door_config(
        door_area=(320, 180, 480, 360),
        inside_direction="left"
    )
    
    if not result.get("success"):
        print(f"❌ Failed to apply configuration: {result.get('error')}")
        return
    
    print(f"✅ Configuration applied: {result}")
    
    # Get the configured model
    model = get_detection_model()
    
    # Test with demo video
    video_path = "app/static/videos/demo.mp4"
    print(f"\n2. Processing {video_path}...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    frame_count = 0
    processed_frames = 0
    
    # Reset counters before processing
    model.reset_counters()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process every 5th frame to speed up testing
        if frame_count % 5 == 0:
            try:
                # Process frame for detections
                detections = model.process_frame(frame)
                processed_frames += 1
                
                # Show progress every 50 processed frames
                if processed_frames % 50 == 0:
                    entries, exits = model.get_entry_exit_count()
                    print(f"   Frame {frame_count}: Entries={entries}, Exits={exits}")
                    
            except Exception as e:
                print(f"⚠️  Error processing frame {frame_count}: {e}")
                continue
    
    cap.release()
    
    # Get final results
    print(f"\n3. Final results after processing {processed_frames} frames:")
    config = get_current_door_config()
    
    print(f"   Door area: {config['door_area']}")
    print(f"   Inside direction: {config['inside_direction']}")
    print(f"   Total entries: {config['entry_count']}")
    print(f"   Total exits: {config['exit_count']}")
    print(f"   People in room: {config['entry_count'] - config['exit_count']}")
    
    # Validate expected results
    expected_entries = 3
    expected_exits = 3
    
    if config['entry_count'] == expected_entries and config['exit_count'] == expected_exits:
        print(f"\n✅ SUCCESS! Detected expected counts: {expected_entries} entries, {expected_exits} exits")
        return True
    else:
        print(f"\n⚠️  UNEXPECTED RESULTS: Expected {expected_entries} entries and {expected_exits} exits")
        return False

def test_reset_functionality():
    """Test the reset functionality"""
    print("\n4. Testing reset functionality...")
    
    # Get counts before reset
    config_before = get_current_door_config()
    print(f"   Before reset: Entries={config_before['entry_count']}, Exits={config_before['exit_count']}")
    
    # Reset counters
    reset_counts()
    
    # Get counts after reset
    config_after = get_current_door_config()
    print(f"   After reset: Entries={config_after['entry_count']}, Exits={config_after['exit_count']}")
    
    if config_after['entry_count'] == 0 and config_after['exit_count'] == 0:
        print("   ✅ Reset successful!")
        return True
    else:
        print("   ❌ Reset failed!")
        return False

if __name__ == "__main__":
    print("🔧 COMPREHENSIVE DOOR CONFIGURATION TEST")
    print("=" * 50)
    
    success = True
    
    try:
        # Test configuration and video processing
        if not test_with_demo_video():
            success = False
        
        # Test reset functionality
        if not test_reset_functionality():
            success = False
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED! The door configuration helper is working correctly.")
        print("\nThe system is ready for production with the optimized door settings:")
        print("   Door area: (320, 180, 480, 360)")
        print("   Inside direction: left")
        print("   Expected accuracy: 3 entries, 3 exits for demo.mp4")
    else:
        print("❌ SOME TESTS FAILED! Please check the configuration.")

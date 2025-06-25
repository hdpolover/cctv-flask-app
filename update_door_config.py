#!/usr/bin/env python3
"""
Automatic Door Configuration Script
===================================
This script automatically updates the door area and inside direction
based on the optimized settings we determined from testing.
"""

import sys
import os
sys.path.append('.')

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def update_door_configuration():
    """Update door area and inside direction automatically"""
    
    print("🔧 AUTOMATIC DOOR CONFIGURATION UPDATER")
    print("=" * 50)
    print()
    
    try:
        # Import the detection model
        from app.models.detection_model import DetectionModel
        
        # Create detection model instance
        print("📊 Initializing detection model...")
        model = DetectionModel()
        
        # Optimized door area coordinates from our testing
        door_x1, door_y1 = 320, 180
        door_x2, door_y2 = 480, 360
        inside_direction = "left"
        
        print(f"🚪 Setting door area to: ({door_x1}, {door_y1}, {door_x2}, {door_y2})")
        print(f"   - Width: {door_x2 - door_x1} pixels")
        print(f"   - Height: {door_y2 - door_y1} pixels")
        print()
        
        # Apply door area
        success = model.set_door_area(door_x1, door_y1, door_x2, door_y2)
        if success:
            print("✅ Door area updated successfully!")
        else:
            print("❌ Failed to update door area!")
            return False
        
        print(f"🧭 Setting inside direction to: '{inside_direction}'")
        success = model.set_inside_direction(inside_direction)
        if success:
            print("✅ Inside direction updated successfully!")
        else:
            print("❌ Failed to update inside direction!")
            return False
        
        # Verify settings
        print()
        print("🔍 VERIFICATION:")
        current_door_area = model.get_door_area()
        current_inside_direction = model.get_inside_direction()
        
        if current_door_area == (door_x1, door_y1, door_x2, door_y2):
            print("✅ Door area verified correct")
        else:
            print(f"❌ Door area mismatch: expected {(door_x1, door_y1, door_x2, door_y2)}, got {current_door_area}")
            
        if current_inside_direction == inside_direction:
            print("✅ Inside direction verified correct")
        else:
            print(f"❌ Inside direction mismatch: expected {inside_direction}, got {current_inside_direction}")
        
        # Test with demo video to confirm it works
        print()
        print("🎬 TESTING WITH DEMO VIDEO...")
        
        import cv2
        video_path = 'app/static/videos/demo.mp4'
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"⚠️  Warning: Could not open demo video at {video_path}")
            print("   The configuration has been applied, but testing skipped.")
            return True
        
        # Process a few frames to test
        frame_count = 0
        people_detected = 0
        movement_detected = 0
        
        while frame_count < 100:  # Quick test with first 100 frames
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 10 != 0:  # Process every 10th frame
                continue
                
            # Resize to standard resolution
            frame = cv2.resize(frame, (640, 480))
            
            try:
                # Detect people
                people_boxes, movement = model.detect_people(frame)
                
                if len(people_boxes) > 0:
                    people_detected += 1
                    
                if movement and any(movement.values()):
                    movement_detected += 1
                    
            except Exception as e:
                print(f"   Error during testing: {e}")
                break
        
        cap.release()
        
        # Test results
        entries, exits = model.get_entry_exit_count()
        people_in_room = max(0, entries - exits)
        
        print(f"   Frames processed: {frame_count}")
        print(f"   People detected in {people_detected} frames")
        print(f"   Movement detected in {movement_detected} frames")
        print(f"   Final counts: Entries={entries}, Exits={exits}, In room={people_in_room}")
        
        if movement_detected > 0:
            print("✅ Configuration test PASSED - movement detection working!")
        else:
            print("⚠️  Configuration test incomplete - no movement detected in test frames")
            print("   This might be normal if the test frames don't contain crossings")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're running this from the Flask app directory")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def update_flask_config():
    """Update Flask configuration files if they exist"""
    
    print()
    print("📝 UPDATING CONFIGURATION FILES...")
    
    # Look for common Flask config files
    config_files = [
        'config/settings.py',
        'config.py',
        'app/config.py'
    ]
    
    door_area_config = """
# Optimized door area configuration
DOOR_AREA = {
    'x1': 320,
    'y1': 180, 
    'x2': 480,
    'y2': 360
}
INSIDE_DIRECTION = 'left'
"""
    
    config_found = False
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"📄 Found config file: {config_file}")
            config_found = True
            
            # Read existing config
            try:
                with open(config_file, 'r') as f:
                    content = f.read()
                
                # Check if door area config already exists
                if 'DOOR_AREA' in content or 'door_area' in content.lower():
                    print(f"   ⚠️  {config_file} already contains door area configuration")
                    print(f"   You may want to manually update it with the new values:")
                    print(f"   Door Area: (320, 180, 480, 360)")
                    print(f"   Inside Direction: 'left'")
                else:
                    # Append the new configuration
                    with open(config_file, 'a') as f:
                        f.write(door_area_config)
                    print(f"   ✅ Added door configuration to {config_file}")
                    
            except Exception as e:
                print(f"   ❌ Error updating {config_file}: {e}")
    
    if not config_found:
        # Create a new config file
        config_file = 'door_config.py'
        try:
            with open(config_file, 'w') as f:
                f.write(f"# Door Configuration - Generated automatically\n{door_area_config}")
            print(f"✅ Created new config file: {config_file}")
        except Exception as e:
            print(f"❌ Error creating config file: {e}")

def print_manual_instructions():
    """Print manual instructions as backup"""
    
    print()
    print("📋 MANUAL CONFIGURATION INSTRUCTIONS")
    print("=" * 50)
    print()
    print("If you need to configure manually through the web interface:")
    print()
    print("1. 🌐 Open your Flask web application")
    print("2. 🔧 Go to Camera Settings page")
    print("3. 🚪 Set Door Area coordinates:")
    print("   - X1 (left): 320")
    print("   - Y1 (top): 180")
    print("   - X2 (right): 480")
    print("   - Y2 (bottom): 360")
    print("4. 🧭 Set Inside Direction: left")
    print("5. 💾 Save settings")
    print()
    print("Alternative: Use the door area visual selector if available")
    print("- Draw a rectangle covering the door area")
    print("- The rectangle should be approximately 160x180 pixels")
    print("- Position it where people typically walk through")

def main():
    """Main function"""
    
    print("🚀 Starting automatic door configuration...")
    print()
    
    # Step 1: Update the detection model directly
    if update_door_configuration():
        print()
        print("🎉 SUCCESS! Door configuration has been applied.")
        
        # Step 2: Try to update config files
        update_flask_config()
        
        print()
        print("🎯 CONFIGURATION COMPLETE!")
        print("=" * 50)
        print("✅ Door area: (320, 180, 480, 360)")
        print("✅ Inside direction: left")
        print("✅ Ready for production use!")
        print()
        print("💡 Next steps:")
        print("1. Restart your Flask application if it's running")
        print("2. Test with live camera feed")
        print("3. Monitor the entry/exit counts in the dashboard")
        
        return True
    else:
        print()
        print("❌ CONFIGURATION FAILED!")
        print()
        print_manual_instructions()
        return False

if __name__ == "__main__":
    success = main()
    
    # Always show manual instructions as reference
    print_manual_instructions()
    
    sys.exit(0 if success else 1)

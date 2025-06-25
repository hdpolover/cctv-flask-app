#!/usr/bin/env python3
"""
Door Configuration Helper
Provides easy programmatic setup for door area and inside direction configuration.
"""

import os
import sys

# Global detection model instance for consistency
_detection_model = None

def get_detection_model():
    """Get or create a shared detection model instance."""
    global _detection_model
    if _detection_model is None:
        from app.models.detection_model import DetectionModel
        _detection_model = DetectionModel()
    return _detection_model

def get_current_door_config():
    """Get current door configuration from detection model."""
    try:
        model = get_detection_model()
        entries, exits = model.get_entry_exit_count()
        return {
            "door_area": model.door_area,
            "inside_direction": model.inside_direction,
            "entry_count": entries,
            "exit_count": exits
        }
    except Exception as e:
        return {"error": str(e)}

def set_door_config(door_area, inside_direction):
    """Set door configuration in detection model."""
    try:
        model = get_detection_model()
        
        # Set door area
        if door_area:
            x1, y1, x2, y2 = door_area
            model.set_door_area(x1, y1, x2, y2)
            
        # Set inside direction  
        if inside_direction:
            model.inside_direction = inside_direction
            
        return {
            "success": True,
            "door_area": model.door_area,
            "inside_direction": model.inside_direction
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def apply_optimized_config():
    """Apply the optimized configuration for the demo video."""
    optimal_config = {
        "door_area": (320, 180, 480, 360),
        "inside_direction": "left"
    }
    
    print("🔧 Applying optimized door configuration...")
    print(f"   Door area: {optimal_config['door_area']}")
    print(f"   Inside direction: {optimal_config['inside_direction']}")
    
    result = set_door_config(
        door_area=optimal_config["door_area"],
        inside_direction=optimal_config["inside_direction"]
    )
    
    if result.get("success"):
        print("✅ Configuration applied successfully!")
        print_config_summary()
    else:
        print(f"❌ Failed to apply configuration: {result.get('error')}")

def reset_counts():
    """Reset entry and exit counts."""
    try:
        model = get_detection_model()
        model.reset_counters()
        print("✅ Entry/exit counts reset to 0")
    except Exception as e:
        print(f"❌ Failed to reset counts: {e}")

def print_config_summary():
    """Print current configuration summary."""
    config = get_current_door_config()
    
    if "error" in config:
        print(f"❌ Error getting configuration: {config['error']}")
        return
        
    print("\n📋 CURRENT DOOR CONFIGURATION")
    print("=" * 50)
    print(f"🚪 Door area: {config['door_area']}")
    
    if config['door_area']:
        x1, y1, x2, y2 = config['door_area']
        width = x2 - x1
        height = y2 - y1
        print(f"📐 Door size: {width} x {height} pixels")
        
        if height > width * 1.2:
            orientation = "vertical"
        else:
            orientation = "horizontal"
        print(f"🔄 Orientation: {orientation}")
    
    print(f"📍 Inside direction: {config['inside_direction']}")
    print(f"🚶‍♂️ Entry count: {config['entry_count']}")
    print(f"🚶‍♀️ Exit count: {config['exit_count']}")
    print(f"👥 People in room: {config['entry_count'] - config['exit_count']}")
    print("=" * 50)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "apply":
            apply_optimized_config()
        elif command == "status":
            print_config_summary()
        elif command == "reset":
            reset_counts()
        else:
            print("Usage: python door_config_helper.py [apply|status|reset]")
    else:
        print("🔧 Door Configuration Helper")
        print("Available commands:")
        print("1. apply - Apply optimized configuration")
        print("2. status - Show current configuration")  
        print("3. reset - Reset entry/exit counts")
        
        choice = input("Enter command (1-3): ").strip()
        
        if choice == "1":
            apply_optimized_config()
        elif choice == "2":
            print_config_summary()
        elif choice == "3":
            reset_counts()
        else:
            print("Invalid choice")

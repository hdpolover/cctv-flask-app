#!/usr/bin/env python3
"""
Door Configuration Helper Tool
Provides recommendations for optimal door area setup based on camera position and door type.
"""

import requests
import json

class DoorConfigHelper:
    """Helper class for optimal door configuration"""
    
    def __init__(self, server_url="http://127.0.0.1:5000"):
        self.server_url = server_url
        
    def calculate_optimal_door_area(self, door_type, camera_position, frame_size):
        """
        Calculate optimal door area based on setup parameters.
        
        Args:
            door_type: "standard" (0.9m), "wide" (1.2m), "double" (1.8m)
            camera_position: "overhead", "side_angle", "front_facing"
            frame_size: (width, height) of video frame
            
        Returns:
            Dictionary with recommended door area coordinates
        """
        frame_width, frame_height = frame_size
        
        # Base door width ratios (as percentage of frame width)
        door_width_ratios = {
            "standard": 0.15,   # 15% of frame width
            "wide": 0.20,       # 20% of frame width  
            "double": 0.30      # 30% of frame width
        }
        
        # Door height ratios based on camera position
        height_ratios = {
            "overhead": 0.25,     # 25% of frame height
            "side_angle": 0.35,   # 35% of frame height
            "front_facing": 0.40  # 40% of frame height
        }
        
        door_width = int(frame_width * door_width_ratios.get(door_type, 0.15))
        door_height = int(frame_height * height_ratios.get(camera_position, 0.35))
        
        # Center the door area
        center_x = frame_width // 2
        center_y = frame_height // 2
        
        x1 = center_x - door_width // 2
        x2 = center_x + door_width // 2
        y1 = center_y - door_height // 2
        y2 = center_y + door_height // 2
        
        return {
            "x1": x1,
            "y1": y1, 
            "x2": x2,
            "y2": y2,
            "width": door_width,
            "height": door_height,
            "orientation": "vertical" if door_height > door_width * 1.2 else "horizontal"
        }
    
    def recommend_inside_direction(self, camera_position, room_layout):
        """
        Recommend inside direction based on camera setup.
        
        Args:
            camera_position: "overhead", "side_angle", "front_facing"
            room_layout: "room_left", "room_right", "room_below", "room_above"
            
        Returns:
            Recommended inside direction string
        """
        direction_map = {
            ("overhead", "room_below"): "down",
            ("overhead", "room_above"): "up", 
            ("overhead", "room_left"): "left",
            ("overhead", "room_right"): "right",
            ("side_angle", "room_left"): "left",
            ("side_angle", "room_right"): "right",
            ("front_facing", "room_below"): "down"
        }
        
        return direction_map.get((camera_position, room_layout), "down")
    
    def get_current_config(self):
        """Get current door configuration from server."""
        try:
            response = requests.get(f"{self.server_url}/api/door-area")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def apply_config(self, door_area, inside_direction):
        """Apply door configuration to server."""
        try:
            config = {
                "x1": door_area["x1"],
                "y1": door_area["y1"],
                "x2": door_area["x2"], 
                "y2": door_area["y2"],
                "inside_direction": inside_direction
            }
            
            response = requests.post(
                f"{self.server_url}/api/door-area",
                headers={"Content-Type": "application/json"},
                data=json.dumps(config)
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def optimize_detection_params(self, environment_type):
        """
        Get recommended detection parameters for different environments.
        
        Args:
            environment_type: "office", "retail", "residential", "outdoor"
            
        Returns:
            Dictionary with recommended detection parameters
        """
        params = {
            "office": {
                "score_threshold": 0.75,
                "iou_threshold": 0.35,
                "tracking_threshold": 45,
                "description": "Balanced for professional environment with moderate traffic"
            },
            "retail": {
                "score_threshold": 0.70,
                "iou_threshold": 0.30,
                "tracking_threshold": 40,
                "description": "Lower threshold for busy retail environment"
            },
            "residential": {
                "score_threshold": 0.80,
                "iou_threshold": 0.40,
                "tracking_threshold": 50,
                "description": "Higher precision for low-traffic residential use"
            },
            "outdoor": {
                "score_threshold": 0.65,
                "iou_threshold": 0.25,
                "tracking_threshold": 60,
                "description": "Adapted for outdoor lighting and weather conditions"
            }
        }
        
        return params.get(environment_type, params["office"])

def interactive_config():
    """Interactive configuration wizard"""
    helper = DoorConfigHelper()
    
    print("🚪 Door Configuration Helper")
    print("=" * 40)
    
    # Get frame size (you can get this from /api/video-status)
    print("1. What's your video frame size?")
    width = int(input("   Frame width (e.g., 640): ") or "640")
    height = int(input("   Frame height (e.g., 480): ") or "480")
    
    print("\n2. What type of door are you monitoring?")
    print("   a) Standard door (0.9m)")
    print("   b) Wide door (1.2m)")  
    print("   c) Double door (1.8m)")
    door_type = {"a": "standard", "b": "wide", "c": "double"}[
        input("   Choice (a/b/c): ").lower() or "a"
    ]
    
    print("\n3. What's your camera position?")
    print("   a) Overhead view")
    print("   b) Side angle")
    print("   c) Front facing")
    camera_pos = {"a": "overhead", "b": "side_angle", "c": "front_facing"}[
        input("   Choice (a/b/c): ").lower() or "a"
    ]
    
    print("\n4. Where is the room relative to the door?")
    print("   a) Room is to the left")
    print("   b) Room is to the right") 
    print("   c) Room is below (camera above)")
    print("   d) Room is above (camera below)")
    room_layout = {"a": "room_left", "b": "room_right", "c": "room_below", "d": "room_above"}[
        input("   Choice (a/b/c/d): ").lower() or "c"
    ]
    
    print("\n5. What's your environment type?")
    print("   a) Office")
    print("   b) Retail store")
    print("   c) Residential")
    print("   d) Outdoor")
    env_type = {"a": "office", "b": "retail", "c": "residential", "d": "outdoor"}[
        input("   Choice (a/b/c/d): ").lower() or "a"
    ]
    
    # Calculate recommendations
    door_area = helper.calculate_optimal_door_area(door_type, camera_pos, (width, height))
    inside_dir = helper.recommend_inside_direction(camera_pos, room_layout)
    detection_params = helper.optimize_detection_params(env_type)
    
    print("\n" + "=" * 40)
    print("📋 RECOMMENDED CONFIGURATION")
    print("=" * 40)
    
    print(f"Door Area: ({door_area['x1']}, {door_area['y1']}) to ({door_area['x2']}, {door_area['y2']})")
    print(f"Door Size: {door_area['width']}x{door_area['height']} pixels")
    print(f"Orientation: {door_area['orientation']}")
    print(f"Inside Direction: {inside_dir}")
    print(f"Environment: {env_type} - {detection_params['description']}")
    
    print(f"\nDetection Parameters:")
    print(f"  Score Threshold: {detection_params['score_threshold']}")
    print(f"  IoU Threshold: {detection_params['iou_threshold']}")
    print(f"  Tracking Threshold: {detection_params['tracking_threshold']}")
    
    # Apply configuration
    apply = input("\nApply this configuration? (y/n): ").lower() == 'y'
    if apply:
        result = helper.apply_config(door_area, inside_dir)
        if "error" in result:
            print(f"❌ Error applying config: {result['error']}")
        else:
            print("✅ Configuration applied successfully!")
            print("💡 Note: Detection parameters need to be updated in the Flask app config")
    
    return {
        "door_area": door_area,
        "inside_direction": inside_dir,
        "detection_params": detection_params
    }

if __name__ == "__main__":
    interactive_config()

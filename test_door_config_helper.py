#!/usr/bin/env python3
"""
Test the door configuration helper functions in the same session
"""

from door_config_helper import set_door_config, get_current_door_config, reset_counts

def test_door_config_session():
    """Test that door configuration works within the same session"""
    print("🧪 Testing door configuration in same session...")
    
    # Get initial config
    print("\n1. Initial configuration:")
    config = get_current_door_config()
    print(f"   Door area: {config.get('door_area')}")
    print(f"   Inside direction: {config.get('inside_direction')}")
    
    # Apply optimized config
    print("\n2. Applying optimized configuration...")
    result = set_door_config(
        door_area=(320, 180, 480, 360),
        inside_direction="left"
    )
    print(f"   Result: {result}")
    
    # Check config again
    print("\n3. Configuration after setting:")
    config = get_current_door_config()
    print(f"   Door area: {config.get('door_area')}")
    print(f"   Inside direction: {config.get('inside_direction')}")
    
    # Test reset
    print("\n4. Testing reset...")
    reset_counts()
    
    print("\n✅ Session test completed!")

if __name__ == "__main__":
    test_door_config_session()

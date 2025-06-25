#!/usr/bin/env python3
"""
Simple validation test for the door configuration helper
"""

from door_config_helper import set_door_config, get_current_door_config, reset_counts

def test_helper_functionality():
    """Test that the helper functions work correctly within the same session"""
    print("🔧 DOOR CONFIGURATION HELPER VALIDATION")
    print("=" * 50)
    
    # Test 1: Check initial configuration
    print("\n1. Initial configuration check:")
    config = get_current_door_config()
    if "error" in config:
        print(f"❌ Error getting config: {config['error']}")
        return False
    
    print(f"   Door area: {config.get('door_area', 'None')}")
    print(f"   Inside direction: {config.get('inside_direction', 'None')}")
    print(f"   Entries: {config.get('entry_count', 0)}")
    print(f"   Exits: {config.get('exit_count', 0)}")
    
    # Test 2: Apply optimized configuration
    print("\n2. Applying optimized configuration...")
    result = set_door_config(
        door_area=(320, 180, 480, 360),
        inside_direction="left"
    )
    
    if not result.get("success"):
        print(f"❌ Failed to apply configuration: {result.get('error')}")
        return False
    
    print(f"✅ Configuration applied successfully")
    print(f"   Door area: {result.get('door_area')}")
    print(f"   Inside direction: {result.get('inside_direction')}")
    
    # Test 3: Verify configuration was set
    print("\n3. Verifying configuration...")
    config = get_current_door_config()
    expected_area = (320, 180, 480, 360)
    expected_direction = "left"
    
    if config.get('door_area') == expected_area and config.get('inside_direction') == expected_direction:
        print("✅ Configuration verified correctly")
        print(f"   Door area: {config['door_area']}")
        print(f"   Inside direction: {config['inside_direction']}")
    else:
        print("❌ Configuration mismatch!")
        print(f"   Expected: area={expected_area}, direction={expected_direction}")
        print(f"   Actual: area={config.get('door_area')}, direction={config.get('inside_direction')}")
        return False
    
    # Test 4: Test reset functionality
    print("\n4. Testing reset functionality...")
    try:
        reset_counts()
        print("✅ Reset command executed successfully")
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        return False
    
    # Test 5: Verify configuration persists after reset
    print("\n5. Verifying configuration persists after reset...")
    config = get_current_door_config()
    
    if config.get('door_area') == expected_area and config.get('inside_direction') == expected_direction:
        print("✅ Configuration persisted after reset")
        print(f"   Door area: {config['door_area']}")
        print(f"   Inside direction: {config['inside_direction']}")
        print(f"   Entries: {config.get('entry_count', 0)}")
        print(f"   Exits: {config.get('exit_count', 0)}")
    else:
        print("❌ Configuration lost after reset!")
        return False
    
    return True

if __name__ == "__main__":
    success = test_helper_functionality()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("\nThe door configuration helper is working correctly!")
        print("\nKey features validated:")
        print("✅ Configuration can be applied programmatically")
        print("✅ Configuration persists within the same session")
        print("✅ Status can be retrieved")
        print("✅ Counters can be reset")
        print("✅ Configuration survives counter reset")
        print("\nOptimal settings for demo.mp4:")
        print("   Door area: (320, 180, 480, 360)")
        print("   Inside direction: left")
        print("\nUsage:")
        print("   python door_config_helper.py apply   # Apply optimal config")
        print("   python door_config_helper.py status  # Check current config")
        print("   python door_config_helper.py reset   # Reset counters")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the door configuration helper implementation.")

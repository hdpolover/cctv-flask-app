"""
Create a test alert in Firebase
"""
import sys
import os
import argparse

# Add parent directory to path so we can import app modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

def create_test_alert(alert_type="system", severity="info", message=None):
    """Create a test alert in Firebase."""
    from app.core.firebase_client import init_firebase, save_alert
    
    # Initialize Firebase
    db = init_firebase()
    if not db:
        print("Failed to initialize Firebase")
        return False
    
    # Default message
    if message is None:
        message = f"Test {severity} alert from test script"
    
    # Create metadata
    metadata = {
        "source": "test_script",
        "test": True,
        "details": "This is a test alert created for verification purposes"
    }
    
    # Save alert
    alert_id = save_alert(alert_type, message, severity, metadata)
    
    print(f"Created {severity} {alert_type} alert with ID: {alert_id}")
    return True

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Create a test alert in Firebase")
    parser.add_argument("-t", "--type", default="system", choices=["system", "security", "crowd"],
                      help="Alert type (default: system)")
    parser.add_argument("-s", "--severity", default="info", choices=["info", "warning", "critical"],
                      help="Alert severity (default: info)")
    parser.add_argument("-m", "--message", default=None,
                      help="Custom alert message")
    
    args = parser.parse_args()
    
    # Create test alert
    create_test_alert(args.type, args.severity, args.message)

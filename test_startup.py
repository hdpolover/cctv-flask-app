"""
Test script for RTSP connectivity and Flask app startup
"""
import os
import sys
import subprocess
import time
import cv2
import logging

# Add the app directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_rtsp_basic(rtsp_url):
    """Basic RTSP connection test"""
    logger.info(f"Testing RTSP connection to: {rtsp_url}")
    
    try:
        cap = cv2.VideoCapture(rtsp_url)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                logger.info("✓ RTSP connection successful")
                logger.info(f"Frame shape: {frame.shape}")
                cap.release()
                return True
            else:
                logger.error("✗ RTSP opened but can't read frames")
        else:
            logger.error("✗ Failed to open RTSP stream")
        
        cap.release()
        return False
        
    except Exception as e:
        logger.error(f"✗ RTSP test failed: {e}")
        return False

def test_flask_app():
    """Test Flask app startup"""
    logger.info("Testing Flask app startup...")
    
    try:
        # Import the app
        from run import app
        
        # Test if we can create the app context
        with app.app_context():
            logger.info("✓ Flask app context created successfully")
            
            # Test basic route
            with app.test_client() as client:
                response = client.get('/')
                logger.info(f"✓ Root route status: {response.status_code}")
                
        return True
        
    except Exception as e:
        logger.error(f"✗ Flask app test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("CCTV Flask App - Startup Test")
    print("=" * 60)
    
    # Test 1: RTSP Connection
    rtsp_url = "rtsp://192.168.1.9/V_ENC_001"
    print(f"\n1. Testing RTSP connection...")
    rtsp_success = test_rtsp_basic(rtsp_url)
    
    # Test 2: Flask App
    print(f"\n2. Testing Flask app...")
    flask_success = test_flask_app()
    
    # Summary
    print(f"\n" + "=" * 60)
    print("TEST SUMMARY:")
    print(f"RTSP Connection: {'✓ PASS' if rtsp_success else '✗ FAIL'}")
    print(f"Flask App:       {'✓ PASS' if flask_success else '✗ FAIL'}")
    print("=" * 60)
    
    if rtsp_success and flask_success:
        print("\n✓ All tests passed! You can now start the Flask app.")
        print("\nTo start the app:")
        print("  python run.py")
        print("\nTo debug RTSP issues:")
        print(f"  python rtsp_test.py {rtsp_url}")
    else:
        print("\n✗ Some tests failed. Please check the issues above.")
        if not rtsp_success:
            print("\nRTSP Troubleshooting:")
            print("- Verify camera IP and URL path")
            print("- Check network connectivity")
            print("- Try the URL in VLC media player")
            print("- Check if authentication is required")
        
        if not flask_success:
            print("\nFlask App Troubleshooting:")
            print("- Check if all dependencies are installed")
            print("- Verify Python environment")
            print("- Check for import errors")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script for camera preview service
"""
import os
import sys
import time
import logging

# Add the app directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.video.camera_preview_service import CameraPreviewService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_camera_preview():
    """Test the camera preview service"""
    logger.info("Testing Camera Preview Service")
    
    # Test with demo video first
    demo_path = 'app/static/videos/demo.mp4'
    if os.path.exists(demo_path):
        logger.info(f"Testing with demo video: {demo_path}")
        preview_service = CameraPreviewService(demo_path, frame_rate=10, resolution=(320, 240))
        
        # Get status
        status = preview_service.get_status()
        logger.info(f"Demo video status: {status}")
        
        # Try to get a few frames
        logger.info("Testing frame capture...")
        for i in range(5):
            frame = preview_service.get_frame()
            if frame is not None:
                logger.info(f"Frame {i+1}: {frame.shape}")
            else:
                logger.warning(f"Frame {i+1}: None")
            time.sleep(0.1)
        
        # Clean up
        preview_service.release()
        logger.info("Demo video test completed")
    else:
        logger.warning(f"Demo video not found at: {demo_path}")
    
    # Test with camera (index 0)
    logger.info("Testing with camera index 0")
    preview_service = CameraPreviewService(0, frame_rate=10, resolution=(320, 240))
    
    # Get status
    status = preview_service.get_status()
    logger.info(f"Camera status: {status}")
    
    # Try to get a few frames
    logger.info("Testing camera frame capture...")
    for i in range(3):
        frame = preview_service.get_frame()
        if frame is not None:
            logger.info(f"Camera frame {i+1}: {frame.shape}")
        else:
            logger.warning(f"Camera frame {i+1}: None")
        time.sleep(0.2)
    
    # Clean up
    preview_service.release()
    logger.info("Camera test completed")

if __name__ == "__main__":
    try:
        test_camera_preview()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

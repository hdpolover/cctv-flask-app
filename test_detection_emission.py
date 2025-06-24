#!/usr/bin/env python3
"""
Test script to verify detection and counter emission is working properly.
"""
import sys
import os
import time
import logging

# Add the app directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

from app import create_app, socketio
from app.core.routes import detection_model, video_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_detection_system():
    """Test the detection system and counter emissions."""
    
    app = create_app()
    
    with app.app_context():
        logger.info("Starting detection system test...")
        
        # Force initialize services
        from app.core.routes import initialize_services
        initialize_services()
        
        # Import after initialization
        from app.core.routes import detection_model, video_service
        
        # Wait for services to initialize
        time.sleep(2)
        
        # Check if services are initialized
        if detection_model is None:
            logger.error("Detection model is not initialized!")
            return False
            
        if video_service is None:
            logger.error("Video service is not initialized!")
            return False
            
        logger.info("✓ Services are initialized")
        
        # Test detection model
        try:
            entries, exits = detection_model.get_entry_exit_count()
            logger.info(f"✓ Detection model working: entries={entries}, exits={exits}")
        except Exception as e:
            logger.error(f"✗ Detection model error: {e}")
            return False
        
        # Test video service detection status
        try:
            status = video_service.get_detection_status()
            logger.info(f"✓ Video service status: {status}")
        except Exception as e:
            logger.error(f"✗ Video service status error: {e}")
            return False
        
        # Test frame capture
        try:
            frame = video_service.get_frame()
            if frame is not None:
                logger.info(f"✓ Frame capture working: shape={frame.shape}")
            else:
                logger.warning("⚠ No frame captured (may be normal for some sources)")
        except Exception as e:
            logger.error(f"✗ Frame capture error: {e}")
            return False
        
        # Test counter emission
        try:
            video_service.force_counter_update()
            logger.info("✓ Counter update emission test completed")
        except Exception as e:
            logger.error(f"✗ Counter emission error: {e}")
            return False
        
        # Test detection processing if frame is available
        if frame is not None:
            try:
                processed_frame = video_service.process_frame(frame.copy())
                if processed_frame is not None:
                    logger.info("✓ Frame processing working")
                else:
                    logger.warning("⚠ Frame processing returned None")
            except Exception as e:
                logger.error(f"✗ Frame processing error: {e}")
                return False
        
        logger.info("🎉 All tests completed successfully!")
        return True

if __name__ == '__main__':
    success = test_detection_system()
    sys.exit(0 if success else 1)

"""
RTSP Connection Test Utility
"""
import cv2
import logging
import time
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_rtsp_connection(rtsp_url, timeout=30):
    """Test RTSP connection with multiple backends and settings.
    
    Args:
        rtsp_url: RTSP URL to test
        timeout: Maximum time to wait for connection
    """
    logger.info(f"Testing RTSP connection to: {rtsp_url}")
    
    # Try different backends
    backends = [
        ('CAP_FFMPEG', cv2.CAP_FFMPEG),
        ('CAP_GSTREAMER', cv2.CAP_GSTREAMER),
        ('CAP_ANY', cv2.CAP_ANY),
        ('Default', None)
    ]
    
    for backend_name, backend in backends:
        logger.info(f"\n--- Testing with {backend_name} backend ---")
        
        try:
            # Initialize capture
            if backend is not None:
                cap = cv2.VideoCapture(rtsp_url, backend)
            else:
                cap = cv2.VideoCapture(rtsp_url)
            
            logger.info(f"VideoCapture created: {cap is not None}")
            
            if cap.isOpened():
                logger.info("✓ Stream opened successfully")
                
                # Set properties
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                
                # Get stream properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                logger.info(f"Stream properties: {width}x{height} @ {fps} FPS")
                
                # Try to read frames
                successful_reads = 0
                failed_reads = 0
                start_time = time.time()
                
                for i in range(10):  # Try to read 10 frames
                    ret, frame = cap.read()
                    
                    if ret and frame is not None and frame.size > 0:
                        successful_reads += 1
                        logger.info(f"✓ Frame {i+1}: {frame.shape}")
                        
                        # Save first frame as test
                        if i == 0:
                            cv2.imwrite(f"rtsp_test_frame_{backend_name.lower()}.jpg", frame)
                            logger.info(f"Saved test frame as rtsp_test_frame_{backend_name.lower()}.jpg")
                    else:
                        failed_reads += 1
                        logger.warning(f"✗ Frame {i+1}: Failed to read")
                    
                    # Add small delay between reads
                    time.sleep(0.1)
                
                logger.info(f"Results: {successful_reads} successful, {failed_reads} failed reads")
                
                if successful_reads > 0:
                    logger.info(f"✓ {backend_name} backend WORKS!")
                    cap.release()
                    return True
                else:
                    logger.error(f"✗ {backend_name} backend failed to read frames")
                
            else:
                logger.error(f"✗ Failed to open stream with {backend_name}")
            
            cap.release()
            
        except Exception as e:
            logger.error(f"✗ Exception with {backend_name}: {e}")
            
        logger.info("-" * 50)
    
    logger.error("All backends failed!")
    return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python rtsp_test.py <rtsp_url>")
        print("Example: python rtsp_test.py rtsp://192.168.1.9/V_ENC_001")
        sys.exit(1)
    
    rtsp_url = sys.argv[1]
    
    # Test the RTSP connection
    success = test_rtsp_connection(rtsp_url)
    
    if success:
        print("\n✓ RTSP connection test PASSED")
    else:
        print("\n✗ RTSP connection test FAILED")
        print("\nTroubleshooting suggestions:")
        print("1. Check if the RTSP URL is correct")
        print("2. Verify network connectivity to the camera")
        print("3. Check if camera requires authentication (user:pass@ip)")
        print("4. Try different RTSP transport (TCP vs UDP)")
        print("5. Check camera's RTSP port (usually 554)")
        print("6. Verify camera supports the stream path (/V_ENC_001)")

if __name__ == "__main__":
    main()

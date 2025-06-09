"""
Video capture initialization and management.
"""
import cv2
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)

class VideoCaptureManager:
    """Manager for video capture devices and streams."""
    
    def __init__(self, video_path=0, frame_rate=30, resolution=(640, 480)):
        """Initialize the video capture manager.
        
        Args:
            video_path: Path to video file or camera index/URL
            frame_rate: Target frame rate for processing
            resolution: Resolution as (width, height) tuple
        """
        self.video_path = video_path
        self.frame_rate = frame_rate
        self.resolution = resolution
        
        # Source type flags
        self.is_file = False
        self.is_rtsp = False
        self.is_camera = False
        self._determine_source_type()
        
        # Initialize capture
        self.cap = self._initialize_capture()
        
    def _determine_source_type(self):
        """Determine the type of video source (file, camera, or RTSP)."""
        # Check if it's a camera index
        if isinstance(self.video_path, int) or (isinstance(self.video_path, str) and self.video_path.isdigit()):
            self.is_camera = True
            self.is_file = False
            self.is_rtsp = False
            return
        
        # Check if it's an RTSP URL
        if isinstance(self.video_path, str):
            # Check for RTSP protocol
            if self.video_path.lower().startswith(('rtsp://', 'rtmp://')):
                self.is_rtsp = True
                self.is_file = False
                self.is_camera = False
                return
                
            # Check if it's a file that exists on disk
            if os.path.exists(self.video_path):
                self.is_file = True
                self.is_rtsp = False
                self.is_camera = False
                return
                
        # Default to camera if we can't determine
        self.is_camera = True
        self.is_file = False
        self.is_rtsp = False
        
    def _initialize_capture(self):
        """Initialize the video capture object.
        
        Returns:
            OpenCV VideoCapture object
        """
        try:
            # Initialize cap to None to avoid reference before assignment issues
            cap = None
            
            # Convert string digit path to integer
            if isinstance(self.video_path, str) and self.video_path.isdigit():
                self.video_path = int(self.video_path)
            
            # Determine the appropriate initialization method based on source type
            if isinstance(self.video_path, int):
                # For camera devices, use DirectShow on Windows for better performance
                logger.info(f"Initializing camera device: {self.video_path}")
                cap = cv2.VideoCapture(self.video_path, cv2.CAP_DSHOW)
                self.is_camera = True
                
            elif isinstance(self.video_path, str) and self.video_path.lower().startswith(('rtsp://', 'rtmp://')):
                # For RTSP streams, use enhanced settings and multiple backend attempts
                logger.info(f"Initializing RTSP stream: {self.video_path}")
                
                cap = None
                backends_to_try = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
                
                for backend in backends_to_try:
                    try:
                        logger.info(f"Trying RTSP with backend: {backend}")
                        cap = cv2.VideoCapture(self.video_path, backend)
                        
                        if cap.isOpened():
                            # Configure RTSP-specific settings for better reliability
                            # Set smaller buffer size for reduced latency
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            
                            # Set timeout values to prevent hanging
                            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10 second timeout
                            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)   # 5 second read timeout
                            
                            # Try to set codec (optional, may not work on all streams)
                            try:
                                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                            except:
                                pass  # Ignore if codec setting fails
                            
                            # Test if we can actually read a frame
                            ret, test_frame = cap.read()
                            if ret and test_frame is not None and test_frame.size > 0:
                                logger.info(f"Successfully connected to RTSP stream with backend: {backend}")
                                # Reset position to beginning
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                break
                            else:
                                logger.warning(f"Backend {backend} opened but couldn't read frames")
                                cap.release()
                                cap = None
                        else:
                            logger.warning(f"Failed to open RTSP stream with backend: {backend}")
                            if cap:
                                cap.release()
                            cap = None
                    except Exception as e:
                        logger.warning(f"Exception with backend {backend}: {e}")
                        if cap:
                            cap.release()
                        cap = None
                
                if cap is None:
                    # If all backends failed, try with default settings one more time
                    logger.info("All backends failed, trying default OpenCV settings...")
                    try:
                        cap = cv2.VideoCapture(self.video_path)
                        if cap.isOpened():
                            # Basic settings for default capture
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            # Test frame read
                            ret, test_frame = cap.read()
                            if not ret or test_frame is None:
                                logger.error("Default capture opened but can't read frames")
                                cap.release()
                                cap = None
                    except Exception as e:
                        logger.error(f"Default capture also failed: {e}")
                        cap = None
                
                self.is_rtsp = True
                
            else:
                # Handle file paths
                if os.path.exists(self.video_path):
                    logger.info(f"Initializing video file: {self.video_path}")
                    cap = cv2.VideoCapture(self.video_path)
                    self.is_file = True
                else:
                    # If path doesn't exist, treat as URL or fallback
                    logger.info(f"Initializing URL or unknown source: {self.video_path}")
                    cap = cv2.VideoCapture(self.video_path)
            
            # Check if camera opened successfully
            if not cap or not cap.isOpened():
                logger.error(f"Failed to open video source: {self.video_path}")
                
                # Try to use default camera as fallback
                if self.video_path != 0:
                    logger.info("Attempting to open default camera instead")
                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    self.is_camera = True
                    self.is_file = False
                    self.is_rtsp = False
                    
                    if not cap.isOpened():
                        logger.error("Failed to open default camera as fallback")
                        return cap
            
            # Set capture properties for performance
            if cap and cap.isOpened():
                if self.is_camera:
                    # Camera-specific settings
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for less latency
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                    cap.set(cv2.CAP_PROP_FPS, self.frame_rate)
                elif self.is_file:
                    # File-specific settings
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                # RTSP settings are already configured above
            
            logger.info(f"Video capture initialized successfully with source: {self.video_path}")
            return cap
        
        except Exception as e:
            logger.exception(f"Error initializing video capture: {e}")
            # Create a minimal capture as fallback
            cap = cv2.VideoCapture(0)
            return cap
    
    def get_source_info(self):
        """Get information about the current video source.
        
        Returns:
            Dict with video source information
        """
        # Get video properties if available
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap else self.resolution[0]
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap else self.resolution[1]
        fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else self.frame_rate
        
        # For video files, get total frame count and duration
        frame_count = 0
        duration = 0
        if self.is_file and self.cap:
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps > 0:
                duration = frame_count / fps
        
        # Determine source name/path for display
        if isinstance(self.video_path, int):
            source_name = f"Camera {self.video_path}"
        else:
            source_name = self.video_path
            # For file paths, show just the filename
            if self.is_file and os.path.exists(self.video_path):
                source_name = os.path.basename(self.video_path)
        
        return {
            "source": source_name,
            "source_type": "camera" if self.is_camera else "rtsp" if self.is_rtsp else "file" if self.is_file else "unknown",
            "resolution": f"{width}x{height}",
            "target_fps": self.frame_rate,
            "original_fps": fps,
            "frame_count": frame_count,
            "duration": duration
        }
        
    def release(self):
        """Release the video capture resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
            logger.info("Video capture resources released")
            
    def reopen(self):
        """Reopen the video capture."""
        if self.cap:
            self.cap.release()
        self.cap = self._initialize_capture()
        return self.cap

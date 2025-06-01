"""
System Health Monitoring Script
Collects and logs system health metrics to Firebase
"""
import os
import sys
import time
import psutil
import argparse
import logging
from datetime import datetime

# Add parent directory to path so we can import app modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_system_metrics():
    """Get current system metrics."""
    try:
        # Get basic system metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        
        # Get CPU temperature if available
        temperature = None
        if hasattr(psutil, 'sensors_temperatures'):
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        # Use the first available temperature reading
                        if entry.current:
                            temperature = entry.current
                            break
                    if temperature:
                        break
        
        # Return all metrics
        return {
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'disk_usage': disk_usage,
            'temperature': temperature,
            'timestamp': datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        return None

def log_metrics_to_firebase(metrics, fps=None):
    """Log system metrics to Firebase."""
    try:
        # Import Firebase client
        from app.core.firebase_client import log_system_health
        
        # Add FPS if provided
        if fps is not None:
            metrics['fps'] = fps
        
        # Log to Firebase
        log_id = log_system_health(
            cpu_usage=metrics['cpu_usage'],
            memory_usage=metrics['memory_usage'],
            disk_usage=metrics['disk_usage'],
            temperature=metrics['temperature'],
            fps=fps
        )
        
        logger.info(f"Logged system health metrics to Firebase (ID: {log_id})")
        return True
    except Exception as e:
        logger.error(f"Error logging to Firebase: {e}")
        return False

def monitor_system_health(interval=60, duration=None, include_fps=False):
    """Monitor system health at regular intervals.
    
    Args:
        interval: Time between measurements in seconds
        duration: Total monitoring duration in seconds (None for indefinite)
        include_fps: Whether to include FPS in monitoring
    """
    # Initialize Firebase
    from app.core.firebase_client import init_firebase
    db = init_firebase()
    
    if not db:
        logger.error("Failed to initialize Firebase. Exiting.")
        return False
    
    logger.info(f"Starting system health monitoring (interval: {interval}s)")
    
    # For FPS calculation
    fps = None
    if include_fps:
        try:
            from app.core.routes import video_service
            if video_service:
                logger.info("Will include FPS in monitoring")
            else:
                logger.warning("Video service not initialized, FPS not available")
                include_fps = False
        except ImportError:
            logger.warning("Could not import video_service, FPS not available")
            include_fps = False
    
    start_time = time.time()
    count = 0
    
    try:
        while True:
            # Get current metrics
            metrics = get_system_metrics()
            
            if metrics:
                # Get FPS if enabled
                if include_fps and 'video_service' in locals():
                    fps = video_service.get_current_fps()
                
                # Log to Firebase
                log_metrics_to_firebase(metrics, fps)
                
                # Increment count
                count += 1
                logger.info(f"Collected metrics {count} times")
            
            # Check if duration has elapsed
            if duration is not None and (time.time() - start_time) >= duration:
                logger.info(f"Monitoring duration of {duration}s completed. Exiting.")
                break
            
            # Wait for next interval
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user.")
    except Exception as e:
        logger.error(f"Error during monitoring: {e}")
    
    return True

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Monitor system health and log to Firebase")
    parser.add_argument("-i", "--interval", type=int, default=60, 
                      help="Interval between measurements in seconds (default: 60)")
    parser.add_argument("-d", "--duration", type=int, default=None,
                      help="Total monitoring duration in seconds (default: indefinite)")
    parser.add_argument("--fps", action="store_true",
                      help="Include FPS in monitoring (requires running app)")
    parser.add_argument("-o", "--once", action="store_true",
                      help="Collect metrics just once and exit")
    
    args = parser.parse_args()
    
    # Run monitoring
    if args.once:
        # Just collect once
        metrics = get_system_metrics()
        if metrics:
            from app.core.firebase_client import init_firebase, log_system_health
            init_firebase()
            log_system_health(
                cpu_usage=metrics['cpu_usage'],
                memory_usage=metrics['memory_usage'],
                disk_usage=metrics['disk_usage'],
                temperature=metrics['temperature']
            )
            logger.info("Collected and logged metrics once.")
    else:
        # Start continuous monitoring
        monitor_system_health(
            interval=args.interval,
            duration=args.duration,
            include_fps=args.fps
        )

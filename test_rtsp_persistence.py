"""
Test script for RTSP stream persistence and reconnection.
"""
import time
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RTSPPersistenceTest:
    """Test RTSP stream persistence and automatic reconnection."""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.test_duration = 300  # 5 minutes
        self.check_interval = 10  # Check every 10 seconds
        
    def get_stream_health(self):
        """Get current stream health status."""
        try:
            response = requests.get(f"{self.base_url}/stream_health", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Health check failed with status {response.status_code}")
                return None
        except requests.RequestException as e:
            logger.error(f"Failed to get stream health: {e}")
            return None
    
    def refresh_stream(self):
        """Manually refresh the stream."""
        try:
            response = requests.post(f"{self.base_url}/refresh_video", timeout=10)
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Stream refresh: {result.get('message', 'Unknown result')}")
                return result.get('success', False)
            else:
                logger.error(f"Stream refresh failed with status {response.status_code}")
                return False
        except requests.RequestException as e:
            logger.error(f"Failed to refresh stream: {e}")
            return False
    
    def run_persistence_test(self):
        """Run the main persistence test."""
        logger.info("Starting RTSP persistence test")
        logger.info(f"Test duration: {self.test_duration} seconds")
        logger.info(f"Check interval: {self.check_interval} seconds")
        
        start_time = time.time()
        test_results = {
            'start_time': start_time,
            'checks': [],
            'reconnections_detected': 0,
            'manual_refreshes': 0,
            'health_failures': 0
        }
        
        while time.time() - start_time < self.test_duration:
            current_time = time.time()
            elapsed = current_time - start_time
            
            logger.info(f"[{elapsed:.0f}s] Checking stream health...")
            
            health = self.get_stream_health()
            if health:
                # Extract key metrics
                is_running = health.get('video_service_running', False)
                is_rtsp = health.get('is_rtsp', False)
                health_monitor = health.get('health_monitor', {})
                rtsp_monitor = health.get('rtsp_monitor', {})
                
                healthy = health_monitor.get('healthy', False)
                consecutive_failures = health_monitor.get('consecutive_failures', 0)
                time_since_last_read = health_monitor.get('time_since_last_read', 0)
                
                # RTSP monitor info
                rtsp_running = rtsp_monitor.get('running', False)
                stream_healthy = rtsp_monitor.get('stream_healthy', False)
                reconnection_count = rtsp_monitor.get('reconnection_count', 0)
                time_since_frame = rtsp_monitor.get('time_since_last_frame', 0)
                
                # Log current status
                status_msg = f"Running: {is_running}, RTSP: {is_rtsp}, Healthy: {healthy}"
                if is_rtsp and rtsp_running:
                    status_msg += f", Stream OK: {stream_healthy}, Reconnects: {reconnection_count}"
                    status_msg += f", Frame Age: {time_since_frame:.1f}s"
                
                logger.info(f"[{elapsed:.0f}s] {status_msg}")
                
                # Record check
                check_result = {
                    'time': current_time,
                    'elapsed': elapsed,
                    'is_running': is_running,
                    'is_rtsp': is_rtsp,
                    'healthy': healthy,
                    'consecutive_failures': consecutive_failures,
                    'time_since_last_read': time_since_last_read,
                    'rtsp_running': rtsp_running,
                    'stream_healthy': stream_healthy,
                    'reconnection_count': reconnection_count,
                    'time_since_frame': time_since_frame
                }
                test_results['checks'].append(check_result)
                
                # Track reconnections
                if len(test_results['checks']) > 1:
                    prev_reconnects = test_results['checks'][-2].get('reconnection_count', 0)
                    if reconnection_count > prev_reconnects:
                        test_results['reconnections_detected'] += (reconnection_count - prev_reconnects)
                        logger.warning(f"[{elapsed:.0f}s] Reconnection detected! Total: {reconnection_count}")
                
                # Track health failures
                if not healthy:
                    test_results['health_failures'] += 1
                    logger.warning(f"[{elapsed:.0f}s] Health check failed! Consecutive failures: {consecutive_failures}")
                    
                    # If health is really bad, try manual refresh
                    if consecutive_failures > 5 and time_since_last_read > 30:
                        logger.info(f"[{elapsed:.0f}s] Attempting manual refresh due to poor health")
                        if self.refresh_stream():
                            test_results['manual_refreshes'] += 1
                
            else:
                logger.error(f"[{elapsed:.0f}s] Failed to get stream health")
                test_results['health_failures'] += 1
            
            # Wait for next check
            time.sleep(self.check_interval)
        
        # Generate test report
        self.generate_report(test_results)
    
    def generate_report(self, results):
        """Generate a test report."""
        duration = results['checks'][-1]['elapsed'] if results['checks'] else 0
        total_checks = len(results['checks'])
        
        logger.info("=" * 60)
        logger.info("RTSP PERSISTENCE TEST REPORT")
        logger.info("=" * 60)
        logger.info(f"Test Duration: {duration:.0f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"Total Health Checks: {total_checks}")
        logger.info(f"Reconnections Detected: {results['reconnections_detected']}")
        logger.info(f"Manual Refreshes: {results['manual_refreshes']}")
        logger.info(f"Health Check Failures: {results['health_failures']}")
        
        if total_checks > 0:
            success_rate = ((total_checks - results['health_failures']) / total_checks) * 100
            logger.info(f"Health Check Success Rate: {success_rate:.1f}%")
            
            # Calculate stream uptime
            healthy_checks = sum(1 for check in results['checks'] if check.get('healthy', False))
            uptime_rate = (healthy_checks / total_checks) * 100
            logger.info(f"Stream Uptime: {uptime_rate:.1f}%")
            
            # Show reconnection frequency
            if results['reconnections_detected'] > 0:
                avg_time_between = duration / results['reconnections_detected']
                logger.info(f"Average Time Between Reconnections: {avg_time_between:.0f} seconds")
        
        logger.info("=" * 60)
        
        # Save detailed results
        with open('rtsp_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        logger.info("Detailed results saved to rtsp_test_results.json")

def main():
    """Main test function."""
    print("RTSP Stream Persistence Test")
    print("=" * 40)
    print("This test will monitor your RTSP stream for 5 minutes")
    print("and check for automatic reconnections and health status.")
    print()
    
    # Get user confirmation
    response = input("Make sure your Flask app is running. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Test cancelled.")
        return
    
    # Test basic connectivity first
    tester = RTSPPersistenceTest()
    print("Testing basic connectivity...")
    
    health = tester.get_stream_health()
    if not health:
        print("ERROR: Cannot connect to Flask app. Make sure it's running on localhost:5000")
        return
    
    print(f"✓ Connected to Flask app")
    print(f"✓ Video service running: {health.get('video_service_running', False)}")
    print(f"✓ RTSP stream: {health.get('is_rtsp', False)}")
    print()
    
    # Start the persistence test
    print("Starting persistence test...")
    try:
        tester.run_persistence_test()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed with error: {e}")

if __name__ == "__main__":
    main()

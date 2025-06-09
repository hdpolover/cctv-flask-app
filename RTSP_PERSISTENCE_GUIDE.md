# RTSP Stream Persistence Troubleshooting Guide

## Problem: RTSP Stream Stops After Some Time

Your RTSP stream works initially but stops streaming after a period of time. This is a common issue with RTSP streams that can be caused by several factors.

## Enhanced Solutions Implemented

### 1. **Automatic RTSP Stream Monitoring**
- **RTSPStreamMonitor**: Continuously monitors stream health every 15 seconds
- **Automatic Reconnection**: Detects when stream stops producing frames and automatically reconnects
- **Smart Timeout Detection**: Considers stream dead after 60 seconds without frames
- **Reconnection Throttling**: Prevents excessive reconnection attempts (max 10 per hour)

### 2. **Enhanced Frame Capture Loop**
- **Proactive Health Checks**: Monitors RTSP streams every 30 seconds for early problem detection
- **Multiple Reconnection Strategies**: Tries simple reopen first, then full recreation
- **Frame Read Validation**: Tests actual frame reading after reconnection attempts
- **Adaptive Timeouts**: Different timeout thresholds for RTSP (30s), camera (10s), and file (5s)

### 3. **Robust Error Handling**
- **Consecutive Failure Tracking**: Allows up to 15 consecutive read failures for RTSP streams
- **Exponential Backoff**: Increases wait time between reconnection attempts
- **Connection Testing**: Validates frame reading capability after each reconnection
- **Fallback Protection**: Falls back to demo video if all reconnection attempts fail

### 4. **Stream Health Monitoring**
- **Real-time Health API**: `/stream_health` endpoint for monitoring stream status
- **Enhanced Debug Page**: `/debug/rtsp` shows comprehensive connection diagnostics
- **Automatic Notifications**: Users get notified when stream issues are detected and resolved

## Common Causes of RTSP Stream Drops

### 1. **Network Issues**
- **Packet Loss**: Poor network quality can cause stream interruption
- **Bandwidth Limitations**: Insufficient bandwidth for sustained streaming
- **Router/Switch Issues**: Network equipment dropping long-lived connections

### 2. **Camera-Side Issues**
- **Connection Limits**: Many cameras limit concurrent RTSP connections
- **Firmware Bugs**: Camera firmware may have timeout or memory leak issues
- **Power Management**: Cameras may enter sleep mode or restart periodically

### 3. **RTSP Server Issues**
- **Session Timeouts**: RTSP servers may close inactive sessions
- **Memory Leaks**: Long-running sessions may exhaust server resources
- **Protocol Compliance**: Some cameras don't fully comply with RTSP standards

### 4. **Client-Side Issues**
- **OpenCV Backend**: Different backends handle RTSP differently
- **Buffer Management**: Frame buffers may overflow or underrun
- **Thread Synchronization**: Timing issues in multi-threaded applications

## How the Enhanced System Addresses These Issues

### **Automatic Recovery**
```python
# The system now includes:
- Continuous stream monitoring (every 15 seconds)
- Automatic reconnection when stream stops
- Multiple reconnection strategies
- Health status tracking and reporting
```

### **Smart Reconnection Logic**
1. **Detection**: Monitor detects no frames for 60 seconds
2. **Simple Reconnection**: Try reopening existing connection
3. **Full Reconnection**: If simple fails, recreate entire connection
4. **Validation**: Test frame reading before considering reconnection successful
5. **Throttling**: Limit reconnections to prevent overwhelming the camera

### **User Feedback**
- **Visual Indicators**: UI shows connection status and health
- **Automatic Notifications**: Users informed of disconnections and reconnections
- **Debug Information**: Comprehensive diagnostics available at `/debug/rtsp`

## Testing the Enhanced System

### 1. **Monitor Stream Health**
```bash
# Check stream health via API
curl http://localhost:5000/stream_health
```

### 2. **View Debug Information**
```bash
# Open debug page in browser
http://localhost:5000/debug/rtsp
```

### 3. **Test Reconnection**
```bash
# Temporarily disconnect camera or block network
# The system should automatically reconnect when connection is restored
```

## Configuration Options

### **RTSP Monitor Settings**
- `check_interval`: How often to check stream (default: 15 seconds)
- `stream_timeout`: Consider dead after this time (default: 60 seconds)  
- `reconnection_cooldown`: Wait between reconnections (default: 30 seconds)
- `max_reconnections_per_hour`: Limit reconnection attempts (default: 10)

### **Health Monitor Settings**
- `rtsp_timeout`: RTSP health timeout (default: 30 seconds)
- `camera_timeout`: Camera health timeout (default: 10 seconds)
- `file_timeout`: File health timeout (default: 5 seconds)

## Best Practices for RTSP Reliability

### 1. **Network Optimization**
- Use wired connections when possible
- Ensure sufficient bandwidth (minimum 2-3x stream bitrate)
- Configure QoS to prioritize video traffic
- Use local network segments to reduce latency

### 2. **Camera Configuration**
- Set appropriate frame rate and resolution
- Configure camera for multiple concurrent connections
- Update camera firmware to latest version
- Use TCP transport for reliability over UDP

### 3. **Application Configuration**
- Set reasonable timeout values
- Use appropriate OpenCV backend for your system
- Monitor system resources (CPU, memory)
- Implement proper logging for troubleshooting

## Monitoring and Maintenance

### **Log Monitoring**
The enhanced system provides detailed logging:
```
INFO - Started RTSP stream monitor for automatic reconnection
WARNING - RTSP stream health check failed - 65.2s since last frame  
INFO - Triggering RTSP stream reconnection
INFO - RTSP stream reconnection successful
```

### **Health Endpoints**
- `/stream_health` - JSON API for programmatic monitoring
- `/debug/rtsp` - Human-readable debug information
- `/refresh_video` - Manual stream refresh endpoint

### **Automatic Features**
- ✅ **Continuous Monitoring**: 24/7 stream health checking
- ✅ **Automatic Reconnection**: No manual intervention required
- ✅ **Smart Throttling**: Prevents overwhelming camera/network
- ✅ **User Notifications**: Real-time status updates
- ✅ **Fallback Protection**: Demo video if all else fails

The enhanced RTSP persistence system should significantly improve stream reliability and automatically handle most common disconnection scenarios without user intervention.

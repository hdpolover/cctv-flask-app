# RTSP Troubleshooting Guide

## Problem: Orange Screen in Web Interface

You're experiencing an orange screen in the web interface while VLC can stream the RTSP feed successfully. This indicates the issue is with OpenCV's RTSP handling, not the camera itself.

## Solutions Implemented

### 1. Enhanced RTSP Connection Handling
- **Multiple Backend Support**: The app now tries different OpenCV backends (FFmpeg, GStreamer, default)
- **Connection Timeout**: Added 10-second connection timeout to prevent hanging
- **Frame Validation**: Tests actual frame reading before considering connection successful
- **Fallback Mechanisms**: Graceful fallback to different backends if one fails

### 2. Improved Video Refresh Functionality
- **Backend Refresh Endpoint**: `/refresh_video` endpoint that reinitializes the entire video service
- **Enhanced Frontend**: Refresh button now calls backend to fully restart video capture
- **Visual Feedback**: Loading states and status messages during refresh
- **Keyboard Shortcut**: Press 'R' to refresh video feed

### 3. Debug Tools Added
- **RTSP Test Utility**: `rtsp_test.py` - Test RTSP connections with different backends
- **Debug Page**: `/debug/rtsp` - View detailed connection diagnostics
- **Test Pattern**: Visual test pattern to verify video processing pipeline

## How to Use the New Features

### Testing RTSP Connection
```bash
# Test your RTSP URL directly
python rtsp_test.py "rtsp://192.168.1.9/V_ENC_001"
```

### Refreshing Video Feed
1. **Click the refresh button** (appears on hover over video)
2. **Press 'R' key** anywhere on the page
3. **Use the debug page** at `/debug/rtsp` for detailed diagnostics

### Startup Testing
```bash
# Test both RTSP and Flask app
python test_startup.py
```

## Common RTSP Issues and Solutions

### 1. Authentication Required
If your camera requires authentication, update the URL format:
```
rtsp://username:password@192.168.1.9/V_ENC_001
```

### 2. Different Stream Paths
Some cameras use different paths. Try these common variations:
```
rtsp://192.168.1.9/stream1
rtsp://192.168.1.9/live
rtsp://192.168.1.9/h264
rtsp://192.168.1.9:554/V_ENC_001
```

### 3. Network/Firewall Issues
- Ensure port 554 (default RTSP) is not blocked
- Try accessing from the same network as the camera
- Test with VLC to confirm basic connectivity

### 4. Camera-Specific Settings
Some cameras require specific transport protocols:
- TCP vs UDP transport
- Specific codec settings
- Buffer size adjustments

## Updated Configuration

The video capture now automatically:
1. **Tries multiple backends** in order of preference
2. **Tests frame reading** before confirming connection
3. **Provides detailed logging** for troubleshooting
4. **Handles reconnection** automatically
5. **Shows visual feedback** for connection status

## Next Steps

1. **Test the enhanced refresh**: Try the refresh button in the web interface
2. **Check the debug page**: Visit `/debug/rtsp` for detailed diagnostics  
3. **Run RTSP test**: Use `python rtsp_test.py` with your camera URL
4. **Monitor logs**: Check application logs for detailed error messages

## Files Modified

- `app/services/video/video_capture.py` - Enhanced RTSP handling
- `app/services/video/frame_processor.py` - Added test pattern generation
- `app/services/video/video_service.py` - Improved error handling
- `app/core/routes.py` - Added refresh endpoint and debug routes
- `app/static/js/video-feed.js` - Enhanced refresh functionality
- `app/static/css/styles.css` - Added refresh button styling
- `app/templates/debug_rtsp.html` - Debug diagnostics page

## Testing Commands

```bash
# Test RTSP connection
python rtsp_test.py "rtsp://192.168.1.9/V_ENC_001"

# Test app startup
python test_startup.py

# Start the Flask app
python run.py
```

The refresh functionality now provides a complete video service restart, which should resolve most RTSP connection issues that cause the orange screen problem.

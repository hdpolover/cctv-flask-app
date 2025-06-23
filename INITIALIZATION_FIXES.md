# Initialization and Multiple Instance Fixes

## Issues Fixed

### 1. Double Initialization in door-config.js
**Problem**: The door configuration JavaScript was being initialized twice due to two separate `DOMContentLoaded` event listeners.

**Fix**: 
- Consolidated into a single initialization function
- Added global variable `doorAreaInstance` to prevent multiple instances
- Added `doorAreaConfigInitialized` flag to track initialization status

### 2. Multiple Video Status Check Intervals
**Problem**: The camera settings page was starting multiple status check intervals that ran simultaneously, causing performance issues and potential conflicts.

**Fix**:
- Added `statusCheckInitialized` flag to prevent duplicate monitoring
- Added proper cleanup for intervals with `beforeunload` event listener
- Consolidated status checking into single timer management system

### 3. Multiple Service Initialization in Routes
**Problem**: The `before_app_request` hook was being called multiple times, potentially causing multiple service initializations.

**Fix**:
- Added thread-safe initialization with `threading.Lock()`
- Added `_initialized` flag to prevent duplicate initialization
- Used double-check pattern for thread safety

### 4. Multiple Video Capture Threads
**Problem**: Video service could start multiple capture threads if called repeatedly.

**Fix**:
- Enhanced `start_capture_thread()` to check for existing threads
- Added proper thread cleanup before starting new ones
- Enhanced `update_settings()` to properly manage thread lifecycle

### 5. Race Conditions in Video Refresh
**Problem**: Multiple rapid video refresh requests could cause conflicts.

**Fix**:
- Added `_refreshing` flag to prevent overlapping refresh operations
- Added proper error handling with `try/finally` blocks
- Added delays to ensure proper thread shutdown

### 6. Door Configuration Save Race Conditions
**Problem**: Multiple simultaneous door area save operations could conflict.

**Fix**:
- Added `_saving` flag to prevent overlapping save operations
- Enhanced error handling and status management

## Additional Issues Found

### 7. Video Fallback Misleading Success Messages
**Problem**: The system falls back to demo video and reports "success" even when the original camera source fails, misleading users about the actual status.

**Symptoms**:
- Logs show "Maximum attempts reached" followed by "demo fallback"
- Video capture reports "success" with demo.mp4
- Web interface shows "connection issues" despite success logs
- Detection model continues running on invalid/empty frames

**Root Causes**:
1. Fallback mechanism doesn't communicate failure to frontend
2. Detection model runs regardless of video source validity
3. Status reporting doesn't distinguish between intended source and fallback
4. Raw video feed generator has different behavior than main video feed

### 8. Detection Model Running on Invalid Frames
**Problem**: Detection model continues processing even when video source is unavailable or invalid.

**Symptoms**:
- Debug logs show detection timing even with no valid input
- GPU resources wasted on empty/invalid frames
- Performance impact from unnecessary processing

## Fixes Needed

1. **Improve fallback communication**: Frontend should know when fallback occurs
2. **Stop detection on invalid source**: Detection should pause when video is unavailable
3. **Better status reporting**: Distinguish between intended source and fallback status
4. **Optimize detection**: Only run when valid frames are available

## Key Improvements

1. **Thread Safety**: All initialization now uses proper locking mechanisms
2. **Resource Management**: Proper cleanup of intervals, threads, and video resources
3. **Race Condition Prevention**: Flags to prevent overlapping operations
4. **Better Error Handling**: More robust error handling with proper cleanup
5. **Performance**: Reduced redundant operations and resource usage

## Files Modified

1. `app/static/js/door-config.js` - Fixed double initialization
2. `app/templates/camera-settings.html` - Fixed multiple status intervals
3. `app/core/routes.py` - Added thread-safe initialization and refresh protection
4. `app/services/video/video_service.py` - Enhanced thread management

## Testing Recommendations

1. Test camera configuration multiple times in succession
2. Test rapid navigation between pages
3. Test door area configuration with quick successive saves
4. Monitor resource usage to ensure no memory leaks
5. Test with different video sources (camera, RTSP, demo video)

## Benefits

- **Stability**: Eliminates crashes from multiple initializations
- **Performance**: Reduces CPU and memory usage
- **User Experience**: Prevents error messages about failed camera settings
- **Reliability**: More consistent behavior across different usage patterns

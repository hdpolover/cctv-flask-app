# Door Area Configuration Fix

## Problem
When changing the door area configuration, the video feed would momentarily show a gray screen with "connection issues" error message. This happened because:

1. The `set_door_area` method was immediately resetting all tracking state with `self.previous_centers = {}`
2. This sudden change was causing a brief interruption in the tracking process
3. The video feed was interpreting this as a connection issue and showing an error

## Solution
The fix implements a gradual tracking state refresh approach:

1. Added logic to determine if door area changes are significant enough to warrant a reset
2. Replaced immediate tracking state reset with a gradual refresh over several frames
3. Added tracking state management to prevent video feed interruptions

## Technical Changes

### 1. Added Area Change Significance Detection
```python
def _area_changed_significantly(self, old_area, new_area):
    # Determine if door area changed enough to warrant tracking reset
    # Only reset if area size changed by >20% or position shifted by >20%
```

### 2. Implemented Gradual Tracking Refresh
```python
def _mark_tracking_state_for_refresh(self):
    # Instead of immediate reset, mark for cleanup over next 10 frames
    self._tracking_refresh_counter = 10
    self._old_previous_centers = self.previous_centers.copy()
```

### 3. Modified Frame Processing Logic
```python
# In detect_people method:
if self._tracking_refresh_counter > 0:
    self._tracking_refresh_counter -= 1
    if self._tracking_refresh_counter == 0:
        # Now fully reset tracking state after a few frames
        self.previous_centers = {}
```

### 4. Updated set_door_area Method
```python
def set_door_area(self, x1, y1, x2, y2):
    # Store previous area to check if it actually changed
    old_area = self.door_area
    new_area = (x1, y1, x2, y2)
    
    # Only reset tracking if area changed significantly
    if old_area is None or self._area_changed_significantly(old_area, new_area):
        # Mark for gradual cleanup instead of immediate reset
        self._mark_tracking_state_for_refresh()
```

## Expected Result

Now when you edit the door area:
1. The video feed should remain stable without interruption
2. Tracking state will be gradually refreshed over ~10 frames (unnoticeable to users)
3. The door area configuration will update smoothly
4. No more gray screen or "connection issues" errors

This fix preserves all functionality while eliminating the interruption in the video feed.

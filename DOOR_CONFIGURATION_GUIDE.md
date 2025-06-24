# 🚪 Door Configuration & Detection Optimization Guide

## Quick Setup for Current System

Your CCTV system is already configured with:
- **Current Door Area**: (266, 74, 451, 315) 
- **Inside Direction**: down
- **Detection Model**: Faster R-CNN with CUDA acceleration

## 📐 Door Area Configuration Best Practices

### 1. **Optimal Door Size Ratios**

| Door Type | Frame Width % | Typical Pixel Size (640x480) |
|-----------|---------------|-------------------------------|
| Standard Door (0.9m) | 15-20% | 96-128 pixels wide |
| Wide Door (1.2m) | 20-25% | 128-160 pixels wide |
| Double Door (1.8m) | 25-35% | 160-224 pixels wide |

### 2. **Camera Position Guidelines**

#### **Overhead/Ceiling Mount (Recommended)**
```
     [Camera]
        ↓
   ╔═══════════╗
   ║   Door    ║  ← Draw door area here
   ║   Area    ║
   ╚═══════════╝
```
- **Pros**: Best accuracy, minimal occlusion
- **Door Height**: 25-30% of frame height
- **Inside Direction**: Usually "down"

#### **Side Angle Mount**
```
[Camera] →  ┃Door┃
            ┃Area┃
            ┃    ┃
```
- **Pros**: Natural viewing angle
- **Door Height**: 35-40% of frame height  
- **Inside Direction**: "left" or "right"

#### **Front-Facing Mount**
```
        [Camera]
           ↓
    ┌─────────────┐
    │    Door     │
    │    Area     │
    └─────────────┘
```
- **Pros**: Easy installation
- **Door Height**: 40-45% of frame height
- **Inside Direction**: Usually "down"

## 🎯 Configuration Methods

### Method 1: API Configuration (Recommended)

```bash
# Check current configuration
curl http://127.0.0.1:5000/api/door-area

# Set new door area
curl -X POST http://127.0.0.1:5000/api/door-area \
  -H "Content-Type: application/json" \
  -d '{
    "x1": 250,
    "y1": 100, 
    "x2": 450,
    "y2": 350,
    "inside_direction": "down"
  }'
```

### Method 2: Interactive Helper Tool

```bash
# Run the interactive configuration helper
python door_config_helper.py
```

### Method 3: Manual Calculation

For a **640x480** frame with **standard door**:
```python
# Center the door area
frame_width, frame_height = 640, 480
door_width = int(frame_width * 0.18)  # 115 pixels
door_height = int(frame_height * 0.35)  # 168 pixels

center_x = frame_width // 2   # 320
center_y = frame_height // 2  # 240

x1 = center_x - door_width // 2   # 263
x2 = center_x + door_width // 2   # 377
y1 = center_y - door_height // 2  # 156
y2 = center_y + door_height // 2  # 324
```

## 🚀 Detection Optimization

### 1. **Environment-Based Parameter Tuning**

| Environment | Score Threshold | IoU Threshold | Tracking Threshold | Use Case |
|-------------|----------------|---------------|-------------------|----------|
| **Office** | 0.75 | 0.35 | 45px | Moderate traffic, good lighting |
| **Retail** | 0.70 | 0.30 | 40px | High traffic, varying lighting |
| **Residential** | 0.80 | 0.40 | 50px | Low traffic, consistent environment |
| **Outdoor** | 0.65 | 0.25 | 60px | Weather conditions, variable lighting |

### 2. **Current System Performance** (From Logs)

Your system is running optimally:
- **FPS**: 5-10 (excellent for people counting)
- **GPU Utilization**: 95%+ (very efficient)
- **Detection Accuracy**: High confidence scores
- **Processing Time**: ~100-200ms per frame

### 3. **Fine-Tuning Recommendations**

#### **For Better Accuracy:**
```python
# In Flask app config or environment
SCORE_THRESHOLD = 0.75  # Current: 0.8 (try lowering slightly)
IOU_THRESHOLD = 0.35    # Current: 0.3 (try increasing slightly)
TRACKING_THRESHOLD = 45 # Current: 50 (try lowering for faster movement)
```

#### **For Better Performance:**
- ✅ Already using CUDA acceleration
- ✅ Already using TensorFloat-32 optimizations
- ✅ Already using non-blocking tensor transfers

## 📊 Real-World Configuration Examples

### Example 1: Office Entrance
```json
{
  "door_area": {"x1": 280, "y1": 120, "x2": 360, "y2": 280},
  "inside_direction": "down",
  "environment": "office"
}
```

### Example 2: Retail Store
```json
{
  "door_area": {"x1": 200, "y1": 100, "x2": 440, "y2": 300},
  "inside_direction": "right", 
  "environment": "retail"
}
```

### Example 3: Residential Door
```json
{
  "door_area": {"x1": 270, "y1": 140, "x2": 370, "y2": 300},
  "inside_direction": "down",
  "environment": "residential"
}
```

## 🔧 Troubleshooting Common Issues

### Issue 1: No Entry/Exit Counts
**Symptoms**: People detected but counts stay at 0
**Solutions**:
1. Check door area placement (should cover door threshold)
2. Verify inside direction matches your room layout
3. Ensure door area isn't too small (minimum 60x80 pixels)

### Issue 2: False Counts
**Symptoms**: Counts increment without people crossing
**Solutions**:
1. Increase score threshold (0.8 → 0.85)
2. Adjust door area to exclude non-door regions
3. Check for objects being detected as people

### Issue 3: Missed Detections
**Symptoms**: People cross but aren't counted
**Solutions**:
1. Lower score threshold (0.8 → 0.7)
2. Increase door area size
3. Check camera angle and lighting

## 📝 Testing Your Configuration

### 1. **Manual Test**
```bash
# Force a counter update to test WebSocket connection
curl -X POST http://127.0.0.1:5000/api/counter/force-update
```

### 2. **Detection Status Check**
```bash
# Check detection status and current counts
curl http://127.0.0.1:5000/api/detection/status
```

### 3. **Live Monitoring**
- Open browser to `http://127.0.0.1:5000`
- Watch the live video feed
- Observe counter updates in real-time
- Check browser console for WebSocket messages

## 🎯 Next Steps

1. **Run the interactive helper**: `python door_config_helper.py`
2. **Test your configuration** with real people walking through
3. **Fine-tune parameters** based on your specific environment
4. **Monitor performance** through the web dashboard

Your system is already well-optimized and running smoothly! The main focus should be on precise door area placement for your specific camera angle and door type.

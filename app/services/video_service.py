"""
Video streaming service for capturing and processing video frames.
This file imports from the modular video package.
"""
from app.services.video.video_service import VideoService
from app.services.video.camera_preview_service import CameraPreviewService

# This file imports and re-exports the video services
# while maintaining backward compatibility for any imports of video_service.py

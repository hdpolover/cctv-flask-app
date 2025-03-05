"""
Main application routes for web interface
"""
from flask import Blueprint, render_template, redirect, url_for, request, Response, current_app
import threading
import logging

from app.services.video_service import VideoService
from app.models.detection_model import DetectionModel
from app.core.firebase_client import fetch_camera_settings, save_camera_settings
from app import socketio

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
main_bp = Blueprint('main', __name__)

# Shared resources
detection_model = None
video_service = None

@main_bp.before_app_first_request
def initialize_services():
    """Initialize shared services before first request."""
    global detection_model, video_service
    
    # Initialize detection model if not already done
    if detection_model is None:
        detection_model = DetectionModel(current_app.config)
        logger.info("Detection model initialized")
    
    # Initialize video service if not already done
    if video_service is None:
        # Get camera settings from config or database
        settings = fetch_camera_settings()
        video_path = settings.get('camera_url', current_app.config['VIDEO_PATH'])
        frame_rate = int(settings.get('frame_rate', current_app.config['FRAME_RATE']))
        
        # Parse resolution
        resolution_str = settings.get('resolution', None)
        if resolution_str and isinstance(resolution_str, str) and ',' in resolution_str:
            width, height = map(int, resolution_str.split(','))
            resolution = (width, height)
        else:
            resolution = current_app.config['RESOLUTION']
        
        # Create video service
        video_service = VideoService(detection_model, socketio, 
                                     video_path, frame_rate, resolution)
        
        # Start capture thread
        video_service.start_capture_thread()
        logger.info("Video service initialized and started")
        
        # Set door area if configured
        if settings and 'door_area' in settings:
            door_area = settings['door_area']
            inside_direction = settings.get('inside_direction', 'right')
            
            try:
                x1 = door_area.get('x1')
                y1 = door_area.get('y1')
                x2 = door_area.get('x2')
                y2 = door_area.get('y2')
                detection_model.set_door_area(x1, y1, x2, y2)
                detection_model.set_inside_direction(inside_direction)
                logger.info(f"Door area set to: {(x1, y1, x2, y2)}, inside: {inside_direction}")
            except Exception as e:
                logger.error(f"Error setting door area: {e}")

@main_bp.route('/')
def index():
    """Landing page route."""
    return render_template('login.html')

@main_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handle login form."""
    # For simplicity, just redirect to dashboard
    # In a real app, you'd validate credentials here
    if request.method == 'POST':
        # Process login attempt
        return redirect(url_for('main.home'))
    return render_template('login.html')

@main_bp.route('/home')
def home():
    """Home page with video feed."""
    global detection_model
    
    # Check if services are initialized
    if detection_model is None:
        initialize_services()
    
    # Get counters
    entries, exits = detection_model.get_entry_exit_count()
    people_in_room = max(0, entries - exits)
    
    return render_template('home.html', 
                          entries=entries, 
                          exits=exits, 
                          people_in_room=people_in_room,
                          door_defined=detection_model.door_defined)

@main_bp.route('/video_feed')
def video_feed():
    """Video streaming route."""
    global video_service
    
    # Check if services are initialized
    if video_service is None:
        initialize_services()
        
    return Response(video_service.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@main_bp.route('/dashboard')
def dashboard():
    """Dashboard with video feed."""
    # Redirect to home since they show the same content
    return redirect(url_for('main.home'))

@main_bp.route('/camera-settings', methods=['GET', 'POST'])
def camera_settings():
    """Camera settings page."""
    global video_service
    
    # Check if services are initialized
    if video_service is None:
        initialize_services()
    
    if request.method == 'POST':
        # Get settings from form
        camera_url = request.form['camera_url']
        frame_rate = int(request.form['frame_rate'])
        resolution_str = request.form['resolution']
        
        # Parse resolution
        if ',' in resolution_str:
            width, height = map(int, resolution_str.split(','))
            resolution = (width, height)
        else:
            resolution = current_app.config['RESOLUTION']

        # Update video service
        if video_service:
            video_service.update_settings(camera_url, frame_rate, resolution)

        # Save settings to database
        save_camera_settings(camera_url=camera_url, 
                             frame_rate=frame_rate, 
                             resolution=resolution_str)
        
        logger.info(f"Camera settings updated: {camera_url}, {frame_rate}, {resolution}")
        return redirect(url_for('main.camera_settings'))

    # Get current settings
    settings = fetch_camera_settings()
    
    # Get door settings
    door_area = None
    inside_direction = 'right'
    
    if detection_model and detection_model.door_defined and detection_model.door_area:
        x1, y1, x2, y2 = detection_model.door_area
        door_area = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        inside_direction = detection_model.inside_direction
    elif settings and 'door_area' in settings:
        door_area = settings['door_area']
        inside_direction = settings.get('inside_direction', 'right')
        
    return render_template('camera-settings.html', 
                          settings=settings,
                          door_area=door_area,
                          inside_direction=inside_direction)

@main_bp.route('/reports')
def reports():
    """Reports page."""
    from app.core.firebase_client import get_people_count_logs
    logs = get_people_count_logs(limit=100)
    return render_template('reports.html', logs=logs)
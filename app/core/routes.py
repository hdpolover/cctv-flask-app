"""
Main application routes for web interface
"""
from flask import Blueprint, render_template, redirect, url_for, request, Response, current_app, session, flash
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

def initialize_services():
    """Initialize shared services before first request."""
    global detection_model, video_service
    
    # Get settings from database
    settings = fetch_camera_settings()
    
    # Initialize detection model if not already done
    if detection_model is None:
        # Create config with settings
        config = current_app.config.copy()
        # Add GPU preference from settings if available
        if settings and 'use_gpu' in settings:
            config['USE_GPU'] = settings['use_gpu']
            
        detection_model = DetectionModel(config)
        logger.info("Detection model initialized")
    
    # Initialize video service if not already done
    if video_service is None:
        # Get camera settings
        video_source = settings.get('video_source', 'camera')
        if video_source == 'demo':
            video_path = 'app/static/videos/demo.mp4'
        else:
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

# Register initialization function to run before first request
@main_bp.before_app_request
def initialize_before_request():
    """Initialize services if not already initialized."""
    if detection_model is None or video_service is None:
        initialize_services()

@main_bp.route('/')
def index():
    """Landing page route."""
    return render_template('login.html')

@main_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'admin123':
            # Set session variables to mark user as logged in
            session['logged_in'] = True
            return redirect(url_for('main.home'))
        else:
            flash('Invalid username or password')
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
        # Get video source setting
        video_source = request.form.get('video_source', 'camera')
        
        if video_source == 'demo':
            # Use demo video with default settings
            video_path = 'app/static/videos/demo.mp4'
            # Use default or existing settings for frame rate and resolution
            settings = fetch_camera_settings()
            frame_rate = int(settings.get('frame_rate', 30))
            resolution_str = settings.get('resolution', '640,480')
        else:
            # Use camera settings from form
            camera_url = request.form.get('camera_url', '0')
            frame_rate = int(request.form.get('frame_rate', 30))
            resolution_str = request.form.get('resolution', '640,480')
            video_path = camera_url
        
        # Parse resolution
        if ',' in resolution_str:
            width, height = map(int, resolution_str.split(','))
            resolution = (width, height)
        else:
            resolution = current_app.config['RESOLUTION']

        # Update video service
        if video_service:
            video_service.update_settings(video_path, frame_rate, resolution)        # Prepare settings to save
        save_data = {
            'video_source': video_source,
            'frame_rate': frame_rate,
            'resolution': resolution_str
        }
        
        # Add camera-specific settings
        if video_source == 'camera':
            save_data['camera_url'] = camera_url
        else:  # demo mode
            save_data['camera_url'] = video_path  # Save the demo video path
            
        # Save all settings
        save_camera_settings(**save_data)
        
        logger.info(f"Saved camera settings: {save_data}")
        
        logger.info(f"Camera settings updated: source={video_source}, path={video_path}")
        return redirect(url_for('main.camera_settings'))    # Get current settings
    settings = fetch_camera_settings()
    if settings is None:
        settings = {}
    
    # Ensure video_source is set in settings
    if 'video_source' not in settings:
        settings['video_source'] = 'camera'  # default value
    
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
                          inside_direction=inside_direction,
                          cuda_available=detection_model.cuda_available)

@main_bp.route('/reports')
def reports():
    """Reports page."""
    from app.core.firebase_client import get_people_count_logs
    logs = get_people_count_logs(limit=100)
    return render_template('reports.html', logs=logs)

@main_bp.route('/toggle-processing-device', methods=['POST'])
def toggle_processing_device():
    """Toggle between CPU and GPU processing."""
    global detection_model
    
    if not detection_model:
        return {'success': False, 'error': 'Detection model not initialized'}
    
    # Get the use_gpu value from the request
    use_gpu = request.json.get('use_gpu', True)
    
    # Update the model's processing device
    result = detection_model.set_processing_device(use_gpu)
    
    # Save the setting to the database
    save_camera_settings(use_gpu=use_gpu)
    
    # Return device information
    return result
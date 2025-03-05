"""
API routes for JSON/AJAX endpoints
"""
from flask import Blueprint, jsonify, request, current_app
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Create API blueprint
api_bp = Blueprint('api', __name__)

@api_bp.route('/door-area', methods=['GET'])
def get_door_area():
    """Get the current door area configuration."""
    from app.core.routes import detection_model
    
    if detection_model and detection_model.door_defined and detection_model.door_area:
        x1, y1, x2, y2 = detection_model.door_area
        return jsonify({
            'door_defined': True,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'inside_direction': detection_model.inside_direction
        })
    return jsonify({'door_defined': False})

@api_bp.route('/door-area', methods=['POST'])
def set_door_area():
    """Set the door area coordinates."""
    from app.core.routes import detection_model
    from app.core.firebase_client import save_camera_settings
    
    try:
        data = request.json
        x1 = int(data.get('x1'))
        y1 = int(data.get('y1'))
        x2 = int(data.get('x2'))
        y2 = int(data.get('y2'))
        inside_dir = data.get('inside_direction', 'right')
        
        if detection_model:
            detection_model.set_door_area(x1, y1, x2, y2)
            detection_model.set_inside_direction(inside_dir)
            
            # Save door settings to Firebase
            door_area = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            save_camera_settings(door_area=door_area, inside_direction=inside_dir)
            
            logger.info(f"Door area set via API: {door_area}, inside: {inside_dir}")
            return jsonify({'success': True, 'message': 'Door area set successfully'})
        else:
            logger.error("Detection model not initialized")
            return jsonify({'success': False, 'message': 'Detection model not initialized'}), 500
    except Exception as e:
        logger.exception(f"Error setting door area: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 400

@api_bp.route('/counter', methods=['GET'])
def get_counter():
    """Get current people counting data."""
    from app.core.routes import detection_model
    
    if detection_model:
        entries, exits = detection_model.get_entry_exit_count()
        people_in_room = max(0, entries - exits)
        
        return jsonify({
            'entries': entries,
            'exits': exits,
            'people_in_room': people_in_room,
            'door_defined': detection_model.door_defined,
            'timestamp': datetime.now().isoformat()
        })
    return jsonify({'error': 'Detection model not initialized'}), 500

@api_bp.route('/counter/reset', methods=['POST'])
def reset_counter():
    """Reset people counting data."""
    from app.core.routes import detection_model
    
    if detection_model:
        detection_model.reset_counters()
        logger.info("People counters reset via API")
        return jsonify({'success': True, 'message': 'Counters reset successfully'})
    return jsonify({'error': 'Detection model not initialized'}), 500

@api_bp.route('/settings', methods=['GET'])
def get_settings():
    """Get current camera and detection settings."""
    from app.core.firebase_client import fetch_camera_settings
    from app.core.routes import video_service, detection_model
    
    # Get settings from database
    settings = fetch_camera_settings()
    
    # Add model settings
    if detection_model:
        settings.update({
            'score_threshold': detection_model.score_threshold,
            'iou_threshold': detection_model.iou_threshold,
            'tracking_threshold': detection_model.tracking_threshold
        })
    
    # Add door settings if defined
    if detection_model and detection_model.door_defined and detection_model.door_area:
        x1, y1, x2, y2 = detection_model.door_area
        settings.update({
            'door_area': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
            'inside_direction': detection_model.inside_direction,
            'door_defined': True
        })
    
    return jsonify(settings)

@api_bp.route('/logs', methods=['GET'])
def get_logs():
    """Get people counting logs."""
    from app.core.firebase_client import get_people_count_logs
    
    # Get query parameters
    limit = request.args.get('limit', 50, type=int)
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Convert date strings to datetime if provided
    start_datetime = None
    end_datetime = None
    
    if start_date:
        try:
            start_datetime = datetime.fromisoformat(start_date)
        except ValueError:
            logger.warning(f"Invalid start date format: {start_date}")
    
    if end_date:
        try:
            end_datetime = datetime.fromisoformat(end_date)
        except ValueError:
            logger.warning(f"Invalid end date format: {end_date}")
    
    # Get logs
    logs = get_people_count_logs(start_date=start_datetime, 
                               end_date=end_datetime, 
                               limit=limit)
    
    return jsonify(logs)
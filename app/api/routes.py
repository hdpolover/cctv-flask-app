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
            return jsonify({'success': True, 'message': 'Area pintu berhasil dikonfigurasi'})
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

# New API routes for enhanced Firebase functionality

@api_bp.route('/alerts', methods=['GET'])
def get_alerts():
    """Get system alerts with filtering options."""
    from app.core.firebase_client import get_alerts
    
    alert_type = request.args.get('type')
    severity = request.args.get('severity')
    acknowledged = request.args.get('acknowledged')
    limit = request.args.get('limit', 50, type=int)
    
    # Convert acknowledged string to boolean if provided
    if acknowledged is not None:
        acknowledged = acknowledged.lower() == 'true'
    
    alerts = get_alerts(
        alert_type=alert_type,
        severity=severity,
        acknowledged=acknowledged,
        limit=limit
    )
    
    return jsonify(alerts)

@api_bp.route('/alerts', methods=['POST'])
def create_alert():
    """Create a new alert."""
    from app.core.firebase_client import save_alert
    
    try:
        data = request.json
        alert_type = data.get('type')
        message = data.get('message')
        severity = data.get('severity', 'info')
        metadata = data.get('metadata')
        
        if not alert_type or not message:
            return jsonify({'success': False, 'message': 'Type and message are required'}), 400
        
        alert_id = save_alert(alert_type, message, severity, metadata)
        
        return jsonify({'success': True, 'alert_id': alert_id})
    except Exception as e:
        logger.exception(f"Error creating alert: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 400

@api_bp.route('/alerts/<alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    """Acknowledge an alert."""
    from app.core.firebase_client import acknowledge_alert
    
    success = acknowledge_alert(alert_id)
    
    if success:
        return jsonify({'success': True, 'message': 'Alert acknowledged'})
    else:
        return jsonify({'success': False, 'message': 'Failed to acknowledge alert'}), 400

@api_bp.route('/system-health', methods=['GET'])
def get_system_health():
    """Get system health logs."""
    from app.core.firebase_client import get_system_health_logs
    
    hours = request.args.get('hours', 24, type=int)
    limit = request.args.get('limit', 100, type=int)
    
    logs = get_system_health_logs(hours=hours, limit=limit)
    
    return jsonify(logs)

@api_bp.route('/system-health', methods=['POST'])
def log_system_health():
    """Log current system health metrics."""
    from app.core.firebase_client import log_system_health
    import psutil  # Make sure to install this package
    
    try:
        # Get system metrics
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        
        # Get optional data from request
        data = request.json or {}
        temperature = data.get('temperature')
        fps = data.get('fps')
        
        # Log health data
        log_id = log_system_health(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            temperature=temperature,
            fps=fps
        )
        
        return jsonify({
            'success': True, 
            'log_id': log_id,
            'metrics': {
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'disk_usage': disk_usage,
                'temperature': temperature,
                'fps': fps
            }
        })
    except Exception as e:
        logger.exception(f"Error logging system health: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 400
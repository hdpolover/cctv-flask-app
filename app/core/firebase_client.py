"""
Firebase client service for interacting with Firestore database
"""
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from flask import current_app
import os

# Global database client connection
db = None

def init_firebase():
    """Initialize Firebase connection and return the client.
    
    Returns:
        Firestore client instance
    """
    global db
    if not firebase_admin._apps:
        # Try several locations for the credentials file
        possible_paths = [
            current_app.config.get('FIREBASE_CREDS_PATH') if current_app else None,
            os.getenv('FIREBASE_CREDS_PATH'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'cctv-app-flask-firebase-adminsdk-xdxtx-8e5ea88cd9.json')
        ]
        
        cred_path = None
        for path in possible_paths:
            if path and os.path.exists(path):
                cred_path = path
                break
        
        if cred_path:
            print(f"Using Firebase credentials from: {cred_path}")
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        else:
            print("WARNING: Firebase credentials file not found. Some features will be disabled.")
            # Initialize Firebase with a dummy configuration to prevent errors
            firebase_admin.initialize_app()
    
    try:
        db = firestore.client()
    except Exception as e:
        print(f"Error connecting to Firestore: {e}")
        db = None
    
    return db

def get_db():
    """Get the database client instance.
    
    Returns:
        Firestore client instance
    """
    global db
    if db is None:
        db = init_firebase()
    return db

# Generic CRUD operations

def create_document(collection_name, data, document_id=None):
    """Create a new document in a collection.
    
    Args:
        collection_name: Name of the collection
        data: Dictionary of data to store
        document_id: Optional ID for the document (auto-generated if None)
    
    Returns:
        Tuple of (success, document_id or error message)
    """
    try:
        db_client = get_db()
        if not db_client:
            return False, "Database connection not available"
        
        # Add timestamp
        data['created_at'] = firestore.SERVER_TIMESTAMP
        
        if document_id:
            # Use provided ID
            doc_ref = db_client.collection(collection_name).document(document_id)
            doc_ref.set(data)
        else:
            # Auto-generate ID
            doc_ref = db_client.collection(collection_name).document()
            doc_ref.set(data)
            document_id = doc_ref.id
            
        return True, document_id
    except Exception as e:
        return False, f"Error creating document: {str(e)}"

def read_document(collection_name, document_id):
    """Read a document from a collection.
    
    Args:
        collection_name: Name of the collection
        document_id: ID of the document to read
    
    Returns:
        Document data or None if not found
    """
    try:
        db_client = get_db()
        if not db_client:
            return None
        
        doc_ref = db_client.collection(collection_name).document(document_id)
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            data['id'] = doc.id
            return data
        else:
            return None
    except Exception as e:
        print(f"Error reading document: {str(e)}")
        return None

def update_document(collection_name, document_id, data):
    """Update a document in a collection.
    
    Args:
        collection_name: Name of the collection
        document_id: ID of the document to update
        data: Dictionary of data to update
    
    Returns:
        Tuple of (success, document_id or error message)
    """
    try:
        db_client = get_db()
        if not db_client:
            return False, "Database connection not available"
        
        # Add updated timestamp
        data['updated_at'] = firestore.SERVER_TIMESTAMP
        
        doc_ref = db_client.collection(collection_name).document(document_id)
        doc_ref.update(data)
        
        return True, document_id
    except Exception as e:
        return False, f"Error updating document: {str(e)}"

def delete_document(collection_name, document_id):
    """Delete a document from a collection.
    
    Args:
        collection_name: Name of the collection
        document_id: ID of the document to delete
    
    Returns:
        Boolean indicating success
    """
    try:
        db_client = get_db()
        if not db_client:
            return False
        
        db_client.collection(collection_name).document(document_id).delete()
        return True
    except Exception as e:
        print(f"Error deleting document: {str(e)}")
        return False

def query_collection(collection_name, filters=None, order_by=None, order_direction='DESCENDING', limit=50):
    """Query documents in a collection with filters.
    
    Args:
        collection_name: Name of the collection
        filters: List of tuples (field, operator, value)
        order_by: Field to order by
        order_direction: 'ASCENDING' or 'DESCENDING'
        limit: Maximum number of documents to return
    
    Returns:
        List of documents
    """
    try:
        db_client = get_db()
        if not db_client:
            return []
        
        query = db_client.collection(collection_name)
        
        # Apply filters
        if filters:
            for field, operator, value in filters:
                query = query.where(field, operator, value)
        
        # Apply ordering
        if order_by:
            direction = firestore.Query.DESCENDING if order_direction == 'DESCENDING' else firestore.Query.ASCENDING
            query = query.order_by(order_by, direction=direction)
        
        # Apply limit
        query = query.limit(limit)
        
        # Execute query
        results = []
        for doc in query.stream():
            data = doc.to_dict()
            data['id'] = doc.id
            results.append(data)
        
        return results
    except Exception as e:
        print(f"Error querying collection: {str(e)}")
        return []

# Specific operations for camera settings

def save_camera_settings(camera_url=None, frame_rate=None, resolution=None, 
                         door_area=None, inside_direction=None, video_source=None, use_gpu=None):
    """Save camera settings to Firestore.
    
    Args:
        camera_url: The URL or index of the camera
        frame_rate: The frame rate for video capture
        resolution: The resolution as tuple (width, height)
        door_area: The door area as dict with x1, y1, x2, y2 keys
        inside_direction: Which side of the door is 'inside' ("left", "right", "up", or "down")
        video_source: Source of video ("camera" or "demo")
        use_gpu: Whether to use GPU acceleration if available
    """
    db_client = get_db()
    settings_ref = db_client.collection("camera_settings").document("settings")
    
    # Get existing settings to update
    current_settings = settings_ref.get().to_dict() or {}
    
    # Update only provided values
    if camera_url is not None:
        current_settings["camera_url"] = camera_url
    if frame_rate is not None:
        current_settings["frame_rate"] = frame_rate
    if resolution is not None:
        current_settings["resolution"] = resolution
    if door_area is not None:
        current_settings["door_area"] = door_area
    if inside_direction is not None:
        current_settings["inside_direction"] = inside_direction
    if video_source is not None:
        current_settings["video_source"] = video_source
    if use_gpu is not None:
        current_settings["use_gpu"] = use_gpu

    # Always update timestamp
    current_settings["last_updated"] = firestore.SERVER_TIMESTAMP
    
    # Save to Firestore
    settings_ref.set(current_settings)
    
    return current_settings

def fetch_camera_settings():
    """Fetch camera settings from Firestore.
    
    Returns:
        Dictionary containing camera settings
    """
    db_client = get_db()
    settings_ref = db_client.collection("camera_settings").document("settings")
    settings = settings_ref.get()
    
    if settings.exists:
        return settings.to_dict()
    else:
        # Return default settings
        return {
            "camera_url": "0",
            "frame_rate": 30,
            "resolution": "640,480"
        }

# Operations for people counting logs

def save_people_count_log(entries, exits, people_in_room):
    """Save a log entry for people counting.
    
    Args:
        entries: Number of people who entered
        exits: Number of people who exited
        people_in_room: Current count of people in the room
    """
    db_client = get_db()
    log_ref = db_client.collection("counting_logs").document()
    
    log_ref.set({
        "entries": entries,
        "exits": exits,
        "people_in_room": people_in_room,
        "timestamp": firestore.SERVER_TIMESTAMP
    })
    
def get_people_count_logs(start_date=None, end_date=None, limit=50):
    """Get people counting logs within a date range.
    
    Args:
        start_date: Start date for filtering logs
        end_date: End date for filtering logs
        limit: Maximum number of logs to return
        
    Returns:
        List of log entries
    """
    db_client = get_db()
    query = db_client.collection("counting_logs")
    
    if start_date:
        query = query.where("timestamp", ">=", start_date)
    if end_date:
        query = query.where("timestamp", "<=", end_date)
        
    query = query.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit)
    
    results = []
    for doc in query.stream():
        entry = doc.to_dict()
        entry["id"] = doc.id
        results.append(entry)
        
    return results

# Alert management functions

def save_alert(alert_type, message, severity="info", metadata=None):
    """Save an alert to Firestore.
    
    Args:
        alert_type: Type of alert (e.g., "security", "system", "crowd")
        message: Alert message
        severity: Alert severity ("info", "warning", "critical")
        metadata: Additional metadata about the alert
        
    Returns:
        Alert document ID
    """
    db_client = get_db()
    alert_ref = db_client.collection("alerts").document()
    
    alert_data = {
        "type": alert_type,
        "message": message,
        "severity": severity,
        "timestamp": firestore.SERVER_TIMESTAMP,
        "acknowledged": False
    }
    
    if metadata:
        alert_data["metadata"] = metadata
    
    alert_ref.set(alert_data)
    return alert_ref.id

def get_alerts(alert_type=None, severity=None, acknowledged=None, limit=50):
    """Get alerts filtered by type, severity, and acknowledgment status.
    
    Args:
        alert_type: Filter by alert type
        severity: Filter by severity
        acknowledged: Filter by acknowledgment status
        limit: Maximum number of alerts to return
        
    Returns:
        List of alert documents
    """
    filters = []
    
    if alert_type:
        filters.append(("type", "==", alert_type))
    
    if severity:
        filters.append(("severity", "==", severity))
    
    if acknowledged is not None:
        filters.append(("acknowledged", "==", acknowledged))
    
    return query_collection("alerts", filters=filters, order_by="timestamp", limit=limit)

def acknowledge_alert(alert_id):
    """Mark an alert as acknowledged.
    
    Args:
        alert_id: ID of the alert to acknowledge
        
    Returns:
        Boolean indicating success
    """
    return update_document("alerts", alert_id, {"acknowledged": True})[0]

# System health monitoring

def log_system_health(cpu_usage, memory_usage, disk_usage, temperature=None, fps=None):
    """Log system health metrics.
    
    Args:
        cpu_usage: CPU usage percentage
        memory_usage: Memory usage percentage
        disk_usage: Disk usage percentage
        temperature: CPU temperature (optional)
        fps: Current processing FPS (optional)
        
    Returns:
        Document ID of the created log
    """
    db_client = get_db()
    health_ref = db_client.collection("system_health").document()
    
    health_data = {
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "disk_usage": disk_usage,
        "timestamp": firestore.SERVER_TIMESTAMP
    }
    
    if temperature is not None:
        health_data["temperature"] = temperature
    
    if fps is not None:
        health_data["fps"] = fps
    
    health_ref.set(health_data)
    return health_ref.id

def get_system_health_logs(hours=24, limit=100):
    """Get system health logs for the past number of hours.
    
    Args:
        hours: Number of hours to look back
        limit: Maximum number of logs to return
        
    Returns:
        List of system health logs
    """
    # Calculate timestamp for 'hours' ago
    start_time = datetime.now().timestamp() - (hours * 3600)
    
    filters = [
        ("timestamp", ">=", start_time)
    ]
    
    return query_collection("system_health", filters=filters, order_by="timestamp", limit=limit)
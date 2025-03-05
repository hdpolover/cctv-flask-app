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
        cred_path = current_app.config.get('FIREBASE_CREDS_PATH') if current_app else os.getenv('FIREBASE_CREDS_PATH')
        if cred_path and os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        else:
            raise FileNotFoundError(f"Firebase credentials file not found at: {cred_path}")
    
    db = firestore.client()
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

def save_camera_settings(camera_url=None, frame_rate=None, resolution=None, 
                         door_area=None, inside_direction=None):
    """Save camera settings to Firestore.
    
    Args:
        camera_url: The URL or index of the camera
        frame_rate: The frame rate for video capture
        resolution: The resolution as tuple (width, height)
        door_area: The door area as dict with x1, y1, x2, y2 keys
        inside_direction: Which side of the door is 'inside' ("left" or "right")
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
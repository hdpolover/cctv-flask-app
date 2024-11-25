import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

def initialize_firebase():
    """Initialize Firebase."""
    if not firebase_admin._apps:
        cred = credentials.Certificate('cctv-app-flask-firebase-adminsdk-xdxtx-8e5ea88cd9.json')
        firebase_admin.initialize_app(cred)
    return firestore.client()

def save_camera_settings(db, camera_url, frame_rate, resolution):
    """Save camera settings to Firestore."""
    settings_ref = db.collection("camera_settings").document("settings")
    settings_ref.set({
        "camera_url": camera_url,
        "frame_rate": frame_rate,
        "resolution": resolution,
        "last_updated": firestore.SERVER_TIMESTAMP
    })

def fetch_camera_settings(db):
    """Fetch camera settings from Firestore."""
    settings_ref = db.collection("camera_settings").document("settings")
    settings = settings_ref.get()
    if settings.exists:
        return settings.to_dict()
    else:
        return {
            "camera_url": "",
            "frame_rate": 30,
            "resolution": "640,480"
        }
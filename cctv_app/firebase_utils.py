import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK
def initialize_firebase():
    """Initialize the Firebase Admin SDK."""
    cred = credentials.Certificate(
        'cctv_app/cctv-app-flask-firebase-adminsdk-xdxtx-8e5ea88cd9.json'
        )  # Path to your Firebase service account key
    firebase_admin.initialize_app(cred)
    return firestore.client()

# Firestore reference to store people count data
def save_to_firestore(db, people_in_room, people_out_room):
    """Save the people count to Firestore."""
    people_ref = db.collection('people_counter')
    people_ref.document('current_count').set({
        'people_in_room': people_in_room,
        'people_out_room': people_out_room,
        'timestamp': firestore.SERVER_TIMESTAMP
    })
    
# save camera setting to firebase with data like: camera_url, door_line, frame_rate, resolution, etc.
def save_camera_settings(db, camera_url, door_line, frame_rate, resolution):
    """Save the camera settings to Firestore."""
    settings_ref = db.collection('camera_settings')
    settings_ref.document('current_settings').set({
        'camera_url': camera_url,
        'door_line': door_line,
        'frame_rate': frame_rate,
        'resolution': resolution,
        'timestamp': firestore.SERVER_TIMESTAMP
    })

# fetch camera setting from firebase
def fetch_camera_settings(db):
    """Fetch the camera settings from Firestore."""
    settings_ref = db.collection('camera_settings').document('current_settings').get()
    return settings_ref.to_dict()

"""
Firebase Tester - Verify and enhance Firebase operations
"""
import os
import sys
import datetime

# Add parent directory to path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.core.firebase_client import (
    init_firebase, 
    get_db, 
    save_camera_settings, 
    fetch_camera_settings,
    save_people_count_log,
    get_people_count_logs
)

def test_firebase_connection():
    """Test Firebase connection and initialize the client."""
    print("Testing Firebase connection...")
    db = init_firebase()
    if db:
        print("✅ Firebase connection successful!")
    else:
        print("❌ Firebase connection failed!")
    return db

def test_camera_settings():
    """Test camera settings CRUD operations."""
    print("\nTesting camera settings operations...")
    
    # Create and update camera settings
    print("Saving camera settings...")
    test_settings = {
        'camera_url': '0',
        'frame_rate': 25,
        'resolution': '800,600',
        'video_source': 'camera',
        'use_gpu': False,
        'door_area': {'x1': 100, 'y1': 100, 'x2': 300, 'y2': 300},
        'inside_direction': 'right'
    }
    
    # Save settings
    save_camera_settings(
        camera_url=test_settings['camera_url'],
        frame_rate=test_settings['frame_rate'],
        resolution=test_settings['resolution'],
        door_area=test_settings['door_area'],
        inside_direction=test_settings['inside_direction'],
        video_source=test_settings['video_source'],
        use_gpu=test_settings['use_gpu']
    )
    
    # Fetch settings
    print("Fetching camera settings...")
    settings = fetch_camera_settings()
    
    if settings:
        print("✅ Camera settings fetch successful!")
        print(f"Settings: {settings}")
        
        # Verify settings match what we saved
        success = all([
            settings.get('camera_url') == test_settings['camera_url'],
            settings.get('frame_rate') == test_settings['frame_rate'],
            settings.get('resolution') == test_settings['resolution'],
            settings.get('video_source') == test_settings['video_source'],
            settings.get('use_gpu') == test_settings['use_gpu']
        ])
        
        if success:
            print("✅ Camera settings values match!")
        else:
            print("❌ Camera settings values don't match!")
    else:
        print("❌ Camera settings fetch failed!")

def test_people_count_logs():
    """Test people count logs CRUD operations."""
    print("\nTesting people count logs operations...")
    
    # Create a log entry
    print("Creating people count log...")
    entries = 5
    exits = 3
    people_in_room = entries - exits
    
    save_people_count_log(entries, exits, people_in_room)
    
    # Fetch logs
    print("Fetching people count logs...")
    logs = get_people_count_logs(limit=10)
    
    if logs:
        print(f"✅ People count logs fetch successful! Found {len(logs)} logs.")
        for log in logs[:3]:  # Print first 3 logs
            print(f"Log: {log}")
    else:
        print("❌ People count logs fetch failed or no logs found!")

def create_delete_test_document():
    """Test creating and deleting a document."""
    print("\nTesting create/delete operations...")
    
    db = get_db()
    if not db:
        print("❌ Database not available!")
        return
    
    # Create a test collection
    test_collection = db.collection('test_documents')
    
    # Create a document with auto-generated ID
    print("Creating test document...")
    doc_ref = test_collection.document()
    doc_ref.set({
        'test_field': 'test_value',
        'number': 42,
        'timestamp': datetime.datetime.now(),
        'nested': {
            'field1': 'value1',
            'field2': 'value2'
        }
    })
    
    doc_id = doc_ref.id
    print(f"✅ Document created with ID: {doc_id}")
    
    # Read the document
    print("Reading test document...")
    doc = test_collection.document(doc_id).get()
    if doc.exists:
        print(f"✅ Document read successful: {doc.to_dict()}")
    else:
        print("❌ Document not found!")
        return
    
    # Update the document
    print("Updating test document...")
    test_collection.document(doc_id).update({
        'test_field': 'updated_value',
        'updated_at': datetime.datetime.now(),
        'nested.field1': 'updated_nested_value'
    })
    
    # Read the updated document
    updated_doc = test_collection.document(doc_id).get()
    if updated_doc.exists and updated_doc.to_dict()['test_field'] == 'updated_value':
        print(f"✅ Document update successful: {updated_doc.to_dict()}")
    else:
        print("❌ Document update failed!")
    
    # Delete the document
    print("Deleting test document...")
    test_collection.document(doc_id).delete()
    
    # Verify deletion
    deleted_doc = test_collection.document(doc_id).get()
    if not deleted_doc.exists:
        print(f"✅ Document deletion successful!")
    else:
        print("❌ Document deletion failed!")

def add_firebase_crud_functions():
    """Add additional CRUD functions to firebase_client.py."""
    # This is just a placeholder for the actual code we'll add to firebase_client.py
    pass

if __name__ == "__main__":
    # Run tests
    db = test_firebase_connection()
    if db:
        test_camera_settings()
        test_people_count_logs()
        create_delete_test_document()
    else:
        print("Skipping other tests due to Firebase connection failure.")
    
    print("\nFirebase testing complete!")

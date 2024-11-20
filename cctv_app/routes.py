from flask import render_template, Response, redirect, url_for, request
import cv2
import torch
import threading
from datetime import datetime
import time  # Import the time module
from . import app  # Import the app instance from __init__.py
from .utils import load_model, detect_people, draw_door_line, has_crossed_line
from .firebase_utils import initialize_firebase, save_to_firestore

# Initialize Firebase
db = initialize_firebase()

# Load Faster R-CNN model and set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA available: {torch.cuda.is_available()}")  # Print if CUDA is available
model = load_model()  # Load your Faster R-CNN model
model.to(device)

# Video capture settings
cap = cv2.VideoCapture(0)  # Default webcam, replace with URL or file path if needed
door_line = ((100, 300), (500, 300))  # Coordinates for the door line
frame_rate = 5  # Default frame rate
resolution = (640, 480)  # Default resolution
prev_time = 0
people_in_room = 0
people_out_room = 0
frame_lock = threading.Lock()

@app.route('/')
def login():
    """Render the login page."""
    return render_template('login.html')

@app.route('/home')
def home():
    """Render the home page."""
    global people_in_room
    return render_template('home.html', people_in_room=people_in_room)

@app.route('/dashboard')
def dashboard():
    """Render the dashboard page."""
    return render_template('dashboard.html')

@app.route('/camera-settings', methods=['GET', 'POST'])
def camera_settings():
    """Handle camera settings updates."""
    global cap, door_line, frame_rate, resolution

    if request.method == 'POST':
        camera_url = request.form['camera_url']
        door_line_coords = list(map(int, request.form['door_line'].split(',')))
        door_line = ((door_line_coords[0], door_line_coords[1]), (door_line_coords[2], door_line_coords[3]))
        frame_rate = int(request.form['frame_rate'])
        resolution = tuple(map(int, request.form['resolution'].split(',')))

        cap.release()
        cap = cv2.VideoCapture(camera_url)

        # Save updated settings to Firebase
        save_camera_settings(db, camera_url, door_line, frame_rate, resolution)
        return redirect(url_for('camera_settings'))

    settings = fetch_camera_settings(db)
    return render_template('camera-settings.html', settings=settings)

@app.route('/reports')
def reports():
    """Render the reports page."""
    return render_template('reports.html')

def generate_frames():
    """Video streaming generator function."""
    global people_in_room, people_out_room, prev_time
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        if (current_time - prev_time) < 1.0 / frame_rate:
            continue
        prev_time = current_time
        resized_frame = cv2.resize(frame, resolution)
        people_boxes, _ = detect_people(resized_frame, model, device)
        try:
            draw_door_line(resized_frame, door_line)
        except Exception as e:
            print(f"Error drawing door line: {e}")
            continue
        remaining_boxes = []
        for box in people_boxes:
            # Draw a green rectangle around the detected person
            cv2.rectangle(resized_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            if has_crossed_line(box, door_line, "in"):
                people_in_room += 1
            elif has_crossed_line(box, door_line, "out"):
                people_out_room += 1
            else:
                remaining_boxes.append(box)
        people_boxes = remaining_boxes
        cv2.putText(
            resized_frame,
            f"In: {people_in_room}  Out: {people_out_room}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )
        ret, buffer = cv2.imencode('.jpg', resized_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)  # Add a delay between frames

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
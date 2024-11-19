from flask import render_template, request, redirect, url_for, Response
import cv2
import threading
from . import app, db, model
from .firebase_utils import save_camera_settings, fetch_camera_settings
from .utils import detect_people, draw_door_line, has_crossed_line

# Initialize video capture
settings = fetch_camera_settings(db)
camera_url = settings.get('camera_url', '0')
cap = cv2.VideoCapture(camera_url)
door_line = tuple(settings.get('door_line', ((100, 300), (500, 300))))
frame_rate = settings.get('frame_rate', 5)
resolution = tuple(settings.get('resolution', (640, 480)))

prev_time = 0
people_in_room = 0
people_out_room = 0
frame_lock = threading.Lock()

@app.route('/', methods=['GET', 'POST'])
def login():
    """Render the login page."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'password':
            return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/home')
def home():
    """Render the home page."""
    return render_template('home.html', people_in_room=people_in_room)

@app.route('/dashboard')
def dashboard():
    """Render the dashboard page."""
    return render_template('dashboard.html')

@app.route('/camera-settings', methods=['GET', 'POST'])
def camera_settings():
    """Render the camera settings page."""
    if request.method == 'POST':
        camera_url = request.form['camera_url']
        door_line = tuple(map(int, request.form['door_line'].split(',')))
        frame_rate = int(request.form['frame_rate'])
        resolution = tuple(map(int, request.form['resolution'].split(',')))
        
        global cap
        cap.release()
        cap = cv2.VideoCapture(camera_url)
        
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
        people_boxes = detect_people(resized_frame, model)
        draw_door_line(resized_frame, door_line)
        for box in people_boxes:
            if has_crossed_line(box, door_line):
                people_in_room += 1
                people_boxes.remove(box)
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

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
from flask import render_template, Response, redirect, url_for, request
import threading
from app import app, detection_model
from firebase_utils import initialize_firebase, save_camera_settings, fetch_camera_settings
from video_stream import VideoStream
from flask_socketio import SocketIO

# Initialize Firebase
db = initialize_firebase()

# Initialize SocketIO
socketio = SocketIO(app)

# Video capture settings
video_path = "rtsp://admin:admin123@@192.168.1.52/V_ENC_000"
frame_rate = 30  # Limit max frame rate to 30 FPS
resolution = (640, 480)  # Default resolution

# Initialize VideoStream
video_stream = VideoStream(video_path, frame_rate, resolution)

@app.route('/')
def login():
    """Render the login page."""
    return render_template('login.html')

@app.route('/home')
def home():
    """Render the home page."""
    left_to_right = detection_model.left_to_right
    right_to_left = detection_model.right_to_left
    return render_template('home.html', left_to_right=left_to_right, right_to_left=right_to_left)

@app.route('/dashboard')
def dashboard():
    """Render the dashboard page."""
    return render_template('dashboard.html')

@app.route('/camera-settings', methods=['GET', 'POST'])
def camera_settings():
    """Handle camera settings updates."""
    global video_stream

    if request.method == 'POST':
        camera_url = request.form['camera_url']
        frame_rate = int(request.form['frame_rate'])
        resolution = tuple(map(int, request.form['resolution'].split(',')))

        video_stream.update_settings(camera_url, frame_rate, resolution)

        # Save updated settings to Firebase
        save_camera_settings(db, camera_url, frame_rate, resolution)
        return redirect(url_for('camera_settings'))

    settings = fetch_camera_settings(db)
    return render_template('camera-settings.html', settings=settings)

@app.route('/reports')
def reports():
    """Render the reports page."""
    return render_template('reports.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    threading.Thread(target=video_stream.capture_frames, args=(socketio,)).start()
    socketio.run(app, host='0.0.0.0', port=5000)
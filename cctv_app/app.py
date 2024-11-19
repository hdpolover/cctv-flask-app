from flask import Flask, render_template, Response
import cv2
from utils import load_model, detect_people, draw_door_line, has_crossed_line

app = Flask(__name__)

# Load the Faster R-CNN model
model = load_model()

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with video file path
door_line = ((100, 300), (500, 300))  # Coordinates for the door line
frame_rate = 5  # Frames per second for processing
prev_time = 0

people_in_room = 0
people_out_room = 0

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

def generate_frames():
    """Video streaming generator function."""
    global people_in_room, people_out_room, prev_time

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Limit frame processing rate
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        if (current_time - prev_time) < 1.0 / frame_rate:
            continue
        prev_time = current_time

        # Resize frame for faster processing
        resized_frame = cv2.resize(frame, (640, 480))

        # Detect people in the frame
        people_boxes = detect_people(resized_frame, model)

        # Draw the door line
        draw_door_line(resized_frame, door_line)

        # Check for line crossings
        for box in people_boxes:
            if has_crossed_line(box, door_line):
                people_in_room += 1
                # Remove the person from tracking to prevent double counting
                people_boxes.remove(box)

        # Overlay the count on the frame
        cv2.putText(
            resized_frame,
            f"In: {people_in_room}  Out: {people_out_room}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        # Encode frame to JPEG for streaming
        _, buffer = cv2.imencode('.jpg', resized_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

import cv2
import os

# Path to your video file
video_path = 'cctv_app/static/videos/demo1.mp4'

# Check if the video file exists
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    exit()

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Reached the end of the video.")
        break

    # Display the frame
    cv2.imshow('Video', frame)

    # Press 'q' to exit the video
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
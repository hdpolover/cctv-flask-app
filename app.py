from flask import Flask
from firebase_utils import initialize_firebase
from detection_model import DetectionModel

app = Flask(__name__)

# Initialize Firebase
db = initialize_firebase()

# Load the detection model
detection_model = DetectionModel()

# Import routes
import routes

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
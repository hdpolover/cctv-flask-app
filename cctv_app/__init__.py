from flask import Flask
from .firebase_utils import initialize_firebase
from .utils import load_model

app = Flask(__name__)

# Initialize Firebase
db = initialize_firebase()

# Load the Faster R-CNN model
model = load_model()

# Import routes
from . import routes
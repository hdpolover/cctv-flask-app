"""
CCTV People Detection Flask Application
Main application package initialization
"""
from flask import Flask
from flask_socketio import SocketIO
import os

# Initialize SocketIO instance at module level
socketio = SocketIO()

def create_app(config_name='default'):
    """Application factory function to create and configure the Flask app.
    
    Args:
        config_name: The configuration profile to use (default, development, production, etc.)
    
    Returns:
        Configured Flask application
    """
    # Create the Flask application instance
    app = Flask(__name__)
    
    # Load configuration
    from config.settings import config
    app.config.from_object(config[config_name])
    
    # Initialize components with the app
    from app.core.firebase_client import init_firebase
    init_firebase()
    
    # Register blueprints
    from app.core.routes import main_bp
    app.register_blueprint(main_bp)
    
    from app.api.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Initialize SocketIO with the app
    socketio.init_app(app)
    
    # Create required directories if they don't exist
    os.makedirs(app.config['LOG_DIR'], exist_ok=True)
    
    return app
"""
CCTV People Detection Flask Application
Main application package initialization
"""
from flask import Flask
from flask_socketio import SocketIO
import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
    
    # Configure logging
    setup_logging(app)
    
    return app

def setup_logging(app):
    """Set up logging for the application."""
    log_file = os.path.join(app.config['LOG_DIR'], 'cctv_app.log')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Create file handler for logging to a file
    file_handler = RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
      # Create console handler for logging to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log application startup
    app.logger.info(f"Application starting")
    app.logger.info(f"Log file created at: {log_file}")
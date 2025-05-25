"""
Main application entry point
"""
import os
import sys
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
def setup_logging(app):
    """Configure logging for the application."""
    # Make sure log directory exists
    os.makedirs(app.config['LOG_DIR'], exist_ok=True)
    
    # Set log level based on environment
    log_level = logging.DEBUG if app.debug else logging.INFO
    
    # Configure file handler for application logs
    file_handler = RotatingFileHandler(
        os.path.join(app.config['LOG_DIR'], 'app.log'),
        maxBytes=1024 * 1024 * 10,  # 10 MB
        backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    ))
    file_handler.setLevel(log_level)
    
    # Configure console handler for stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    ))
    console_handler.setLevel(log_level)
    
    # Add handlers to root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
      # Configure Flask logger
    app.logger.handlers = []
    app.logger.propagate = True
    
    # Log startup message
    app.logger.info(f"Starting application")

# Create and configure app
def create_app():
    """Create and configure the Flask application."""
    from app import create_app as factory_create_app
    
    # Get environment from environment variable or use development
    config_name = os.getenv('FLASK_CONFIG', 'development')
    app = factory_create_app(config_name)
    
    # Set up logging
    setup_logging(app)
    
    return app

if __name__ == '__main__':
    # Create app
    app = create_app()
    
    # Run with SocketIO
    from app import socketio
    
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = app.config.get('DEBUG', False)
    
    app.logger.info(f"Running server on {host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
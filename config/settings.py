"""
Application configuration settings
"""
import os

# Base directory of the application
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

class Config:
    """Base configuration class with common settings"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    FIREBASE_CREDS_PATH = os.path.join(BASE_DIR, 'cctv-app-flask-firebase-adminsdk-xdxtx-8e5ea88cd9.json')
    
    # Video capture settings - defaults
    VIDEO_PATH = 0  # Default camera
    FRAME_RATE = 30
    RESOLUTION = (640, 480)
    
    # Detection model settings
    SCORE_THRESHOLD = 0.8
    IOU_THRESHOLD = 0.3
    TRACKING_THRESHOLD = 50
    
    @staticmethod
    def init_app(app):
        """Initialize application with this configuration"""
        pass


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    # In production, use a more secure randomly generated key
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(24)


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    # Use in-memory database for testing
    VIDEO_PATH = os.path.join(BASE_DIR, 'app/static/videos/demo1.mp4')


# Configuration dictionary to map config names to objects
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
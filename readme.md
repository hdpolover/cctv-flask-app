# CCTV People Counter Application

A Flask-based application for monitoring CCTV camera feeds and tracking people entering and exiting rooms using computer vision. The application uses pre-trained object detection models and provides a real-time web interface for monitoring and configuration.

## Features

- Real-time people detection and counting using PyTorch and Faster R-CNN
- Configurable door area for tracking entries and exits
- Live video streaming with detection overlays
- Camera settings management through web interface
- Historical data tracking with Firebase integration
- GPU acceleration support via CUDA
- Responsive web interface with real-time updates

## Project Structure

```
cctv_flask_app/
├── app/                    # Main application package
│   ├── api/               # API endpoints
│   ├── core/              # Core application functionality
│   ├── models/            # ML models and detection logic
│   ├── services/          # Business services
│   ├── static/            # Static assets (CSS, JS, etc.)
│   ├── templates/         # HTML templates
│   └── utils/            # Utility functions
├── config/                # Configuration settings
├── logs/                  # Application logs
└── run.py                # Application entry point
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- Webcam or RTSP camera stream
- Firebase account (for data persistence)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cctv_flask_app
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Set up Firebase:
- Create a new Firebase project
- Download service account key JSON file
- Save as `cctv-app-flask-firebase-adminsdk.json` in project root

## Configuration

### Environment Variables

Key environment variables in `.env`:

- `FLASK_APP`: Application entry point (default: run.py)
- `FLASK_ENV`: Environment mode (development/production)
- `VIDEO_PATH`: Camera source (0 for webcam, or RTSP URL)
- `FRAME_RATE`: Video processing frame rate
- `RESOLUTION`: Video resolution (format: width,height)

### Detection Settings

Configurable in `config/settings.py`:

- `SCORE_THRESHOLD`: Confidence threshold for detections
- `IOU_THRESHOLD`: Intersection-over-Union threshold
- `TRACKING_THRESHOLD`: Pixel threshold for movement tracking

## Running the Application

Development mode:
```bash
flask run --debug
```

Production mode:
```bash
flask run --host=0.0.0.0
```

Access the application at `http://localhost:5000`

## Usage

1. **Login**: Access the application through the login page
2. **Camera Setup**:
   - Navigate to Camera Settings
   - Configure camera source and parameters
   - Draw door area for entry/exit tracking

3. **Monitoring**:
   - View live feed on Home page
   - Monitor entries/exits in real-time
   - Check historical data in Reports

4. **Door Configuration**:
   - Use mouse to draw door area on video feed
   - Set inside/outside directions
   - Save configuration for persistent tracking

## Development

### Adding New Features

1. Follow the modular structure in the `app` package
2. Add routes to appropriate blueprints
3. Include tests for new functionality
4. Update documentation

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Include docstrings for functions and classes
- Add logging for important operations

## Troubleshooting

### Common Issues

1. **Video Feed Issues**:
   - Check camera connection
   - Verify VIDEO_PATH in .env
   - Ensure OpenCV installation is complete

2. **Detection Performance**:
   - Lower resolution for better performance
   - Adjust FRAME_RATE in settings
   - Enable GPU support if available

3. **Firebase Connection**:
   - Verify credentials file exists
   - Check Firebase project settings
   - Ensure proper permissions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch for the pre-trained models
- Flask for the web framework
- Firebase for data persistence

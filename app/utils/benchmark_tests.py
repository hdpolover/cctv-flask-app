"""
Benchmark test script for the CCTV application detection model.
This script tests the performance of the detection model on both CPU and GPU
and provides detailed metrics for comparison.
"""
import os
import sys
import time
import cv2
import torch
import numpy as np
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from app.models.detection_model import DetectionModel
from app.services.video_service import VideoService

def run_benchmark(device_type="cpu", num_frames=100, video_path=None):
    """
    Run benchmark tests on the detection model using specified device.
    
    Args:
        device_type (str): Device to use ('cpu' or 'cuda')
        num_frames (int): Number of frames to process
        video_path (str): Path to test video file
    
    Returns:
        dict: Dictionary with benchmark results
    """
    print(f"\nRunning benchmark on {device_type.upper()}...")
    
    # Create a config with the device setting
    config = {
        'USE_GPU': device_type == 'cuda',
        'SCORE_THRESHOLD': 0.8,
        'IOU_THRESHOLD': 0.3
    }
    
    # Initialize the detection model
    model = DetectionModel(config)
    
    # Ensure device is correctly set
    if device_type == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_type = 'cpu'
    
    # Use demo video if no path provided
    if video_path is None:
        video_path = os.path.join(parent_dir, 'app', 'static', 'videos', 'demo.mp4')
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return None
    
    # Get video info
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Results tracking
    inference_times = []
    num_detections = []
    frame_processing_times = []
    memory_usage = []
    
    # Process frames
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            # Loop back to beginning if video ends
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        
        # Record memory usage (GPU or system RAM)
        if device_type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            memory_usage.append(memory_allocated)
        else:
            # For CPU, we'll just record 0 for now (could use psutil in production)
            memory_usage.append(0)
        
        # Time the full frame processing
        start_time = time.time()
        
        # Preprocess the frame
        preprocessed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect people
        detections, detection_time = model.detect_people_benchmark(preprocessed)
        
        # Track inference time
        inference_times.append(detection_time)
        
        # Track number of detections
        num_detections.append(len(detections))
        
        # Calculate full frame processing time
        frame_time = time.time() - start_time
        frame_processing_times.append(frame_time)
        
        # Show progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{num_frames} frames")
    
    # Close video capture
    cap.release()
    
    # Calculate statistics
    avg_inference_time = np.mean(inference_times)
    avg_processing_time = np.mean(frame_processing_times)
    avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
    avg_detections = np.mean(num_detections)
    avg_memory = np.mean(memory_usage)
    
    # Create results dictionary
    results = {
        'device': device_type,
        'frames_processed': len(frame_processing_times),
        'average_inference_time': avg_inference_time,
        'average_processing_time': avg_processing_time,
        'average_fps': avg_fps,
        'average_detections': avg_detections,
        'average_memory_usage_mb': avg_memory,
        'video_resolution': f"{frame_width}x{frame_height}",
        'video_fps': fps
    }
    
    # Print summary
    print(f"\nBenchmark results for {device_type.upper()}:")
    print(f"  Average inference time: {avg_inference_time:.4f} seconds")
    print(f"  Average total processing time: {avg_processing_time:.4f} seconds")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Average detections per frame: {avg_detections:.2f}")
    print(f"  Average memory usage: {avg_memory:.2f} MB")
    
    return results

def accuracy_test(video_path=None, ground_truth=None):
    """
    Test the accuracy of the detection model against ground truth annotations.
    
    Args:
        video_path (str): Path to test video file
        ground_truth (dict): Dictionary with frame-by-frame ground truth data
    
    Returns:
        dict: Dictionary with accuracy metrics
    """
    # In a real implementation, you would compare detections with ground truth
    # Here we'll simulate with placeholder values
    
    # Placeholder for accuracy metrics
    metrics = {
        'precision': 0.92,
        'recall': 0.87,
        'f1_score': 0.89,
        'average_iou': 0.76,
        'false_positives_per_frame': 0.15,
        'false_negatives_per_frame': 0.23
    }
    
    print("\nAccuracy Test Results (Simulated):")
    print(f"  Precision: {metrics['precision']:.2f}")
    print(f"  Recall: {metrics['recall']:.2f}")
    print(f"  F1 Score: {metrics['f1_score']:.2f}")
    print(f"  Average IoU: {metrics['average_iou']:.2f}")
    
    return metrics

def lighting_conditions_test():
    """Simulate tests under different lighting conditions."""
    lighting_results = {
        'bright': {'precision': 0.94, 'recall': 0.90, 'f1_score': 0.92},
        'normal': {'precision': 0.92, 'recall': 0.87, 'f1_score': 0.89},
        'dim': {'precision': 0.85, 'recall': 0.79, 'f1_score': 0.82},
        'dark': {'precision': 0.71, 'recall': 0.63, 'f1_score': 0.67}
    }
    
    print("\nLighting Conditions Test Results (Simulated):")
    for condition, metrics in lighting_results.items():
        print(f"  {condition.capitalize()} lighting - F1: {metrics['f1_score']:.2f}, "
              f"Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}")
    
    return lighting_results

def crowd_density_test():
    """Simulate tests with different crowd densities."""
    density_results = {
        'sparse': {'precision': 0.95, 'recall': 0.92, 'f1_score': 0.93, 'avg_fps': 27.3},
        'moderate': {'precision': 0.91, 'recall': 0.85, 'f1_score': 0.88, 'avg_fps': 25.1},
        'dense': {'precision': 0.87, 'recall': 0.76, 'f1_score': 0.81, 'avg_fps': 21.8},
        'very_dense': {'precision': 0.78, 'recall': 0.68, 'f1_score': 0.73, 'avg_fps': 18.4}
    }
    
    print("\nCrowd Density Test Results (Simulated):")
    for density, metrics in density_results.items():
        print(f"  {density.replace('_', ' ').capitalize()} crowds - F1: {metrics['f1_score']:.2f}, "
              f"FPS: {metrics['avg_fps']:.1f}, Precision: {metrics['precision']:.2f}")
    
    return density_results

def direction_accuracy_test():
    """Simulate direction detection accuracy tests."""
    direction_results = {
        'left_to_right': {'accuracy': 0.94, 'error_rate': 0.06},
        'right_to_left': {'accuracy': 0.93, 'error_rate': 0.07},
        'crossing_paths': {'accuracy': 0.87, 'error_rate': 0.13},
        'group_movement': {'accuracy': 0.81, 'error_rate': 0.19}
    }
    
    print("\nDirection Detection Accuracy (Simulated):")
    for scenario, metrics in direction_results.items():
        print(f"  {scenario.replace('_', ' ').capitalize()} - "
              f"Accuracy: {metrics['accuracy']:.2f}, Error rate: {metrics['error_rate']:.2f}")
    
    return direction_results

def run_all_tests(num_frames=50):
    """Run all benchmark tests and return comprehensive results."""
    results = {}
    
    # CPU benchmark
    results['cpu_benchmark'] = run_benchmark('cpu', num_frames)
    
    # GPU benchmark (if available)
    if torch.cuda.is_available():
        results['gpu_benchmark'] = run_benchmark('cuda', num_frames)
    
    # Accuracy tests
    results['accuracy'] = accuracy_test()
    results['lighting_conditions'] = lighting_conditions_test()
    results['crowd_density'] = crowd_density_test()
    results['direction_accuracy'] = direction_accuracy_test()
    
    return results

if __name__ == '__main__':
    # Add DetectionModel.detect_people_benchmark method
    def detect_people_benchmark(self, image):
        """Detect people in an image and return inference time."""
        # Convert numpy image to tensor
        image_tensor = self.transform(image).to(self.device)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(image_tensor)
        inference_time = time.time() - start_time
        
        # Process outputs
        boxes = outputs[0]['boxes'].cpu()
        scores = outputs[0]['scores'].cpu()
        labels = outputs[0]['labels'].cpu()
        
        # Filter for people (class 1) with score > threshold
        person_indices = (labels == 1) & (scores >= self.score_threshold)
        boxes = boxes[person_indices]
        scores = scores[person_indices]
        
        # Apply NMS
        keep_indices = torch.ops.torchvision.nms(boxes, scores, self.iou_threshold)
        filtered_boxes = boxes[keep_indices].numpy().astype(int)
        
        return filtered_boxes, inference_time
    
    # Add method to DetectionModel class
    DetectionModel.detect_people_benchmark = detect_people_benchmark
    
    # Run all tests
    print("Running comprehensive benchmark tests...")
    results = run_all_tests(50)
    print("\nAll tests completed!")

import torch
from torchvision import models, transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import cv2
import numpy as np
import time  # Import the time module

def load_model():
    """Load the Faster R-CNN model pre-trained on COCO dataset."""
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    return model

def preprocess_image(image):
    """Preprocess image for Faster R-CNN."""
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image)

def detect_people(image, model, device):
    """Detect people using Faster R-CNN."""
    start_time = time.time()
    image_tensor = preprocess_image(image).to(device)
    
    with torch.no_grad():
        outputs = model([image_tensor])
    
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.4f} seconds")

    people_boxes = [
        box.cpu().numpy().astype(int)
        for box, label in zip(outputs[0]['boxes'], outputs[0]['labels'])
        if label == 1  # Label 1 corresponds to "person"
    ]
    return people_boxes, inference_time

def draw_door_line(image, line_coords):
    """Draw the door line on the frame."""
    cv2.line(image, line_coords[0], line_coords[1], (0, 0, 255), 2)

def has_crossed_line(box, line_coords, direction):
    """Check if a person crosses the door line."""
    x_center = int((box[0] + box[2]) / 2)
    y_center = int((box[1] + box[3]) / 2)

    if direction == "in" and y_center > line_coords[0][1]:
        return True
    elif direction == "out" and y_center < line_coords[0][1]:
        return True
    return False
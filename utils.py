import torch
from torchvision import models, transforms, ops
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

def detect_people(image, model, device, score_threshold=0.8, iou_threshold=0.3):
    """Detect people using Faster R-CNN."""
    start_time = time.time()
    image_tensor = preprocess_image(image).to(device)
    
    with torch.no_grad():
        outputs = model([image_tensor])
    
    inference_time = time.time() - start_time
    # print(f"Inference time: {inference_time:.4f} seconds")

    boxes = outputs[0]['boxes']
    scores = outputs[0]['scores']
    labels = outputs[0]['labels']

    # Filter out non-person detections and low-confidence scores
    person_indices = (labels == 1) & (scores >= score_threshold)
    boxes = boxes[person_indices]
    scores = scores[person_indices]

    # Apply NMS
    keep_indices = ops.nms(boxes, scores, iou_threshold)
    people_boxes = boxes[keep_indices].cpu().numpy().astype(int)

    return people_boxes, inference_time
import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from PIL import Image  # Add this import

def load_model():
    """Load the Faster R-CNN model with COCO weights."""
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()  # Set to evaluation mode
    if torch.cuda.is_available():
        model = model.to('cuda')  # Use GPU if available
    return model

from PIL import Image  # Add this import

def detect_people(frame, model):
    """Run the object detection model on the frame."""
    from torchvision.transforms.functional import to_pil_image  # Use this to convert NumPy to PIL

    # Convert the NumPy frame to a PIL image
    pil_image = to_pil_image(frame)

    # Apply the transform for the Faster R-CNN model
    transform = FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()
    input_tensor = transform(pil_image).unsqueeze(0)  # Add batch dimension

    if torch.cuda.is_available():
        input_tensor = input_tensor.to('cuda')

    with torch.no_grad():
        outputs = model(input_tensor)[0]  # Get the detection results

    # Extract people boxes
    people_boxes = []
    for i, label in enumerate(outputs['labels']):
        if label == 1 and outputs['scores'][i] > 0.5:  # Label 1 is "person"
            people_boxes.append(outputs['boxes'][i].cpu().numpy())

    return people_boxes


def draw_door_line(frame, door_line):
    """Draw the door line on the frame."""
    cv2.line(frame, door_line[0], door_line[1], (0, 255, 0), 2)

def has_crossed_line(box, door_line):
    """Check if a person has crossed the door line."""
    x1, y1, x2, y2 = box
    person_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    return door_line[0][1] < person_center[1] < door_line[1][1]

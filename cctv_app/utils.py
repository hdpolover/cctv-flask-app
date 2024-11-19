import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from PIL import Image

def load_model():
    """Load the Faster R-CNN model with COCO weights."""
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    if torch.cuda.is_available():
        model = model.to('cuda')
    return model

def detect_people(frame, model):
    """Run the object detection model on the frame."""
    from torchvision.transforms.functional import to_pil_image
    pil_image = to_pil_image(frame)
    transform = FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()
    input_tensor = transform(pil_image).unsqueeze(0)
    if torch.cuda.is_available():
        input_tensor = input_tensor.to('cuda')
    with torch.no_grad():
        outputs = model(input_tensor)
    return outputs[0]['boxes'].cpu().numpy()

def draw_door_line(frame, door_line):
    """Draw the door line on the frame."""
    cv2.line(frame, door_line[0], door_line[1], (0, 255, 0), 2)

def has_crossed_line(box, door_line):
    """Check if a bounding box has crossed the door line."""
    x1, y1, x2, y2 = box
    line_x1, line_y1, line_x2, line_y2 = door_line[0] + door_line[1]
    return y1 < line_y1 and y2 > line_y2

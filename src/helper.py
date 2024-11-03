import cv2
import numpy as np
from supervision import BoxAnnotator, Detections


def read_image(path: str):
    img = cv2.imread(path)
    img_original = img.copy().astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    return img


def image_to_tensor(img):
    # Normalize the image (mean and std are placeholders; replace with actual values)
    img = img.astype(np.float32)
    # Standardize using the provided mean and std
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    img = (img - mean) / std
    # Reshape the image to (3, 512, 512)
    img = img.transpose(2, 0, 1)
    return img


def get_bboxes(outputs):
    # Convert bounding boxes into a format suitable for plotting
    boxes = []
    for box in outputs:
        x_min, y_min, x_max, y_max, confidence = box  # Adjust based on your output
        if confidence > 0.5:
            boxes.append([x_min, y_min, x_max, y_max, confidence])
    boxes = np.array(boxes)
    return boxes


def annotate_image(image, boxes):
    detections = Detections(
        xyxy=boxes[:, 0:4],
        class_id=np.array([0]*boxes.shape[0])
    )
    annotator = BoxAnnotator()

    # Plot the bounding boxes
    for box in boxes:
        x_min, y_min, x_max, y_max, confidence = box
        annotator.annotate(image, detections=detections)
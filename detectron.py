import cv2
import os
import numpy as np
import random
import torch
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from pycocotools import mask as maskUtils

# Set up the logger
setup_logger()

# Set up the configuration and the pre-trained model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
conf_threshold = 0.1
# Set the minimum and maximum area thresholds (in pixels)
min_area = 1000  # Minimum area threshold, e.g., 1000 square pixels
max_area = 100000  # Maximum area threshold, e.g., 100000 square pixels

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold  # Set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

# Create the predictor
predictor = DefaultPredictor(cfg)

# Load an image
image_path = "clutter_1467.png"
image = cv2.imread(image_path)

# Make a prediction
outputs = predictor(image)

# Process the prediction results
masks = outputs["instances"].pred_masks.cpu().numpy()
class_ids = outputs["instances"].pred_classes.cpu().numpy()
scores = outputs["instances"].scores.cpu().numpy()

# Get the class names
metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
class_names = metadata.get("thing_classes", None)

# Save the binary mask to file
mask_folder = "binary_masks"
os.makedirs(mask_folder, exist_ok=True)

# Filter masks based on confidence and area
filtered_masks = []
filtered_class_ids = []
filtered_scores = []
filtered_areas = []
for i, mask in enumerate(masks):
    # Convert the binary mask to an RLE encoded mask
    #rle = maskUtils.encode(np.array(mask[:, :, np.newaxis], order="F"))[0]

    # Compute the area of the mask
    #mask_area = maskUtils.area(rle)
    mask_area = cv2.countNonZero(mask.astype(np.uint8))

    # Filter the mask based on confidence and area
    if scores[i] > conf_threshold and (mask_area > min_area and mask_area < max_area):

        filtered_masks.append(mask)
        filtered_class_ids.append(class_ids[i])
        filtered_scores.append(scores[i])
        filtered_areas.append(mask_area)

        mask_file = os.path.join(mask_folder, f"mask_{i}.png")
        cv2.imwrite(mask_file, mask.astype(np.uint8) * 255)

print('filtered_areas: ', filtered_areas)

# Generate random colors for class list
detection_colors = []
for i in range(len(filtered_masks)):
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    detection_colors.append((b,g,r))

# Create a copy of the original image to draw masks on
image_with_masks = image.copy()

# Iterate through the masks and superpose them on the image
for i, mask in enumerate(filtered_masks):
    # Convert the mask to a color image
    mask = (mask * 255).astype(np.uint8)
    color = np.array(detection_colors[i], dtype=np.uint8)
    colored_mask = cv2.merge([mask, mask, mask]) * color

    # Superpose the colored mask on the original image
    image_with_masks = cv2.addWeighted(image_with_masks, 1, colored_mask, 0.5, 0)

    # Add the name of the object and the confidence score
    class_name = class_names[filtered_class_ids[i]] if class_names is not None else f"class_{filtered_class_ids[i]}"
    cv2.putText(image_with_masks, f"{class_name} ({filtered_scores[i]:.2f}) - {filtered_areas[i]:.0f}", (10, 30*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 1, detection_colors[i], 2)
# Save the image with superposed masks
cv2.imwrite('image_with_masks.png', image_with_masks)
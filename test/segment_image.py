import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import supervision as sv

def convert_bbox_x1y1x2y2_to_xywh(x1, y1, x2, y2):
    """Convert bounding box from (x1, y1, x2, y2) format to (x, y, w, h) format."""
    w = x2 - x1
    h = y2 - y1
    return x1, y1, w, h

def get_device():
    """Return the current device (GPU if available, otherwise CPU)."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_image_paths(image_dir):
    """Retrieve all image paths from a directory."""
    return [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]

def segment_image(yolo, mask_predictor, image_path, output_dir):
    """Perform image segmentation using YOLO for bounding boxes and SAM for mask prediction."""
    print(f"Processing image: {image_path}")
    
    # Predict bounding boxes using YOLO
    yolo_output = yolo.predict(image_path, conf=0.5)
    bounding_boxes = []

    for result in yolo_output:
        for bbox in result.boxes.data:  # Corrected syntax (removed extraneous comma)
            # Convert tensor to numpy array and cast to integer
            box_np = bbox.int().cpu().numpy()
            for b in box_np:
                # If needed, convert the box to another format here.
                # x, y, w, h = convert_bbox_x1y1x2y2_to_xywh(b[0], b[1], b[2], b[3])
                bounding_boxes.append({
                    "box": b[:4],  # Coordinates in (x1, y1, x2, y2) format
                    "score": b[5]  # Assuming b[5] is the confidence score
                })

    # Read the image and convert to RGB
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Set the image for SAM once outside the bounding box loop
    mask_predictor.set_image(image_rgb)

    # Initialize a combined mask with boolean type
    mask_combined = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=bool)

    for bbox_info in bounding_boxes:
        box = np.array(bbox_info["box"])
        # Predict mask using SAM with multimask output enabled
        masks, scores, logits = mask_predictor.predict(box=box, multimask_output=True)
        
        # For example: select the mask with the largest area
        mask_areas = np.array([m.sum() for m in masks])
        best_mask = masks[np.argmax(mask_areas)]
        
        # Combine masks using logical OR
        mask_combined = np.logical_or(mask_combined, best_mask)

    # Create the output image: apply the combined mask on the original image
    output_image = np.zeros_like(image_rgb)
    output_image[mask_combined] = image_rgb[mask_combined]

    # Save the segmented image
    base_name = os.path.basename(image_path).replace('.jpg', '.png')
    save_path = os.path.join(output_dir, f"outfit_{base_name}")
    cv2.imwrite(save_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    print(f"Segmented image saved at: {save_path}")

def main():
    # Define the base directory as the project root.
    # If your script is located inside a subfolder, this will move one level up.
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Set paths for images and models
    IMAGE_DIR = os.path.join("C:/Users/ADMIN/Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/images/original_images")
    OUTPUT_DIR = os.path.join("C:/Users/ADMIN/Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/images/segmented_images")
    MODELS_DIR = os.path.join("C:/Users/ADMIN/Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/models")
    
    # Model configurations and weight file paths
    MODEL_TYPE = "vit_h"
    CHECKPOINT_PATH = os.path.join(MODELS_DIR, "sam_weights.pth")
    YOLO_WEIGHTS = os.path.join(MODELS_DIR, "yolo_weights.pt")

    # Create the output directory if it does not exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize YOLO and SAM models
    yolo = YOLO(YOLO_WEIGHTS)
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=get_device())
    mask_predictor = SamPredictor(sam)
    
    # Retrieve list of images to process
    image_paths = get_image_paths(IMAGE_DIR)
    for image_path in image_paths:
        # Check if the segmented image already exists
        base_name = os.path.basename(image_path).replace('.jpg', '.png')
        segmented_image_path = os.path.join(OUTPUT_DIR, f"outfit_{base_name}")
        if os.path.exists(segmented_image_path):
            print(f"Segmented image {segmented_image_path} already exists, skipping.")
            continue
        # Perform segmentation
        segment_image(yolo, mask_predictor, image_path, OUTPUT_DIR)

if __name__ == "__main__":
    main()

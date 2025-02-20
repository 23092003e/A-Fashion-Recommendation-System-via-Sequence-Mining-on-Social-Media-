import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import supervision as sv

def convert_bbox_x1y1x2y2_to_xywh(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    x = x1
    y = y1
    return x, y, w, h  

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_image_paths(image_dir):
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
    return image_paths

def segment_image(yolo, mask_predictor, image_path):
    yolo_output = yolo.predict(image_path, conf=0.5)
    
    # Bounding box
    r = []
    for result in yolo_output:
        for bbox in result.boxes.data,:
            box = bbox.int().cpu().numpy()
            for b in box:
                x, y, w, h = convert_bbox_x1y1x2y2_to_xywh(b[0], b[1], b[2], b[3])
                r.append([b[0], b[1], b[2], b[3], b[5]])
                
    
    # Create the image variable and box_annotator
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return   
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_combined = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    output = np.zeros_like(image)

    for i, box in enumerate(r):
        box = box[:-1]
        box = np.array(box)

        mask_predictor.set_image(image)
    
        masks, scores, logits = mask_predictor.predict(box=box, multimask_output=True)

        detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks)
        detections = detections[detections.area == np.max(detections.area)]

        # Combine the masks with a logical OR operation within the loop
        for m in masks:
            mask_combined = np.logical_or(mask_combined, m)
        
    # Use the combined mask to select the pixels of the original image
    output[mask_combined] = image[mask_combined]
    
    # Save the image
    save_path = f"C:/Users/ADMIN\Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/images/segmented_images/outfit_{os.path.basename(image_path).replace('.jpg', '.png')}"
    cv2.imwrite(save_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving

def main():

    MODEL_TYPE = "vit_h"
    CHECKPOINT_PATH = os.path.join(os.getcwd(), "C:/Users/ADMIN/Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/models", "sam_weights.pth")
    YOLO_WEIGHTS = "C:/Users/ADMIN/Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/models/yolo_weights.pt"
    IMAGE_DIR = "C:/Users/ADMIN\Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/images/original_images"
    
    # Make sure the segmented images directory exists
    if not os.path.exists("../images/segmented_images"):
        os.makedirs("../images/segmented_images")

    # Create the YOLO and SAM models
    yolo = YOLO(YOLO_WEIGHTS)
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=get_device())
    mask_predictor = SamPredictor(sam)
    
    # Segment all images in the directory
    image_paths = get_image_paths(IMAGE_DIR)
    for image_path in image_paths:
        # Define the path of the segmented image
        segmented_image_path = f"C:/Users/ADMIN\Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/images/segmented_images/outfit_{os.path.basename(image_path).replace('.jpg', '.png')}"
        
        # Check if the segmented image already exists
        if os.path.exists(segmented_image_path):
            print(f"Segmented image {segmented_image_path} already exists, skipping.")
            continue

        # If it doesn't exist, perform the segmentation
        segment_image(yolo, mask_predictor, image_path)


if __name__ == "__main__":
    main()
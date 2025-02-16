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
        for bbox in result.boxes.data:
            box = bbox.int().cpu().numpy()
            r.append(box)  # Store the complete box data
                
    # Create the image variable
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Set the image for the SAM predictor
    mask_predictor.set_image(image_rgb)
    
    # Initialize an empty combined mask
    mask_combined = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
    output = np.zeros_like(image_rgb)
    
    for box in r:
        # Extract the bounding box coordinates
        x1, y1, x2, y2 = box[0:4]
        input_box = np.array([x1, y1, x2, y2])
        
        masks, scores, logits = mask_predictor.predict(
            box=input_box,
            multimask_output=True
        )
        
        # Select the mask with the highest score
        best_mask_idx = np.argmax(scores)
        mask = masks[best_mask_idx]
        
        # Combine the masks
        mask_combined = np.logical_or(mask_combined, mask)
    
    # Use the combined mask to select the pixels of the original image
    output[mask_combined] = image_rgb[mask_combined]
    
    # Save the image
    save_path = os.path.join(os.path.dirname(image_path), f"outfit_{os.path.basename(image_path).replace('.jpg', '.png')}")
    cv2.imwrite(save_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    print(f"Segmented image saved at: {save_path}")

def main():
    MODEL_TYPE = "vit_h"
    CHECKPOINT_PATH = "C:/Users/ADMIN/Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/sam2.1_b.pt"
    YOLO_WEIGHTS = "C:/Users/ADMIN/Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/yolo11n.pt"
    IMAGE_DIR = "C:/Users/ADMIN/Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/images/segmented_images"
    
    # Initialize YOLO
    yolo = YOLO(YOLO_WEIGHTS)
    
    # Initialize SAM
    device = get_device()
    sam = sam_model_registry[MODEL_TYPE](checkpoint=None).to(device=device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model", checkpoint)
        # Nếu state_dict chứa key "model", bóc tách thêm 1 lớp nữa
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        sam.load_state_dict(state_dict,strict=False)
    else:    
        sam = checkpoint.to(device)
    
    mask_predictor = SamPredictor(sam)


    # Process images
    image_paths = get_image_paths(IMAGE_DIR)
    for image_path in image_paths:
        segment_image_path = os.path.join(
            IMAGE_DIR,
            f"outfit_{os.path.basename(image_path).replace('.jpg', '.png')}"
        )
        
        if os.path.exists(segment_image_path):
            print(f"Segmented image already exists: {segment_image_path}. Skipping...")
            continue
        
        segment_image(yolo, mask_predictor, image_path)

if __name__ == "__main__":
    main()
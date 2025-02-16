import os
import cv2
import numpy as np
from ultralytics import YOLO, SAM
import supervision as sv


def get_image_paths(image_dir):
    """Get paths of all JPG images in the specified directory."""
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
    return image_paths


def segment_image(yolo_model, sam_model, image_path):
    """Segment objects in an image using YOLO and SAM models."""
    # Read and process image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get YOLO predictions
    yolo_results = yolo_model.predict(image_path, conf=0.5)[0]
    
    # Initialize mask array
    mask_combined = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
    output = np.zeros_like(image)
    
    # Process each YOLO detection
    if len(yolo_results.boxes.data) > 0:
        for box in yolo_results.boxes.data:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Get SAM prediction for this bounding box
            sam_results = sam_model.predict(
                source=image,
                input_boxes=np.array([[x1, y1, x2, y2]]),
                save=False,
                conf=0.5,
                retina_masks=True
            )[0]
            
            # Process SAM masks
            if len(sam_results.masks) > 0:
                for mask in sam_results.masks.data:
                    mask_array = mask.cpu().numpy()
                    mask_combined = np.logical_or(mask_combined, mask_array)
    
    # Apply combined mask to image
    output[mask_combined] = image[mask_combined]
    
    # Save the segmented image
    save_path = f"C:/Users/ADMIN/Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/images/segmented_images/outfit_{os.path.basename(image_path).replace('.jpg', '.png')}"
    cv2.imwrite(save_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))


def main():
    """Main function to perform segmentation on all images in a directory."""
    # Define paths
    YOLO_WEIGHTS = "C:/Users/ADMIN/Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/models/yolo_weights.pt"
    SAM_WEIGHTS = "C:/Users/ADMIN/Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/models/sam2.1_b.pt"  # or "sam2_l.pt" for lighter version
    IMAGE_DIR = "C:/Users/ADMIN/Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/images/segmented_images"
    
    # Create output directory if it doesn't exist
    os.makedirs("C:/Users/ADMIN/Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/images/segmented_images", exist_ok=True)
    
    # Initialize models
    yolo_model = YOLO(YOLO_WEIGHTS)
    sam_model = SAM(SAM_WEIGHTS)
    
    # Process all images
    image_paths = get_image_paths(IMAGE_DIR)
    for image_path in image_paths:
        # Define output path
        segmented_image_path = f"C:/Users/ADMIN/Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/images/segmented_images/outfit_{os.path.basename(image_path).replace('.jpg', '.png')}"
        
        # Skip if already processed
        if os.path.exists(segmented_image_path):
            print(f"Segmented image {segmented_image_path} already exists, skipping.")
            continue
        
        print(f"Processing {image_path}...")
        segment_image(yolo_model, sam_model, image_path)
        print(f"Saved segmented image to {segmented_image_path}")


if __name__ == "__main__":
    main()
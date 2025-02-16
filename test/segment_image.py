import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO

def get_device():
    """Trả về device hiện tại (GPU nếu có, ngược lại CPU)."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_image_paths(image_dir):
    """Lấy danh sách đường dẫn các ảnh (.jpg) trong thư mục."""
    return [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]

def segment_image(yolo, mask_predictor, image_path, output_dir):
    """Thực hiện phân đoạn ảnh bằng cách dùng YOLO để dự đoán bounding box và SAM để dự đoán mask."""
    print(f"Processing image: {image_path}")
    
    # Dự đoán bounding boxes bằng YOLO11
    yolo_output = yolo.predict(image_path, conf=0.5)
    bounding_boxes = []
    for result in yolo_output:
        # Giả sử result.boxes.data chứa các giá trị [x1, y1, x2, y2, confidence, class]
        boxes = result.boxes.data.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2, conf, class_id = box
            if conf < 0.5:
                continue
            bbox_int = np.array([int(x1), int(y1), int(x2), int(y2)])
            bounding_boxes.append({"box": bbox_int, "score": conf})
    
    # Đọc ảnh và chuyển sang RGB
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Thiết lập ảnh cho SAM2 (chỉ cần thực hiện 1 lần)
    mask_predictor.set_image(image_rgb)
    
    # Khởi tạo mask kết hợp với kiểu dữ liệu boolean
    mask_combined = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=bool)
    
    # Với mỗi bounding box dự đoán, sử dụng SAM2 để dự đoán segmentation mask
    for bbox_info in bounding_boxes:
        box = bbox_info["box"]
        # Dự đoán mask với multimask_output=True để lấy nhiều khả năng mask
        masks, scores, logits = mask_predictor.predict(box=box, multimask_output=True)
        # Chọn mask có diện tích lớn nhất (số pixel mask=1 nhiều nhất)
        mask_areas = np.array([m.sum() for m in masks])
        best_mask = masks[np.argmax(mask_areas)]
        mask_combined = np.logical_or(mask_combined, best_mask)
    
    # Tạo ảnh kết quả: áp dụng mask lên ảnh gốc
    output_image = np.zeros_like(image_rgb)
    output_image[mask_combined] = image_rgb[mask_combined]
    
    # Lưu ảnh kết quả
    base_name = os.path.basename(image_path).replace('.jpg', '.png')
    save_path = os.path.join(output_dir, f"outfit_{base_name}")
    cv2.imwrite(save_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    print(f"Segmented image saved at: {save_path}")

def main():
    # Đường dẫn (điều chỉnh cho phù hợp với cấu trúc của bạn)
    IMAGE_DIR = r"C:\Users\ADMIN\Desktop\ITDSIU21099_HoangVanManh\Fashion-Marketing-Automation-Solutions\images\original_images"
    OUTPUT_DIR = r"C:\Users\ADMIN\Desktop\ITDSIU21099_HoangVanManh\Fashion-Marketing-Automation-Solutions\images\segmented_images"
    MODELS_DIR = r"C:\Users\ADMIN\Desktop\ITDSIU21099_HoangVanManh\Fashion-Marketing-Automation-Solutions\models"
    
    # Cấu hình model và đường dẫn tới weights
    # Lưu ý: thay đổi MODEL_TYPE nếu cần. Ở đây mình giả sử key cho SAM2 là "sam_v2"
    MODEL_TYPE = "sam_v2"
    CHECKPOINT_PATH = os.path.join(MODELS_DIR, "sam2_weights.pt")
    YOLO_WEIGHTS = os.path.join(MODELS_DIR, "yolo11_weights.pt")
    
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Khởi tạo mô hình YOLO11
    yolo = YOLO(YOLO_WEIGHTS)
    
    # Khởi tạo SAM2 và load weights
    device = get_device()
    # Khởi tạo SAM2 mà không truyền checkpoint ban đầu
    sam = sam_model_registry[MODEL_TYPE](checkpoint=None).to(device=device)
    # Load checkpoint, nếu file checkpoint chứa thêm thông tin (ví dụ: "model", "date", …)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    sam.load_state_dict(state_dict)
    
    mask_predictor = SamPredictor(sam)
    
    # Lấy danh sách ảnh cần xử lý
    image_paths = get_image_paths(IMAGE_DIR)
    for image_path in image_paths:
        base_name = os.path.basename(image_path).replace('.jpg', '.png')
        segmented_image_path = os.path.join(OUTPUT_DIR, f"outfit_{base_name}")
        if os.path.exists(segmented_image_path):
            print(f"Segmented image {segmented_image_path} already exists, skipping.")
            continue
        segment_image(yolo, mask_predictor, image_path, OUTPUT_DIR)

if __name__ == "__main__":
    main()

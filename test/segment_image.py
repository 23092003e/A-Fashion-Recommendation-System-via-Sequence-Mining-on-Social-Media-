import os
import cv2
import numpy as np
from ultralytics import YOLO, SAM

def download_models():
    """Download và lưu models vào thư mục models/"""
    os.makedirs("models", exist_ok=True)
    
    # Download YOLO model
    yolo_path = "models/yolo_weights.pt"
    if not os.path.exists(yolo_path):
        print("Đang tải YOLO model...")
        model = YOLO('yolov8x.pt')
        # Sao chép file model trực tiếp
        import shutil
        shutil.copy('yolov8x.pt', yolo_path)
        print(f"Đã lưu YOLO model tại: {yolo_path}")
    
    # Download SAM2 model
    sam_path = "models/sam_weights.pt"
    if not os.path.exists(sam_path):
        print("Đang tải SAM2 model...")
        model = SAM('sam2_h.pt')
        # Sao chép file model trực tiếp
        import shutil
        shutil.copy('sam2_h.pt', sam_path)
        print(f"Đã lưu SAM2 model tại: {sam_path}")
def main():
    # Tạo thư mục lưu kết quả
    os.makedirs("images/segmented_images", exist_ok=True)
    
    # Download models nếu chưa có
    download_models()
    
    # Khởi tạo models từ files đã lưu
    yolo_model = YOLO('models/yolo_weights.pt')
    sam_model = SAM('models/sam_weights.pt')
    
    # Đường dẫn đến folder ảnh gốc
    input_folder = "images/original_images"
    
    # Lấy danh sách ảnh
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Xử lý từng ảnh
    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        output_path = f"images/segmented_images/segmented_{image_file}"
        
        if os.path.exists(output_path):
            print(f"Ảnh {output_path} đã tồn tại, bỏ qua.")
            continue
        
        print(f"Đang xử lý ảnh: {image_file}")
        
        # Đọc ảnh
        image = cv2.imread(input_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Nhận diện đối tượng bằng YOLO
        yolo_results = yolo_model.predict(image, conf=0.5)[0]
        
        # Khởi tạo mask tổng hợp
        mask_combined = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        output = np.zeros_like(image)
        
        # Xử lý từng đối tượng được phát hiện
        if len(yolo_results.boxes.data) > 0:
            for box in yolo_results.boxes.data:
                x1, y1, x2, y2 = map(int, box[:4])
                
                # Dùng SAM để segment chi tiết
                sam_results = sam_model.predict(
                    source=image,
                    input_boxes=np.array([[x1, y1, x2, y2]]),
                    save=False,
                    conf=0.5,
                    retina_masks=True
                )[0]
                
                # Kết hợp các mask
                if len(sam_results.masks) > 0:
                    for mask in sam_results.masks.data:
                        mask_array = mask.cpu().numpy()
                        mask_combined = np.logical_or(mask_combined, mask_array)
        
        # Áp dụng mask vào ảnh
        output[mask_combined] = image[mask_combined]
        
        # Lưu ảnh kết quả
        cv2.imwrite(output_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        print(f"Đã lưu ảnh kết quả: {output_path}")

if __name__ == "__main__":
    main()
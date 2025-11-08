import cv2
import os
import json
from tqdm import tqdm

def extract_frame(video_path, save_path, timestamp_sec):
    # TODO (Giao cho Kỹ sư Data):
    # 1. Dùng cv2.VideoCapture để mở video_path
    # 2. Dùng cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000) để tua đến đúng thời điểm
    # 3. Đọc frame: ret, frame = cap.read()
    # 4. Lưu frame: cv2.imwrite(save_path, frame)
    # 5. Nhớ cap.release()
    pass

def main():
    print("Starting frame extraction...")
    # TODO (Giao cho Kỹ sư Data):
    # 1. Đọc 'data/train.json'
    # 2. Tạo thư mục 'data/train_frames/' (os.makedirs)
    # 3. Dùng vòng lặp (có thể dùng tqdm) lặp qua từng mẫu trong 'data'
    # 4. Lấy video_path, support_frames (là 1 list)
    # 5. Với mỗi timestamp trong support_frames:
    #    a. Tạo tên file ảnh (ví dụ: 'train_0001_frame_0.jpg')
    #    b. Tạo đường dẫn lưu (ví dụ: 'data/train_frames/train_0001_frame_0.jpg')
    #    c. Gọi hàm extract_frame(video_path, save_path, timestamp)
    print("Frame extraction complete.")

if __name__ == "__main__":
    main()
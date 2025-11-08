import cv2
import os
import json
from tqdm import tqdm

TRAIN_JSON_PATH = "data/train.json"
VIDEO_ROOT_DIR = "data" 
FRAME_OUTPUT_DIR = "data/train_frames" 

def extract_frame(video_path, save_path, timestamp_sec):
    # TODO (Giao cho Kỹ sư Data):
    # 1. Dùng cv2.VideoCapture để mở video_path
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # 2. Dùng cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000) để tua đến đúng thời điểm
    timestamp_msec = timestamp_sec * 1000
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_msec)

    # 3. Đọc frame: ret, frame = cap.read()
    ret, frame = cap.read()

    # 4. Lưu frame: cv2.imwrite(save_path, frame)
    if ret:
        cv2.imwrite(save_path, frame)
        cap.release()
        return True
    else:
        print(f"Warning: Could not read frame at {timestamp_sec} seconds in {video_path}")
        cap.release()
        return False
    
    # 5. Nhớ cap.release()
    

def main():
    print("Starting frame extraction from {TRAIN_JSON_PATH}")
    # TODO (Giao cho Kỹ sư Data):
    # 1. Đọc 'data/train.json'
    with open (TRAIN_JSON_PATH, 'r', encoding = 'utf-8') as f:
        data = json.load(f)["data"]

    # 2. Tạo thư mục 'data/train_frames/' (os.makedirs)
    os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)
    print(f"Extracting frames to {FRAME_OUTPUT_DIR}")

    # 3. Dùng vòng lặp (có thể dùng tqdm) lặp qua từng mẫu trong 'data'
    success_count = 0
    fail_count = 0

    for item in tqdm(data, desc = "Extracting frames"):
        item_id = item["id"] #eg: 'train_0001'
        video_path_relative = item["video_path"] #eg: 'train/videos/train_2b840...mp4'
        support_frames_sec = item["support_frames"] #eg: [4.427...]

        video_path_full = os.path.join(VIDEO_ROOT_DIR, video_path_relative)

        if not os.path.exists(video_path_full):
            print(f"Warning: Video file does not exist {video_path_full}")
            fail_count += len(support_frames_sec)
            continue

        for frame_idx, timestamp in enumerate(support_frames_sec):
            #eg: 'train_0001_frame_0.jpg'
            output_filename = f"{item_id}_frame_{frame_idx}.jpg"
            output_save_path = os.path.join(FRAME_OUTPUT_DIR, output_filename)

            if extract_frame(video_path_full, output_save_path, timestamp):
                success_count += 1
            else:
                fail_count += 1

    # 4. Lấy video_path, support_frames (là 1 list)
    # 5. Với mỗi timestamp trong support_frames:
    #    a. Tạo tên file ảnh (ví dụ: 'train_0001_frame_0.jpg')
    #    b. Tạo đường dẫn lưu (ví dụ: 'data/train_frames/train_0001_frame_0.jpg')
    #    c. Gọi hàm extract_frame(video_path, save_path, timestamp)
    print("Frame extraction complete.")
    print(f"Total frames extracted successfully: {success_count}")
    print(f"Total frames failed to extract: {fail_count}")

if __name__ == "__main__":
    main()
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image

class TrafficBuddyDataset(Dataset):
    def __init__(self, json_path, frames_dir, tokenizer, image_processor):
        # TODO (Giao cho Kỹ sư Data):
        # 1. Đọc json_path
        # 2. self.data = ...
        self.frames_dir = frames_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        print("Dataset initialized (TODO)")

    def __len__(self):
        # TODO (Giao cho Kỹ sư Data):
        # return len(self.data)
        return 10 # Trả về số 10 để test

    def __getitem__(self, idx):
        # TODO (Giao cho Kỹ sư Data):
        # 1. Lấy mẫu data[idx]
        # 2. Lấy câu hỏi, các choices, và video_path/id
        # 3. Tạo câu prompt (ví dụ: "Question: ... A: ... B: ... Answer:")
        # 4. Tìm đường dẫn ảnh trong self.frames_dir (ví dụ: 'train_0001_frame_0.jpg')
        # 5. Đọc ảnh: image = Image.open(image_path)
        # 6. Xử lý ảnh và text:
        #    inputs = self.image_processor(image, return_tensors='pt')
        #    text_inputs = self.tokenizer(prompt, return_tensors='pt')
        # 7. Return inputs (ảnh, text, và label (câu trả lời đúng))
        pass
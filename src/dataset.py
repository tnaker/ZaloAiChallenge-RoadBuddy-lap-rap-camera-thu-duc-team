from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image # Dùng PIL (Pillow) để đọc ảnh
import json

class TrafficBuddyDataset(Dataset):
    def __init__(self, json_path, frames_dir, processor, max_length=128):
        """
        json_path: Đường dẫn đến 'train.json'
        frames_dir: Đường dẫn đến 'data/train_frames/'
        processor: 'processor' của Hugging Face (ví dụ: LlavaProcessor)
                   Nó bao gồm cả image_processor và tokenizer.
        """
        print(f"Đang tải dữ liệu từ: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)["data"]
            
        self.frames_dir = frames_dir
        self.processor = processor
        self.max_length = max_length
        print(f"Đã tìm thấy {len(self.data)} mẫu.")

    def __len__(self):
        """Trả về tổng số lượng mẫu."""
        return len(self.data)
        
    def __getitem__(self, idx):
        """
        Lấy 1 mẫu (ảnh + text) tại vị trí idx.
        """
        # Lấy thông tin của mẫu
        item = self.data[idx]
        item_id = item['id']
        question = item['question']
        choices = item['choices']
        answer = item['answer']
        
        # ---- 1. Xử lý Ảnh ----
        # Giả định: Chúng ta chỉ dùng ảnh đầu tiên trong support_frames
        # Tên ảnh sẽ là: "train_xxxx_frame_0.jpg"
        image_filename = f"{item_id}_frame_0.jpg"
        image_path = os.path.join(self.frames_dir, image_filename)
        
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"[Error] Không tìm thấy ảnh: {image_path}. Trả về ảnh rỗng.")
            image = Image.new('RGB', (224, 224), color = 'black') # Trả về ảnh đen
        
        # ---- 2. Xử lý Text (Prompt) ----
        # Đây là một phần quan trọng (Prompt Engineering)
        # Chúng ta phải "dạy" model cách trả lời.
        
        # Ghép các lựa chọn thành một chuỗi
        formatted_choices = "\n".join(choices)
        
        # Tạo prompt đầu vào cho LLaVA
        # LLaVA dùng một định dạng chat đặc biệt
        # USER: <ảnh> [câu hỏi]
        # ASSISTANT: [câu trả lời]
        prompt = f"USER: <image>\n{question}\n{formatted_choices}\nASSISTANT:"
        
        # ---- 3. Chuẩn bị đầu ra ----
        # Dùng processor để xử lý cả ảnh và text
        # `processor` sẽ xử lý ảnh, và tokenize `prompt` (phần USER)
        # và tokenize `answer` (phần ASSISTANT) để làm 'labels'
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        # Chúng ta cần tokenize 'answer' riêng để làm 'labels'
        # Thêm 'answer' vào cuối prompt để tạo labels
        prompt_with_answer = prompt + " " + answer
        
        labels = self.processor(
            text=prompt_with_answer,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )["input_ids"]

        # Với LLaVA, chúng ta không muốn model tính loss cho phần prompt (phần USER)
        # Chúng ta chỉ muốn nó học cách trả lời (phần ASSISTANT)
        # Chúng ta set tất cả token của prompt là -100 (để bỏ qua loss)
        
        # Tìm vị trí bắt đầu của câu trả lời (ASSISTANT:)
        # Tạm thời, để đơn giản cho Giai đoạn 1, chúng ta có thể bỏ qua bước này
        # và chỉ huấn luyện trên toàn bộ chuỗi.
        # Khi tối ưu, chúng ta sẽ quay lại đây.
        
        # Lấy kết quả từ `inputs` (chỉ lấy phần tử đầu tiên vì batch_size=1)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = labels.squeeze(0)
        
        return inputs
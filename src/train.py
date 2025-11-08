import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
import os

# -- (1) Import "đường ống nguyên liệu" của Kỹ sư Data --
# (Giả định file src/dataset.py đã được hoàn thành)
try:
    from dataset import TrafficBuddyDataset
except ImportError:
    print("="*50)
    print("LỖI: Không thể import 'TrafficBuddyDataset' từ 'src/dataset.py'")
    print("Hãy đảm bảo Kỹ sư Data đã hoàn thành file đó!")
    print("="*50)
    exit()

# --- (2) Cấu hình Mô hình (Trái tim nhà máy) ---
MODEL_ID = "llava-hf/llava-1.5-7b-hf" # <-- Model chúng ta đã chốt!

# Cấu hình để load 4-bit (Quantization)
# Đây là phần của Kỹ sư Ops (cài bitsandbytes)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# --- (3) Hàm chính để "Chạy thông" ---
def run_end_to_end_test():
    """
    Đây là hàm "chạy thông" cho Giai đoạn 1.
    Nó sẽ tải mô hình, tải 1 batch dữ liệu, và chạy 1 bước huấn luyện.
    """
    print(f"--- Giai đoạn 1: Bắt đầu 'Chạy thông' (End-to-End Test) ---")

    # 3.1. Tải Processor (Bộ xử lý ảnh + text)
    print(f"Đang tải Processor cho model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    # 3.2. Tải Model (Phần quan trọng!)
    print(f"Đang tải model 4-bit: {MODEL_ID}. Việc này có thể mất vài phút...")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto" # Tự động đẩy lên GPU nếu có
        #llm_int8_enable_fp32_cpu_offload=True
    )
    print("Tải model thành công!")

    # 3.3. Chuẩn bị "Nguyên liệu" (Dùng code của Kỹ sư Data)
    print("Đang khởi tạo 'TrafficBuddyDataset' (dùng code Kỹ sư Data)...")
    train_dataset = TrafficBuddyDataset(
        json_path="data/train.json",
        frames_dir="data/train_frames", # <-- Thư mục ảnh Kỹ sư Data tạo ra
        processor=processor,
        max_length=128 # Giữ max_length nhỏ để chạy test cho nhanh
    )
    
    # (Tùy chọn) Chúng ta chỉ test với 10 mẫu cho Giai đoạn 1
    # Bằng cách tạo một "Subset" (tập con)
    from torch.utils.data import Subset
    test_subset = Subset(train_dataset, range(10))
    print(f"Đã tạo một tập con 10 mẫu để 'chạy thông'.")


    # 3.4. Tạo "Băng chuyền" (DataLoader)
    # data_collator sẽ xử lý padding cho chúng ta
    # (LLaVA không cần collator đặc biệt, nó đã pad trong Dataset)
    test_dataloader = DataLoader(
        test_subset, 
        batch_size=2, # Batch size nhỏ để test
        shuffle=True
    )

    # 3.5. Lấy "lô hàng" đầu tiên
    print("Đang lấy lô hàng (batch) đầu tiên từ băng chuyền...")
    try:
        batch = next(iter(test_dataloader))
        print("Lấy batch thành công!")
    except Exception as e:
        print(f"LỖI khi lấy batch! Kỹ sư Data hãy kiểm tra lại src/dataset.py: {e}")
        return

    # 3.6. Đẩy "lô hàng" lên GPU
    try:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        print(f"Đã chuyển batch lên thiết bị: {model.device}")
    except Exception as e:
        print(f"LỖI khi chuyển batch lên GPU. Kỹ sư Ops hãy kiểm tra CUDA/GPU: {e}")
        return

    # 3.7. "NHÀ MÁY HOẠT ĐỘNG" (BƯỚC QUAN TRỌNG NHẤT)
    print("Đang đưa batch vào model (forward pass)...")
    
    # Cho model "học"
    outputs = model(**batch)
    
    # Lấy "lỗi"
    loss = outputs.loss

    print("\n" + "="*50)
    print("           🎉 CHÚC MỪNG CẢ ĐỘI! 🎉")
    print("     PIPELINE ĐÃ CHẠY THÔNG (END-TO-END)!")
    print(f"     Loss của batch đầu tiên: {loss.item()}")
    print("="*50 + "\n")
    print("Giai đoạn 1 (Nền tảng) coi như HOÀN THÀNH.")
    print("Nhiệm vụ tiếp theo (Giai đoạn 2):")
    print(" - Core: Hoàn thiện vòng lặp training, dùng 'Trainer' của Hugging Face.")
    print(" - Data: Bắt đầu tạo dữ liệu tổng hợp (synthetic data).")
    print(" - Ops: Tập trung tối ưu tốc độ inference và hoàn thiện Docker.")


if __name__ == "__main__":
    # Đây là lệnh Kỹ sư Ops sẽ chạy bên trong Docker:
    # python src/train.py
    run_end_to_end_test()
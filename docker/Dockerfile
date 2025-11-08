# TODO (Giao cho Kỹ sư Ops):
# 1. Chọn 1 base image của NVIDIA (ví dụ: nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 2. Cài đặt các thư viện hệ thống (python3, python3-pip, git, opencv)
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-pip python3.10-venv git \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Tạo thư mục làm việc
WORKDIR /app

# 4. Copy file requirements.txt VÀO TRƯỚC
COPY requirements.txt .

# 5. Cài đặt thư viện Python (để tận dụng Docker cache)
RUN pip3 install --no-cache-dir -r requirements.txt

# 6. Copy TOÀN BỘ code của dự án vào
COPY . .

# 7. (Lệnh CMD/ENTRYPOINT sẽ được thêm sau, khi có script inference.py)
# CMD ["echo", "TODO: Add inference command"]
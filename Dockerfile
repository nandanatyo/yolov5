FROM python:3.9-slim
WORKDIR /app

# Install dependencies sistem
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch dengan versi CPU yang lebih ringan
RUN pip install --no-cache-dir torch==1.8.0+cpu torchvision==0.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && pip cache purge

# Clone repositori YOLOv5
RUN git clone https://github.com/ultralytics/yolov5.git /app/yolov5 \
    && cd /app/yolov5 \
    && pip install --no-cache-dir -r requirements.txt

# Salin kode aplikasi Anda
COPY . .

# Pastikan modul YOLO dapat diimport
ENV PYTHONPATH="${PYTHONPATH}:/app/yolov5"

EXPOSE 8000
CMD ["python", "app.py"]
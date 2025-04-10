FROM python:3.9-slim

WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn websockets


COPY . .


EXPOSE 8000


CMD ["python", "app.py"]
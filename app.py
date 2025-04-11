import base64
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, APIRouter
from pathlib import Path
import torch
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
import traceback

app = FastAPI()

api_router = APIRouter(prefix="/api/v1")
ktp_router = APIRouter(prefix="/ktp")

weights = 'runs/train/exp10/weights/best.pt'
device = select_device('')
model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
stride = model.stride
imgsz = (640, 640)
model.eval()

def process_image(img):
    img_resized = letterbox(img, imgsz, stride=stride, auto=True)[0]
    img_resized = img_resized.transpose((2, 0, 1))[::-1]
    img_resized = np.ascontiguousarray(img_resized)
    img_tensor = torch.from_numpy(img_resized).float().to(device)
    img_tensor /= 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

@ktp_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client Connected!")

    while True:
        try:
            data = await websocket.receive_text()
            img_data = base64.b64decode(data)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                await websocket.send_json({"message": "Invalid image data"})
                continue

            img_tensor = process_image(img)
            pred = model(img_tensor)
            pred = non_max_suppression(pred, 0.3, 0.45)[0]

            if pred is not None and len(pred):
                pred[:, :4] = scale_boxes(img_tensor.shape[2:], pred[:, :4], img.shape).round()
                x1, y1, x2, y2, conf, cls = pred[0].cpu().numpy()

                w, h = img.shape[1], img.shape[0]
                box_w, box_h = x2 - x1, y2 - y1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                center_left = w * 0.4
                center_right = w * 0.6

                if box_w < w * 0.9:
                    msg = "KTP kurang dekat"
                elif box_w > w * 0.995:
                    msg = "KTP terlalu dekat"
                elif cx < center_left:
                    msg = "KTP geser ke kanan"
                elif cx > center_right:
                    msg = "KTP geser ke kiri"
                else:
                    msg = "OK"

                result = {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": float(conf),
                    "center": [float(cx), float(cy)],
                    "box_size": [float(box_w), float(box_h)],
                    "message": msg
                }
            else:
                result = {"message": "KTP tidak terdeteksi"}

            await websocket.send_json(result)

        except Exception as e:
            error_detail = traceback.format_exc()
            print(f"Error: {e}\n{error_detail}")
            await websocket.send_json({"error": str(e), "detail": error_detail})
            continue

api_router.include_router(ktp_router)
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    print(f"Model loaded: {weights}")
    print(f"Device: {device}")
    print(f"Model stride: {stride}")
    print(f"Input image size: {imgsz}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
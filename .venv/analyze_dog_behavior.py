import cv2
import os
import numpy as np
from ultralytics import YOLO
import argparse

# Argument parsing: use --single for single-camera mode
parser = argparse.ArgumentParser(description='DogSentinelAI Video Demo')
parser.add_argument('--single', action='store_true', help='Tek kamera modu')
args = parser.parse_args()

# Model yükleme (yolunuzu gerektiği gibi güncelleyin)
model = YOLO('C:/Users/Sibel Kaya/Desktop/DogIdwithYOLO/.venv/runs/train/dog_detector/weights/best.pt')
# Sınıf isimleri (model.names, dict ya da list olarak)
class_names = model.names

# Kamera yolları
camera_paths = [
    'videos/kamera1.mp4',
    'videos/kamera2.mp4',
    'videos/kamera3.mp4'
]

# Seçilen moda göre stream'leri oluştur
selected_paths = [camera_paths[0]] if args.single else camera_paths
caps = [cv2.VideoCapture(p) if os.path.exists(p) else None for p in selected_paths]

# Boş frame oluşturma
def create_blank_frame():
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    cv2.putText(frame, 'Kamera bulunamadı', (150, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

# Kare işleme: tespit ve kutu çizimi
def process_frame(frame):
    # Model tahmin
    results = model.predict(frame, imgsz=640, conf=0.3)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    classes = results[0].boxes.cls.cpu().numpy().astype(int)

    for box, cls in zip(boxes, classes):
        label = class_names[cls]
        # Sadece 'dog' sınıfı için çizim
        if label.lower() == 'dog':
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label.capitalize(), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame

# Ana döngü
window_name = 'SingleCam' if args.single else 'MultiCam'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    frames = []
    for cap in caps:
        if cap is None:
            frames.append(create_blank_frame())
        else:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                frame = process_frame(frame)
                frames.append(frame)
            else:
                frames.append(create_blank_frame())

    # Görüntüleri birleştir veya tek göster
    output = frames[0] if args.single else np.hstack(frames)

    cv2.imshow(window_name, output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
for cap in caps:
    if cap:
        cap.release()
cv2.destroyAllWindows()

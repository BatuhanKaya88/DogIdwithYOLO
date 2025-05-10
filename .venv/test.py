from ultralytics import YOLO

if __name__ == "__main__":
    # Başlangıç modelini yükle
    model = YOLO('yolov8n.pt')

    # Eğitimi başlat
    model.train(
        data='C:/Users/Sibel Kaya/Desktop/DogIdwithYOLO/.venv/dataset/data.yaml',  # Data dosyasının doğru yolu
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        project='C:/Users/Sibel Kaya/Desktop/DogIdwithYOLO/.venv/runs/train',
        # Eğitim çıktılarının kaydedileceği klasör
        name='dog_detector'  # Eğitim adı
    )

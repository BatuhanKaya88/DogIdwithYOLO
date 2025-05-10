from ultralytics import YOLO

if __name__ == "__main__":
    # Load the initial model
    model = YOLO('yolov8n.pt')

    # Start training
    model.train(
        data='C:/Users/YOURDESKTOP/Desktop/DogIdwithYOLO/.venv/dataset/data.yaml',  # Correct path to the data file
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        project='C:/Users/YOURDESKTOP/Desktop/DogIdwithYOLO/.venv/runs/train',  # Directory to save training outputs
        name='dog_detector'  # Training name
    )

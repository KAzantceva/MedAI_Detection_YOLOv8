from ultralytics import YOLO

# Загрузка модели
model = YOLO("yolov8n.pt")  # предварительно обученная модель

# Обучение модели
model.train(data="alzheimer.yaml", epochs=10)  # укажите количество эпох, которые вам нужны

from ultralytics import YOLO
model = YOLO("models/best.pt")
model.export(format="ncnn", imgsz=320)
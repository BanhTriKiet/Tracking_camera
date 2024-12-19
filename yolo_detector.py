from ultralytics import YOLO


class YoloDetector:
  def __init__(self, model_path, confidence):
    self.model = YOLO(model_path, task='detect')
    #self.model.export(format="ncnn")
    self.classList = ["Person"]
    self.confidence = confidence

  def detect(self, image):
    results = self.model.predict(image, conf=self.confidence, imgsz=320)
    result = results[0]
    detections = self.make_detections(result)
    return detections

  def make_detections(self, result):
    boxes = result.boxes
    detections = []
    for box in boxes:
      x1, y1, x2, y2 = box.xyxy[0]
      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
      w, h = x2 - x1, y2 - y1
      #class_number = 0
      #print("class_number: ", class_number)
      if result.names[0] not in self.classList:
        continue
      conf = box.conf[0]
      detections.append((([x1, y1, w, h]), 0, conf))
    return detections

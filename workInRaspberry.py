import time
import numpy as np
import math
from absl import flags
import cv2
from multiprocessing import Process, Value
from PCA9685 import PCA9685
from picamera2 import Picamera2
import tensorflow as tf
from yolov3_tf2.models import YoloV3Tiny
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

# Global variables
image_size = 310
num_classes = 80
weight_file = './weights/yolov3-tiny.tf'
pwm = PCA9685()
pwm.setPWMFreq(50)

current_angle = Value('i', 90)  # Shared variable for multiprocessing
target_angle = Value('i', 90)

# Servo control function
def control_servo(servo, current_angle, target_angle):
    while True:
        if target_angle.value != current_angle.value:
            if target_angle.value > current_angle.value and current_angle.value + 1 < 180:
                current_angle.value += 1
                pwm.setRotationAngle(servo, current_angle.value)
            elif target_angle.value < current_angle.value and current_angle.value - 1 > 0:
                current_angle.value -= 1
                pwm.setRotationAngle(servo, current_angle.value)
        #time.sleep(0.01)  # Reduce CPU usage

# Calculate target angle based on object direction
def cal_distance(servo, direction, current_angle, target_angle):
    if servo == 1:
        if direction < 320:
            angle = math.ceil((320 - direction) / 45)
            target_angle.value = min(current_angle.value + angle, 180)
        elif direction > 320:
            angle = math.ceil((direction - 320) / 45)
            target_angle.value = max(current_angle.value - angle, 0)

# Main YOLO and tracking function
def yolo_tracking(current_angle, target_angle):
    # Setup YOLO and Deep Sort
    flags.FLAGS(['program_name'])  # Initialize absl flags
    max_cosine_distance = 0.5
    nn_budget = None
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    yolo = YoloV3Tiny(classes=num_classes)
    yolo.load_weights(weight_file)

    # Camera setup
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))
    cam.start()

    fps = 0.0  # Initialize FPS variable

    while True:
        img = cam.capture_array()
        if img is None:
            time.sleep(0.1)
            continue

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, image_size)

        t2 = time.time()  # Measure processing start time
        boxes, scores, classes, nums = yolo.predict(img_in)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)

        detections = [Detection(converted_boxes[i], scores[0][i], 'person', features[i])
                      for i in range(len(converted_boxes)) if scores[0][i] > 0.5]

        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            direction = int((bbox[0] + bbox[2]) / 2)
            cal_distance(1, direction, current_angle, target_angle)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(img, 'person-' + str(track.track_id),
                        (int(bbox[0]), int(bbox[1]) - 10), 0, 0.75, (255, 255, 255), 2)

        # Calculate FPS
        fps = (fps + (1.0 / (time.time() - t2))) / 2

        # Display FPS on the image
        cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        # Display the image
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    servo_process = Process(target=control_servo, args=(1, current_angle, target_angle))
    servo_process.start()

    yolo_tracking_process = Process(target=yolo_tracking, args=(current_angle, target_angle))
    yolo_tracking_process.start()

    servo_process.join()
    yolo_tracking_process.join()

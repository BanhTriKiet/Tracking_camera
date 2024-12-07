import time
import numpy as np
from absl import app
import cv2
# import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import (
 YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
# from PIL import Image

image_size =310
num_classes=80
weight_file='./weights/yolov3-tiny.tf'
def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None  
    #initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    yolo = YoloV3Tiny(classes=num_classes)


    yolo.load_weights(weight_file)
    print('weights loaded')


    try:
        vid = cv2.VideoCapture(int('./data/video/test.mp4'))
    except:
        vid = cv2.VideoCapture('./data/video/test.mp4')
    
    fps = 0.0
    count = 0
    names = ['person']
    names = np.array(names)    
    # for i in range(3):

    #     names.append('person')
    while True:
        _, img = vid.read()

        if img is None:
            print("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else:
                break
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, image_size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)
        if converted_boxes and scores[0][0] and features.all():
            detections = [Detection(converted_boxes[0], scores[0][0], 'person', features[0])]
        else:
            detections=[]
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        color = (255, 255, 255)
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()

            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(img, 'person' + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
        # print fps on screen 
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('output', 800, 600)  # Kích thước cửa sổ (800x600)
        # Hiển thị hình ảnh trong cửa sổ
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break
    vid.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
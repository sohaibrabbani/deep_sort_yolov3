#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image

from TrackingFlowObject import TrackingFlowObject
import pickle
from yolo import YOLO
from collections import OrderedDict

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils.video
from videocaptureasync import VideoCaptureAsync

warnings.filterwarnings('ignore')


def serialize_flow_data(flows, height, width, frame_count, fps, file_path):
    # Saving the flow file as a list to make it compatible for the unfolding_viz code - Sohaib
    flows_list = [flows[i] for i in flows.keys() if len(flows[i]) > 32]

    # Saving the TrackFlowObject for the serialization - Sohaib
    track_flow = TrackingFlowObject(height=height,
                                    width=width,
                                    flows=flows_list,
                                    framecount=frame_count,
                                    fps=fps)

    # Extracting the name from the file path to save the pickle file - Sohaib
    track_flow_pickle = open(file_path.split('/')[-1].split('.')[0] + ".pickle", "wb")

    # Dumping the TrackFlowObject in a pickle file - Sohaib
    pickle.dump(track_flow, track_flow_pickle)
    track_flow_pickle.close()


def main(yolo):
    # Definition of the parameters
    h = 0
    w = 0
    frame_index = -1
    fps = 0.0
    flows = OrderedDict()

    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True
    asyncVideo_flag = False

    file_path = '/home/sohaibrabbani/Downloads/overhead_people_clips/baggageclaim.mp4'
    if asyncVideo_flag:
        video_capture = VideoCaptureAsync(file_path)
    else:
        video_capture = cv2.VideoCapture(file_path)

    if asyncVideo_flag:
        video_capture.start()

    if writeVideo_flag:
        if asyncVideo_flag:
            w = int(video_capture.cap.get(3))
            h = int(video_capture.cap.get(4))
        else:
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output_yolov3.avi', fourcc, video_capture.get(cv2.CAP_PROP_FPS), (w, h))
        # frame_index = -1

    fps_imutils = imutils.video.FPS().start()
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break

        t1 = time.time()

        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs = yolo.detect_image(image)[0]
        confidence = yolo.detect_image(image)[1]

        features = encoder(frame, boxs)

        detections = [Detection(bbox, confidence, feature) for bbox, confidence, feature in
                      zip(boxs, confidence, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 60:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

            # Initializing the flow list for saving the flow data of each person - Sohaib
            if track.track_id not in flows:
                flows[track.track_id] = OrderedDict()

            # Saving location of a person in a frame - Sohaib
            flows[track.track_id][frame_index + 1] = np.array([int(bbox[0]), int(bbox[1])])

        for det in detections:
            bbox = det.to_tlbr()
            score = "%.2f" % round(det.confidence * 100, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cv2.putText(frame, score + '%', (int(bbox[0]), int(bbox[3])), 0, 5e-3 * 130, (0, 255, 0), 2)

        cv2.imshow('', frame)

        if writeVideo_flag:  # and not asyncVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        fps_imutils.update()

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("FPS = %f" % (fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calling the serialize function to save the pickle file - Sohaib
    serialize_flow_data(flows=flows,
                        height=h,
                        width=w,
                        file_path=file_path,
                        frame_count=video_capture.get(cv2.CAP_PROP_FRAME_COUNT),
                        fps=video_capture.get(cv2.CAP_PROP_FPS))

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    if asyncVideo_flag:
        video_capture.stop()
    else:
        video_capture.release()

    if writeVideo_flag:
        out.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO())

# USAGE
# python unfolding_viz.py --input output/_bizlobby_flows.pickle

__author__ = "Ruben Acuna"

import numpy as np
import argparse
import cv2
import os
import pickle
import time
from collections import OrderedDict
from TrackingFlowObject import TrackingFlowObject


def render_flow(flow, color, frame):
    frames = len(flow)

    """
    time_start = time.time()
    heatmap = np.zeros((height, width, 1), np.single)
    #heatmap[15][15] = 1
    #heatmap[16][16] = 1
    #heatmap[17][17] = 1
    #heatmap[18][18] = 1
    #heatmap[19][19] = 1

    points = [np.array((1, 1)), np.array((10, 10))]

    whole = 1 / float(2)

    for x in range(0, 150):
        for y in range(0, 15):
            a = np.array((x, y))

            for point in points:
                dist = np.linalg.norm(a-point) + 0.001
                #factor = 1 - 1/dist

                if dist < .5:
                    heatmap[x][y] = whole
                else:
                    heatmap[x][y] += whole/dist
                #print("x:" + str(x) + ", y:" + str(y) + " heat:" + str(heatmap[x][y]))

    time_end = time.time()
    elapsed = (time_end - time_start)
    print("[INFO] single frame took {:.4f} seconds".format(elapsed))

    #for x in range(0, 150):
    #    for y in range(0, 15):
    #        frame[x][y] = tuple([int(c * heatmap[x][y]) for c in color_white])

    frame = color_white * heatmap
    """

    # compute heatmap for control points
    if FLOW_USE_DENSITY:
        whole = 1 / float(len(flow))
        step_heat = OrderedDict()

        for step in flow.keys():
            step_location = np.array(flow[step])
            heat = float(0)

            for otherstep in flow.keys():
                otherstep_location = np.array(flow[otherstep])
                dist = np.linalg.norm(step_location - otherstep_location)
                #if dist == 0:
                #    heat += whole
                #else:
                #    heat += whole/dist
                #if dist * dist > 1:
                #    heat += whole / (dist * dist)
                #else:
                #    heat += whole
                heat += (1 / whole) * (dist * dist)
            step_heat[step] = heat

        maxheat = step_heat[max(step_heat, key=step_heat.get)]
        for heat in step_heat.keys():
            norm = step_heat[heat] / maxheat

            if norm < 0:
                norm = 0.01
            elif norm > 1:
                norm = .99
            step_heat[heat] = norm

    # draw each step in flow
    for step in flow.keys():
        centroid = flow[step]

        if FLOW_USE_DENSITY:
            final_color = tuple([int(c * step_heat[step]) for c in color])
        else:
            final_color = color

        if FLOW_USE_INTERFRAME_APPROX:
            if step-1 not in flow:  # use circle method
                cv2.circle(frame, (int(centroid[0]), int(centroid[1])), int(FLOW_THICKNESS / 2), final_color, -1)

            else:
                centroid_last = flow[step-1]
                image = cv2.line(frame, (int(centroid_last[0]), int(centroid_last[1])), (int(centroid[0]), int(centroid[1])),
                                 final_color, FLOW_THICKNESS)
            pass
        else:  # use circle method
            cv2.circle(frame, (int(centroid[0]), int(centroid[1])), int(FLOW_THICKNESS / 2), final_color, -1)


##############################
# PARAMETERS
FLOW_THICKNESS = 16
FLOW_USE_INTERFRAME_APPROX = True

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input pickle")
args = vars(ap.parse_args())

# visualization related parameters
FLOW_THICKNESS = 8
FLOW_USE_INTERFRAME_APROX = True
FLOW_USE_DENSITY = True

# named colors
COLOR_WHITE = (255, 255, 255)

# load flow data
filename = args["input"]
with open(filename, 'rb') as handle:
    flow_data = pickle.load(handle)
    # flow_data = joblib.load(handle)

height = flow_data.input_height
width = flow_data.input_width
flows = flow_data.flows
fps = flow_data.fps
frame_count = flow_data.framecount

flow_count = len(flows)
flow_idx = list(range(flow_count))
viz_colors = [[int(c) for c in x] for x in np.random.randint(0, 255, size=(flow_count, 3), dtype="uint8")]

print("Person Flow Count:", flow_count)

cv2.namedWindow('visualization', cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("visualization", 0, 0)

key = -1

while True:
    if key == ord("q"):
        # filename = os.sep + os.path.basename(args["input"])[:-7] + "_unfolding.jpg"
        filename = '/home/sohaibrabbani/PycharmProjects/deep_sort_yolov3/' + "_unfolding.jpg"
        cv2.imwrite(filename, frame_viz)
        break

    elif key == 91:  # [
        FLOW_THICKNESS = max(FLOW_THICKNESS - 1, 1)
    elif key == 93:  # ]
        FLOW_THICKNESS += 1
    elif key == 100:  # d
        FLOW_USE_DENSITY = not FLOW_USE_DENSITY
    elif key == 108:  # l
        FLOW_USE_INTERFRAME_APPROX = not FLOW_USE_INTERFRAME_APPROX
    elif key != -1:
        print("key="+str(key))

    # render flows
    frame_viz = np.zeros((height, width, 3), np.uint8)

    for idx in flow_idx:
        render_flow(flows[idx], viz_colors[idx], frame_viz)

    cv2.imshow("visualization", frame_viz)

    key = cv2.waitKey(1)

cv2.destroyAllWindows()
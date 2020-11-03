import cv2
import numpy as np
import time

from collections import deque
import argparse
import numpy as np

count = 0

#####
# initialize the list of tracked points, the frame counter, and the coordinate deltas
counter = 0
(dX, dY) = (0, 0)
direction = ""
####

net = cv2.dnn.readNet('yolov3_training_2000.weights', 'yolov3_testing.cfg')

classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture('parkinglot.mp4')
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))
starting_time = time.time()
frame_id = 0
while True:
    _, img = cap.read()
    frame_id += 1
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)

                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.0001, 0.4)

    c = 0
    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            # if (count<3):
            #     print(x, y)
            #     count = count + 1
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
            cv2.putText(img, "x: " + str(x), (x, y-30), font, 2, (255, 255, 255), 2)
            cv2.putText(img, "y: " + str(y), (x, y), font, 2, (255, 255, 255), 2)
            c = c + 1

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(img, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)
    cv2.putText(img, "No of spaces: " + str(c), (10, 80), font, 4, (0, 0, 0), 3)
    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()
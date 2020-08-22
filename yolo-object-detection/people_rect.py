import numpy as np
import argparse
import time
import cv2
import os
import math

def load_yolo():
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet("yolo-coco/yolov3.cfg", "yolo-coco/yolov3.weights")
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # labels = open(labelsPath).read().strip().split("\n")
    return (net, ln)

CAMERA_INPUT = 0
DEFAULT_CONFIDENCE = 0.5
DEFAULT_THRESOLD = 0.3

def capture_image(camera=None):
    if camera is None:
        ## 800 x 1280 x 3
        camera = cv2.VideoCapture(CAMERA_INPUT)
    return_value, image = camera.read()
    return image

import time
if __name__ == "__main__":
    (net, ln) = load_yolo()
    
    image = capture_image()
    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
    swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            if classID != 0:
                continue;
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > DEFAULT_THRESOLD:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                print("centerX:%s, centerY:%s, width:%s, height:%s" % (centerX, centerY, width, height))

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    print("Found Boxes: ", boxes)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, DEFAULT_CONFIDENCE,
        DEFAULT_THRESOLD)
    # ensure at least one detection exists
    maxloc = (-1, -1)
    maxsize = (0, 0)
    if len(idxs) > 0:
        # loop over the indexes we are keeping        
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
        color = 0
        # draw a bounding box rectangle and label on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format("Person", 0)
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # return (x, y, w, h)

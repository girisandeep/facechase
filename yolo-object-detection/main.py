import numpy as np
import argparse
import time
import cv2
import os
import math
import serial

# 5v 5A SMPS

CAMERA_INPUT = 1
DEFAULT_CONFIDENCE = 0.5
DEFAULT_THRESOLD = 0.3
def capture_image(camera=None):
    if camera is None:
        ## 800 x 1280 x 3
        camera = cv2.VideoCapture(CAMERA_INPUT)
    return_value, image = camera.read()
    return image

def load_yolo():
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet("yolo-coco/yolov3.cfg", "yolo-coco/yolov3.weights")
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # labels = open(labelsPath).read().strip().split("\n")
    return (net, ln)

def get_biggest_person_bb(image, net, ln):
    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
    swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    # show timing information on YOLO
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
        maxdiag = 0
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            diag = math.sqrt(w*w + h*h)
            if diag > maxdiag:
                maxloc = (x, y)
                maxsize = (w, h)
                maxdiag = diag
        (x, y) = maxloc
        (w, h) = maxsize
        color = 0
        # draw a bounding box rectangle and label on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format("Person", 0)
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return (x, y, w, h)
    return None

def getSerial():
    return serial.Serial('/dev/tty.usbmodem1411', 9600)

def center(ser):
    angles = "%s %s\n" % (90, 90)
    ser.write(angles.encode())

import time
if __name__ == "__main__":
    ser = getSerial()
    (net, ln) = load_yolo()
    
    not_found_times = 0

    touching_top_times = 0
    touching_bottom_times = 0
    while True:
        image = capture_image()
        person_bb = get_biggest_person_bb(image, net, ln)
        if person_bb:
            (x, y, w, h) = person_bb
            Bcx = x + w/2
            Bcy = y + h/2

            print("person_bb: ", person_bb)
            MW = 1280
            MH = 800
            

            Cx = 1280/2
            Cy = 800/2

            deltax = Cx - Bcx
            deltay = Cy - Bcy
            print("Distances from Center: X: Y: " , deltax, deltay)
            alphax = deltax*60/1280
            alphay = deltay*60/800

            # 75 Diag
            # MAXA = 75
            # MD = math.sqrt(MH*MH + MW*MW)
            print("Absolute Angle H, V" , alphax, alphay)
            hangle = round(90+alphax)
            vangle = round(90-alphay)
            angles = "%s %s\n" % (hangle, vangle)
            print("Angles (before): ", angles)

            if y < 60:
                touching_top_times += 1
            
            if y > 400:
                touching_top_times = 0

            vangle -= 10*touching_top_times
            # vangle += 10*touching_bottom_times

            angles = "%s %s\n" % (hangle, vangle)
            print("Angles (after): ", angles)
            ser.write(angles.encode())
        else:
            if not_found_times > 5:
                angles = "%s %s\n" % (90, 90)
                print("Angles (after): ", angles)
                not_found_times = 0
                touching_top_times = 0
            not_found_times += 1
        
        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(5)
        # time.sleep(10)
        
        # cv2.destroyAllWindows()
        # ans = input("Do you want to continue? Y/N")
        # if ans.lower() == "n":
        #     break

# Create your views here.
from cProfile import label
from cgitb import html
from html.entities import html5
from inspect import getframeinfo
from multiprocessing.sharedctypes import Value
from os import name
from tkinter import Frame
from webbrowser import get
import cv2
import numpy as np
from django.shortcuts import render
from django.http import StreamingHttpResponse

def getCameraId():  #my os gives random video ids to wencams on every boot, tesaile loop lagayera check garya, idk whats alternative
    for i in range(0, 10):
        vd=cv2.VideoCapture(i,cv2.CAP_DSHOW)
        if vd.isOpened():
            return i
class VideoCamera(object):
    def __init__(self):
        id=getCameraId()
        self.video = cv2.VideoCapture(id,cv2.CAP_DSHOW)

    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        net = cv2.dnn.readNet('yolov4-sign_last.weights', 'yolov4-sign.cfg')
        classes = []
        with open('sign.names', 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        cap = self.video
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                img = frame
                height, width, n_channels = img.shape
                blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)
                class_ids = []
                boxes = []
                confidences = []
                for out in outs:
                    for det in out:
                        scores = det[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]

                        if confidence > 0.5:
                            cx = int(det[0] * width)
                            cy = int(det[1] * height)

                            w = int(det[2] * width)
                            h = int(det[3] * height)

                            x = int(cx - w / 2)
                            y = int(cy - h / 2)
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
                n_det = len(boxes)
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # removes boxes those are alike
                font = cv2.FONT_HERSHEY_PLAIN
                lab = []
                for i in range(n_det):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        color = colors[i]
                        cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)
                        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
                        text_inf = label    
                        lab.append(text_inf)          
                rets, jpegs = cv2.imencode('.jpg', img)
                return jpegs.tobytes()
            else:
                break
        cap.release()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(lab)
        return jpeg.tobytes(),lab

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def test(request):
    name = 
    return render(request,'test.html')

def video_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()),content_type='multipart/x-mixed-replace; boundary=frame')

def index(request):
    return render(request,'index.html')

def detect(request):
    return render(request,'detect.html')
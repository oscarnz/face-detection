from distutils.command.upload import upload
import streamlit as st
from PIL import ImageDraw, Image
from components.server import *
import ast
import cv2
import av
from components.url_endpoint import url, endpoint_facemask, endpoint_facemask_yolov5
import asyncio
from typing import List

#Labels for each object detection in facemask detection   
FACEMASK_LABELS = {
    0: "Without Mask",
    1: "Incorrect Mask",  
    2: "With Mask"
}

#Bounding box color code for each object detection in facemask detection
LABEL_COLORS = {
    "Without Mask": (0, 0, 255), # red
    "Incorrect Mask": (0, 255, 255), # yellow
    "With Mask": (0, 255, 0)  # green
}


def facemask_yolov5_detection():

    #Type of image: jpg, png
    uploaded_image = st.file_uploader("Upload Images", type=["jpg","png"]) 
    
    if uploaded_image is not None:
        #actual image
        img = load_image(uploaded_image) 

        output = process(uploaded_image, url+endpoint_facemask_yolov5)

        o = ast.literal_eval(output)
    
        LABEL_COUNT = {name : 0 for name, _ in LABEL_COLORS.items()}

        for i in range(len(o)):
            x1 = int(o[i]["x1"])
            y1 = int(o[i]["y1"])
            x2 = int(o[i]["x2"])
            y2 = int(o[i]["y2"])
            confidence = str(o[i]["confidence"])
            label = o[i]["label"]
            label_color = LABEL_COLORS[label]

            draw = ImageDraw.Draw(img)
            draw.rectangle([x1, y1, x2, y2], fill=None, outline=label_color, width = 3)
            draw.text([x1, y1-10], label + " " + confidence, fill=label_color, outline=label_color, stroke_width=3 )
            LABEL_COUNT[label] += 1 #numbers for each labels 
            print(LABEL_COUNT)

        #get the input image type, size, dimension and filename to output
        file_details = {"filename":uploaded_image.name, 
                        "filetype":uploaded_image.type,
                        "filesize":uploaded_image.size,
                        "dimension of original images": str(img.width) + "x" + str(img.height)}
        st.write(file_details)
        #display original images
        st.image(img,width=None) 

        #display metric - analysis
        st.header("Analysis")
        col1, col2, col3 = st.columns(3)
        col1.metric("Without Mask", LABEL_COUNT['Without Mask'])
        col2.metric("Incorrect Mask", LABEL_COUNT['Incorrect Mask'])
        col3.metric("With Mask", LABEL_COUNT['With Mask'])

def facemask_yolov5_detection_video():
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        frame1 = cv2.flip(frame, 1)
        
        frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        image = load_video(frame)
        output = process_video(image, url+endpoint_facemask_yolov5)
        o = ast.literal_eval(output)

        LABEL_COUNT = {name : 0 for name, _ in LABEL_COLORS.items()}
        
        for i in range(len(o)):
            x1 = int(o[i]["x1"])
            y1 = int(o[i]["y1"])
            x2 = int(o[i]["x2"])
            y2 = int(o[i]["y2"])
            confidence = str(o[i]["confidence"])
            label = o[i]["label"]
            label_color = LABEL_COLORS[label]

            cv2.rectangle(frame, [x1, y1], [x2, y2], color=label_color, thickness=5)
            cv2.putText(frame, label + " " + confidence, [x1, y1-10],  cv2.FONT_HERSHEY_COMPLEX, 1, label_color, 2)
            LABEL_COUNT[label] += 1 #numbers for each labels
        
        FRAME_WINDOW.image(frame)

    else:
        st.write('Stopped')

class VideoProcessor_facemask_yolov5:
    """Perform the operations on the video frame by frame (fps) for facemask detection
    """
    def recv(self, frame):
        fps = 100
        img = frame.to_ndarray(format="bgr24")
        
        img = cv2.flip(img, 1)

        image = load_video(img)

        output = process_video(image, url+endpoint_facemask_yolov5)
        #time.sleep(1/fps)

        o = ast.literal_eval(output)
        LABEL_COUNT = {name : 0 for name, _ in LABEL_COLORS.items()}
        
        for i in range(len(o)):
            x1 = int(o[i]["x1"])
            y1 = int(o[i]["y1"])
            x2 = int(o[i]["x2"])
            y2 = int(o[i]["y2"])
            confidence = str(o[i]["confidence"])
            label = o[i]["label"]
            label_color = LABEL_COLORS[label]
            
            cv2.rectangle(img, [x1, y1], [x2, y2], color=label_color, thickness=5)
            cv2.putText(img, label + " " + confidence, [x1, y1-10],  cv2.FONT_HERSHEY_COMPLEX, 1, label_color, 2)
            LABEL_COUNT[label] += 1 #numbers for each labels
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def facemask_yolov5_detection_video_2():
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        frame1 = cv2.flip(frame, 1)
        
        frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        
        image = load_video(frame)
        output = process_video(image, url+endpoint_facemask_yolov5)
        o = ast.literal_eval(output)

        LABEL_COUNT = {name : 0 for name, _ in LABEL_COLORS.items()}
        
        for i in range(len(o)):
            x1 = int(o[i]["x1"])
            y1 = int(o[i]["y1"])
            x2 = int(o[i]["x2"])
            y2 = int(o[i]["y2"])
            confidence = str(o[i]["confidence"])
            label = o[i]["label"]
            label_color = LABEL_COLORS[label]

            cv2.rectangle(frame, [x1, y1], [x2, y2], color=label_color, thickness=5)
            cv2.putText(frame, label + " " + confidence, [x1, y1-10],  cv2.FONT_HERSHEY_COMPLEX, 1, label_color, 2)
            LABEL_COUNT[label] += 1 #numbers for each labels
        
        FRAME_WINDOW.image(frame)

    else:
        st.write('Stopped')
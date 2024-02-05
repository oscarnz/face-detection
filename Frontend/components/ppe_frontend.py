from distutils.command.upload import upload
import streamlit as st
from PIL import ImageDraw, Image
from components.server import load_image,load_video, process_video
import ast
import cv2
import av
import time
import torch
from components.url_endpoint import url, endpoint_ppe

#Labels for each object detection in PPE detection   
PPE_LABELS = {
    0: "person",
    1: "vest",  
    2: "blue helmet",   
    3: "red helmet", 
    4: "white helmet",
    5: "yellow helmet"
}

#Bounding box color code for each object detection in PPE detection
LABEL_COLORS = {
    "person": (0, 255, 0),    # green
    "vest": (28, 0, 128),  # purple
    "blue helmet":(0,0,255),   # blue
    "red helmet": (55, 0, 0), # red
    "white helmet": (255, 255, 255), #white
    "yellow helmet": (255, 255, 0)  # yellow
}


def draw_image_with_boxes(image, objects_detected, x_ratio, y_ratio):
    """Draw bounding boxes for PPE detection

    Args:
        image (_type_): uploaded images
        objects_detected (_type_): coordinate for the detected object
        x_ratio (_type_): the scale ratio for x axis
        y_ratio (_type_): the scale ratio for y axis

    Returns:
        _type_: _description_
    """

    draw = ImageDraw.Draw(image) 
    LABEL_COUNT = {name : 0 for name, _ in LABEL_COLORS.items()}
   
    for object in objects_detected:
        x1 = int(object[0]) #x1
        y1 = int(object[1]) #y1
        x2 = int(object[2]) #x2
        y2 = int(object[3]) #y2
        label = object[5] #label for each class of ppe
        label_name = PPE_LABELS[int(label)]
        label_color = LABEL_COLORS[label_name]

        #Calculation for getting the size of original images for the bounding box.
        x = int(x1 * x_ratio)
        y = int(y1 * y_ratio)
        w = int(x2 * x_ratio)
        h = int(y2 * y_ratio)
       
        print(label, x, y, w, h)
        draw.rectangle([x, y, w, h], fill=None, outline=label_color)
        draw.text([x, y-10], label_name, fill=label_color, stroke_width=20)
        LABEL_COUNT[label_name] += 1 

    return image, LABEL_COUNT

def ppe_detection():
    #load model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', device="cpu") 
    #Type of image: jpg, png
    uploaded_image = st.file_uploader("Upload Images", type=["jpg","png"]) 
    

    if uploaded_image is not None:
        #actual image
        img = load_image(uploaded_image) 
        
        #resize image
        targetSize = 200
        y_percent = (targetSize) / float(img.size[1]) #height
        y_ratio = (float(img.size[1]) / targetSize) 
        x_resize = int((float(img.size[0]) * float(y_percent))) #width size
        x_ratio = (float(img.size[0]) / x_resize)
        image = img.resize((x_resize, targetSize), Image.Resampling.LANCZOS)
        #object detection on resize 
        output = model(image) 

        #bounding box
        img, label_count = draw_image_with_boxes((img), list(output.xyxy[0]), x_ratio, y_ratio) #bounding box on original image
        print(label_count)
        
        #get the input image type, size, dimension and filename to output
        file_details = {"filename":uploaded_image.name, 
                        "filetype":uploaded_image.type,
                        "filesize":uploaded_image.size,
                        "dimension of original images": str(img.width) + "x" + str(img.height),
                        "dimension of resize images": str(image.width) + "x" + str(image.height)}
        st.write(file_details)
        #display original images
        st.image(img,width=None) 

        #display metric - analysis
        st.header("Analysis")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Person", label_count['person'])
        col2.metric("Vest", label_count['vest'])
        col3.metric("Blue Helmet", label_count['blue helmet'])
        col4.metric("Red Helmet", label_count['red helmet'])
        col5.metric("White Helmet", label_count['white helmet'])
        col6.metric("Yellow Helmet", label_count['yellow helmet'])

def ppe_detection_video():
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        frame1 = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        image = load_video(frame)
        
        output = process_video(image, url+endpoint_ppe)
        print(output)
        #model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', device="cpu") 
        #output = model(image)
        o = ast.literal_eval(output)
        LABEL_COUNT = {name : 0 for name, _ in LABEL_COLORS.items()}
        
        for i in range(len(o)):
            x1 = int(o[i]["x1"])
            y1 = int(o[i]["y1"])
            x2 = int(o[i]["x2"])
            y2 = int(o[i]["y2"])
            label = o[i]["label"]
            #label_name = PPE_LABELS[int(label)]
            label_color = LABEL_COLORS[label]
            print(x1, y1, x2, y2)
            cv2.rectangle(frame, [x1, y1], [x2, y2], color=label_color, thickness=5)
            cv2.putText(frame, label, [x1, y1-10],  cv2.FONT_HERSHEY_COMPLEX, 1, label_color, 2)
            LABEL_COUNT[label] += 1 #numbers for each labels
        
        FRAME_WINDOW.image(frame)

    else:
        st.write('Stopped')

class VideoProcessor_ppe:
    """Perform the operations on the video frame by frame (fps) for PPE detection
    """
    def recv(self, frame):
        fps = 100
        img = frame.to_ndarray(format="bgr24")
        
        img = cv2.flip(img, 1)

        image = load_video(img)

        output = process_video(image, url+endpoint_ppe)
        time.sleep(1/fps)

        o = ast.literal_eval(output)
        LABEL_COUNT = {name : 0 for name, _ in LABEL_COLORS.items()}
        
        for i in range(len(o)):
            x1 = int(o[i]["x1"])
            y1 = int(o[i]["y1"])
            x2 = int(o[i]["x2"])
            y2 = int(o[i]["y2"])
            label = o[i]["label"]
            label_color = LABEL_COLORS[label]
            print(x1, y1, x2, y2)
            cv2.rectangle(img, [x1, y1], [x2, y2], color=label_color, thickness=5)
            cv2.putText(img, label, [x1, y1-10],  cv2.FONT_HERSHEY_COMPLEX, 1, label_color, 2)
            LABEL_COUNT[label] += 1 #numbers for each labels
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")



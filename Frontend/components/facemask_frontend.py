import streamlit as st
from PIL import ImageDraw, Image
from components.server import load_image, process, load_video, process_video
import ast
import cv2
import av
from threading import Timer
import time
from components.url_endpoint import url, endpoint_facemask

def facemask(): 
    """To predict if the person wearing mask, not wearing mask or incorrectly wearing mask by uploading the images to the Streamlit
    """
    uploaded_image = st.file_uploader("Upload Images", type=["jpg"])

    if uploaded_image is not None:
        #Feed the uploaded image into var image
        image = load_image(uploaded_image)

        #Pass the uploaded image and api endpoint url into process function to perform post request from backend.
        #Process will return json data object in string.
        output = process(uploaded_image, url+endpoint_facemask)

        #The ast.literal_eval will convert the string into dictionary which is valid datatype for json
        o = ast.literal_eval(output)
        
        #Count for each label of object detection
        maskCount = 0
        nomaskCount = 0
        incorrectMaskCount = 0

        for i in range(len(o)):
            x = int(o[i]["x0"])
            y = int(o[i]["y0"])
            w = int(o[i]["x1"])
            h = int(o[i]["y1"])
            label = o[i]["label"]

            draw = ImageDraw.Draw(image)
            print(output)
            print(label)
            
            #Draw a rectangle based on the coordinates extracted.
            if label == "Mask":
                draw.rectangle([x, y, w, h], fill=None, outline="green", width = 3)
                draw.text([x, y-10], "Mask", fill="green", stroke_width=10 )
                maskCount = maskCount+1
            elif label == "Incorrect Mask":
                draw.rectangle([x, y, w, h], fill=None, outline="yellow", width = 3)
                draw.text([x, y-10], "Incorrect Mask", fill="yellow", stroke_width=10 )
                incorrectMaskCount = incorrectMaskCount+1
            else:
                draw.rectangle([x, y, w, h], fill=None, outline="red", width = 3)
                draw.text([x, y-10], "No Mask", fill="red", stroke_width=10 )
                nomaskCount = nomaskCount+1
            
        #Get the input image type, size, dimension and filename to output
        file_details = {"filename":uploaded_image.name,
                        "filetype":uploaded_image.type,
                        "filesize":uploaded_image.size,
                        "dimension": str(image.width) + "x" + str(image.height)}
        st.write(file_details)
        st.image(image,width=None)
       
       
        #Analysis of the prediction which consist of total of faces, number of faces with mask and without mask 
        st.header("Analysis")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Detected", maskCount+nomaskCount+incorrectMaskCount)
        col2.metric("With masks", maskCount)
        col3.metric("Without masks", nomaskCount)
        col4.metric("Incorrect mask", incorrectMaskCount)
   

def facemask_video():
    """To predict if the person wearing mask, not wearing mask or incorrectly wearing mask by using the video stream in the Streamlit    """
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        frame1 = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        image = load_video(frame)
        output = process_video(image, url+endpoint_facemask)
        o = ast.literal_eval(output)

        for i in range(len(o)):
            x = int(o[i]["x0"])
            y = int(o[i]["y0"])
            w = int(o[i]["x1"])
            h = int(o[i]["y1"])
            label = o[i]["label"]
            
            #Draw a rectangle based on the coordinates extracted
            if label == "Mask":
                cv2.rectangle(frame, (x, y), (w, h), (0, 128, 0), 3)
                cv2.putText(frame, 'Have Mask', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 0), 2)
            elif label == "Incorrect Mask":
                cv2.rectangle(frame, (x, y), (w, h), (255, 255, 0), 3)
                cv2.putText(frame, 'Incorrect Mask', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            else :
                cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 3)
                cv2.putText(frame, 'No Mask', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        FRAME_WINDOW.image(frame)

    else:
        st.write('Stopped')

class VideoProcessor:
    """Perform the operations on the video frame by frame (fps) for face mask detection
    """
    def recv(self, frame):
        fps = 100
        img = frame.to_ndarray(format="bgr24")
        
        img = cv2.flip(img, 1)

        image = load_video(img)

        output = process_video(image, url+endpoint_facemask)
        time.sleep(1/fps)

        o = ast.literal_eval(output)
        for i in range(len(o)):
            x = int(o[i]["x0"])
            y = int(o[i]["y0"])
            w = int(o[i]["x1"])
            h = int(o[i]["y1"])
            label = o[i]["label"]

            #Draw a rectangle based on the coordinates extracted
            if label == "Mask":
                print(x, y, w, h)
                cv2.rectangle(img, (x, y), (w, h), (0, 128, 0), 3)
                cv2.putText(img, 'Have Mask', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 0), 2)
            elif label == "Incorrect Mask":
                cv2.rectangle(img, (x, y), (w, h), (255, 255, 0), 3)
                cv2.putText(img, 'Incorrect Mask', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            else :
                cv2.rectangle(img, (x, y), (w, h), (255, 0, 0), 3)
                cv2.putText(img, 'No Mask', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        
        return av.VideoFrame.from_ndarray(img, format="bgr24")



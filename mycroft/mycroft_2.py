import streamlit as st
from server import load_video, process_video
import ast
import cv2
from threading import Lock, Thread
from url_endpoint import url, endpoint_facemask_yolov5
from time import sleep, time
import json
import websockets

class Camera:
    last_frame = None
    last_ready = None
    lock = Lock()

    def __init__(self, rtsp_link):
        self.capture = cv2.VideoCapture(rtsp_link, cv2.CAP_FFMPEG)
#        pipline_r = f'rtspsrc location={rtsp_link} latency=100 '
#                    f'! rtph264depay ! h264parse ! v4l2h264dec '
#                    f'capture-io-mode=4 ! v4l2video12convert '
#                    f'output-io-mode=5 capture-io-mode=4 ! appsink '
#                    f'sync=false'
#        self.capture = cv2.VideoCapture(pipline_r)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.thread = Thread(target=self.rtsp_cam_buffer,
                             args=(self.capture,), name="rtsp_read_thread")
        self.thread.daemon = True
        self.thread.start()

    def rtsp_cam_buffer(self, capture):
        while True:
            with self.lock:
                self.last_ready, self.last_frame = capture.read()

    def getFrame(self):
        if (self.last_ready is not None) and (self.last_frame is not None):
            return self.last_frame.copy()
        else:
            return None

    def release(self):
        self.capture.release()

async def mycroft2():
    fps = 100
    #run = st.checkbox('Run')
    placeholder = st.empty()
    with placeholder.container():
        column1, column2, column3 = st.columns(3)
        column1.metric("Not Wearing Mask", "-")
        column2.metric("Incorrect Mask", "-")
        column3.metric("Wearing Mask", "-")
        FRAME_WINDOW = st.image([])

    camera = Camera('rtsp://agora402.klif.slb.com:8554/proxied')
    column1, column2 = st.columns(2)
    start_time = 0
    while True:
        if time() - start_time >= 1/fps:
            frame = camera.getFrame()
            while frame is None:
                frame = camera.getFrame()
            #frame1 = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            FRAME_WINDOW.image(frame)


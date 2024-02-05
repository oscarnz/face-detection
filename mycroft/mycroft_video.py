import streamlit as st
from server import load_image, process, load_video, process_video
import ast
import cv2
from threading import Lock, Thread
from url_endpoint import url, endpoint_facemask
from time import sleep, time
import json
import websockets
import requests

from mv_extractor import VideoCap

from ip_websocket import VideoCapture

IP = "163.184.4.15"
WS_URL = f"ws://{IP}/"
URL = f'http://{IP}/api.html?n=login'
WIDTH, HEIGHT = 1920, 1080
ADDRESS = 'rtsp://agora402.klif.slb.com:8554/proxied?tcp'
# ADDRESS = 'rtsp://192.168.3.4:554/0'
MODE = 'rtsp'

# width = 1920
width = 1280
# height = 1080
height = 720

box_X1, box_Y1 = int(width/4), int(height/4)
box_X2, box_Y2 = int(width/4*3), int(height/4*3)

class Camera:
    last_frame = None
    last_ready = None
    motion_vectors = None
    frame_type = None
    timestamp = None
    lock = Lock()

    def __init__(self, address, mode='rtsp'):
        if mode=='rtsp':
            # self.capture = VideoCap()
            self.capture = cv2.VideoCapture(address, cv2.CAP_FFMPEG)
            self.last_ready = self.capture.open(address)
            if not self.last_ready:
                raise RuntimeError(f"Could not open {address}")
    #        pipline_r = f'rtspsrc location={rtsp_link} latency=100 '
    #                    f'! rtph264depay ! h264parse ! v4l2h264dec '
    #                    f'capture-io-mode=4 ! v4l2video12convert '
    #                    f'output-io-mode=5 capture-io-mode=4 ! appsink '
    #                    f'sync=false'
    #        self.capture = cv2.VideoCapture(pipline_r)
            # self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            # self.capture.set(cv2.CAP_PROP_POS_FRAMES, 30)
            self.thread = Thread(target=self.rtsp_cam_buffer,
                                args=(), name="rtsp_read_thread")
                                # args=(self.capture,), name="rtsp_read_thread")
        else:
            payload = {
                "api": "login",
                "data": {
                    "username": "admin",
                    "password": "21232f297a57a5a743894a0e4a801fc3"
                    }
                }
            self.capture = VideoCapture(URL, payload)
            self.last_ready = self.capture.open(address)
            if not self.last_ready:
                raise RuntimeError(f"Could not open {address}")
    #        pipline_r = f'rtspsrc location={rtsp_link} latency=100 '
    #                    f'! rtph264depay ! h264parse ! v4l2h264dec '
    #                    f'capture-io-mode=4 ! v4l2video12convert '
    #                    f'output-io-mode=5 capture-io-mode=4 ! appsink '
    #                    f'sync=false'
    #        self.capture = cv2.VideoCapture(pipline_r)
            # self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            self.thread = Thread(target=self.websocket_buffer,
                                args=(self.capture,), name="ws_read_thread")
        self.thread.daemon = True
        self.thread.start()

    def websocket_buffer(self, payload):
        while True:
            with self.lock:
                # self.last_ready, self.last_frame = capture.read()
                self.last_ready, self.last_frame, self.motion_vectors, \
                self.frame_type, self.timestamp = self.capture.read()        

    def rtsp_cam_buffer(self):
        while True:
            with self.lock:
                self.last_ready, self.last_frame = self.capture.read()
                # self.last_ready, self.last_frame, self.motion_vectors, \
                # self.frame_type, self.timestamp = self.capture.read()


    def getFrame(self):
        if (self.last_ready is not None) and (self.last_frame is not None):
            return self.last_frame.copy()
        else:
            return None

    def release(self):
        self.capture.release()


def calculateIntersection(a0, a1, b0, b1):
    if a0 >= b0 and a1 <= b1:  # Contained
        intersection = a1 - a0
    elif a0 < b0 and a1 > b1:  # Contains
        intersection = b1 - b0
    elif a0 < b0 and a1 > b0:  # Intersects right
        intersection = a1 - b0
    elif a1 > b1 and a0 < b1:  # Intersects left
        intersection = b1 - a0
    else:  # No intersection (either side)
        intersection = 0

    return intersection


def get_xy(object):
    x0 = int(object["x0"])
    y0 = int(object["y0"])
    x1 = int(object["x1"])
    y1 = int(object["y1"])
    return x0, y0, x1, y1


async def mycroft333():
    fps = 1
    run = st.checkbox('Run')
    placeholder = st.empty()
    with placeholder.container():
        column1, column2 = st.columns(2)
        column1.metric("Not Wearing Mask", "-")
        column2.metric("Wearing Mask", "-")
        FRAME_WINDOW = st.image([])

    camera = Camera(ADDRESS, MODE)
    column1, column2 = st.columns(2)
    start_time = 0
    mycroft_timer = -10000
    frame = camera.getFrame()
    while frame is None:
        frame = camera.getFrame()
    while True:
        if time() - start_time >= 1/fps or True:
            captured_frame = camera.getFrame()
            # while frame is None:
                # frame = camera.getFrame()
            if captured_frame is not None:
                frame = captured_frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            start_time = time()
        else:
            sleep((1/fps))


        cv2.rectangle(frame,
                      (box_X1, box_Y1),
                      (box_X2, box_Y2), (255, 0, 0), 3)
        wearing_mask_count = 0
        no_mask_count = 0
        if run:
            detecting_frame = frame[int(box_Y1):int(box_Y2),
                                    int(box_X1):int(box_X2)]

            image = load_video(detecting_frame)
            output = process_video(image, url + endpoint_facemask)
            output = ast.literal_eval(output)
            filtered_output = filter_overlap(output)
            for index in range(len(filtered_output)):
                x, y, w, h = get_xy(filtered_output[index])
                label = filtered_output[index]["label"]

                # Draw a rectangle based on the coordinates extracted
                wearing_mask_count, \
                    no_mask_count, \
                    detecting_frame = draw_frame_and_count(wearing_mask_count,
                                                           no_mask_count,
                                                           detecting_frame,
                                                           x, y, w, h,
                                                           label)
            frame[int(box_Y1):int(box_Y2),
                  int(box_X1):int(box_X2)] = detecting_frame
            cv2.rectangle(frame,
                          (box_X1, box_Y1),
                          (box_X2, box_Y2), (255, 0, 0), 3)
        with placeholder.container():
            column1, column2 = st.columns(2)
            column1.metric("Not Wearing Mask", no_mask_count)
            column2.metric("Wearing Mask", wearing_mask_count)
            FRAME_WINDOW.image(frame)
        if no_mask_count > 0 and time() - mycroft_timer > 10:
            ws = 'ws://poc-api.klif.slb.com:8181/core'
            persons = 'is 1 person' if no_mask_count == 1 else \
                f'are {no_mask_count} persons'
            print(f'There {persons} not wearing mask')
            # await mycroft_says(ws, f'There {persons} not wearing mask')
            mycroft_timer = time()


def filter_overlap(output):
    filtered_output = [output[0]] if len(output) > 1 else output
    if len(output) > 1:
        for index in range(1, len(output)):
            x0, y0, x1, y1 = get_xy(filtered_output[-1])
            X0, Y0, X1, Y1 = get_xy(output[index])
            width = calculateIntersection(x0, x1, X0, X1)
            height = calculateIntersection(y0, y1, Y0, Y1)
            current_area = abs(x0-x1)*abs(y0-y1)
            overlapped_area = width * height
            if 0.5*current_area > overlapped_area:
                filtered_output += [output[index]]
    return filtered_output


def draw_frame_and_count(wearing_mask_count, no_mask_count,
                         frame, x, y, w, h, label):
    if label == "Mask" or label == "Incorrect Mask":
        cv2.rectangle(frame, (x, y), (w, h),
                      (0, 128, 0), 3)
        cv2.putText(frame, 'Have Mask', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 0), 2)
        wearing_mask_count += 1
    elif label == "Incorrect Mask":
        cv2.rectangle(frame, (x, y), (w, h),
                      (255, 255, 0), 3)
        cv2.putText(frame, 'Incorrect Mask', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 0), 2)
        wearing_mask_count += 1
    else:
        cv2.rectangle(frame, (x, y), (w, h),
                      (255, 0, 0), 3)
        cv2.putText(frame, 'No Mask', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        no_mask_count += 1
    return wearing_mask_count, no_mask_count, frame


async def let_mycroft_speak():
    ws = 'ws://poc-api.klif.slb.com:8181/core'
    txt = st.text_area('Ask myCroft to say', '')
    if st.button('Send to myCroft'):
        message = {"type": "recognizer_loop:utterance",
                   "data": {"utterances": [f"say {txt}"],
                            "lang": "en-us"},
                   "context": {"client_name": "mycroft_cli",
                               "source": "debug_cli",
                               "destination": ["skills"]}}
        await send_to_mycroft(ws, json.dumps(message))


async def mycroft_says(ws, text):
    message = {"type": "recognizer_loop:utterance",
               "data": {"utterances": [f"say {text}"],
                        "lang": "en-us"},
               "context": {"client_name": "mycroft_cli",
                           "source": "debug_cli",
                           "destination": ["skills"]}}
    await send_to_mycroft(ws, json.dumps(message))


async def send_to_mycroft(ws, message):
    async with websockets.connect(ws) as websocket:
        await websocket.send(message)


def place_holder():
    placeholder = st.empty()
    for seconds in range(60):
        with placeholder.container():
            kpi1, _ = st.columns(2)
            txt = f"Hourglass with flowing sand {seconds} seconds have passed"
            kpi1.write(txt)
            sleep(1)
        placeholder.empty()

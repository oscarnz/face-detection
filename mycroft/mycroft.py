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
    x0 = int(object["x1"])
    y0 = int(object["y1"])
    x1 = int(object["x2"])
    y1 = int(object["y2"])
    return x0, y0, x1, y1


async def mycroft():
    fps = 15
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
    mycroft_timer = -10000
    while True:
        if time() - start_time >= 1/fps:
            frame = camera.getFrame()
            while frame is None:
                frame = camera.getFrame()
            #frame1 = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            start_time = time()
        else:
            sleep((1/fps))

        height, width, _ = frame.shape
        cv2.rectangle(frame,
                      (int(width/5), int(height/5)),
                      (int(width/5*4), int(height/5*4)), (255, 0, 0), 3)
        wearing_mask_count = 0
        no_mask_count = 0
        incorrect_mask_count = 0
        #if run:
        detecting_frame = frame[int(height/5):int(height/5*4),
                                int(width/5):int(width/5*4)]

        image = load_video(detecting_frame)
        output = process_video(image, url + endpoint_facemask_yolov5)
        output = ast.literal_eval(output)

        filtered_output = filter_overlap(output)
        for index in range(len(filtered_output)):
            x, y, w, h = get_xy(filtered_output[index])
            label = filtered_output[index]["label"]

            # Draw a rectangle based on the coordinates extracted
            wearing_mask_count, \
                no_mask_count, \
                 incorrect_mask_count, \
                    detecting_frame = draw_frame_and_count(wearing_mask_count,
                                                        no_mask_count,
                                                        incorrect_mask_count,
                                                        detecting_frame,
                                                        x, y, w, h,
                                                        label)

        with placeholder.container():
            column1, column2, column3 = st.columns(3)
            column1.metric("Not Wearing Mask", no_mask_count)
            column2.metric("Incorrect Mask", incorrect_mask_count)
            column3.metric("Wearing Mask", wearing_mask_count)
            FRAME_WINDOW.image(frame)
        if no_mask_count > 0 and time() - mycroft_timer > 10:
            ws = 'ws://poc-api.klif.slb.com:8181/core'
            persons = 'is 1 person' if no_mask_count == 1 else \
                f'are {no_mask_count} persons'
            print(f'There {persons} not wearing mask')
            await mycroft_says(ws, f'There {persons} not wearing mask')
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


def draw_frame_and_count(wearing_mask_count, no_mask_count, incorrect_mask_count,
                         frame, x, y, w, h, label):
    if label == "With Mask":
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
        incorrect_mask_count += 1
    else:
        cv2.rectangle(frame, (x, y), (w, h),
                      (255, 0, 0), 3)
        cv2.putText(frame, 'No Mask', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        no_mask_count += 1
    return wearing_mask_count, no_mask_count, incorrect_mask_count, frame


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

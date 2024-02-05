from cv2 import log
import requests
import websocket
import _thread
import tempfile
from mv_extractor import VideoCap
import logging
import cv2
from threading import Lock, Thread

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

# Define WebSocket callback functions
# def ws_message(ws, message):
#     print("WebSocket thread: %s" % message)

# def ws_open(ws):
#     ws.send('{"event":"subscribe", "subscription":{"name":"trade"}, "pair":["XBT/USD","XRP/USD"]}')

# def ws_thread(*args):
#     ws = websocket.WebSocketApp("wss://ws.kraken.com/", on_open = ws_open, on_message = ws_message)
#     ws.run_forever()

# Start a new thread for the WebSocket interface
# _thread.start_new_thread(ws_thread, ())

class VideoCapture:
    ws = None
    last_ready = None
    lock = Lock()

    def __init__(self, URL, payload, websocket_url) -> None:
        self.file = tempfile.NamedTemporaryFile()
        logging.info(self.file.name)
        self.capture = VideoCap()
        self.connect(URL, payload, websocket_url)
        # if self.last_ready:
        # self.run_forever()
        self.ws.run_forever()
        self.thread = Thread(target=self.run_forever,
                            args=(self.capture,), name="ws_read_thread")
        self.thread.daemon = True
        self.thread.start()
        logging.info('Thread started')

    def connect(self, URL, payload, websocket_url):
        self.session = requests.session()
        self.post_result = self.session.post(url=URL, json=payload)
        self.session_id = self.session.cookies.get_dict()['sessionID']
        self.ws_cookie = f'{{"index": "0", "sessionID": "{self.session_id}"}}'
        logging.debug(self.ws_cookie)
        self.ws = websocket.WebSocketApp(websocket_url ,on_open=self.open,
                                         on_message=self.message
                                         )
        # self.run_forever()

    def open(self, ws):
        logging.info('Connected')
        ws.send(self.ws_cookie)
        logging.info('Request sent')
    
    def message(self, ws, message):
        if len(message) >= 0:
            logging.info('Getting message')
            with open(self.file.name, 'wb') as stream:
                stream.write(message)
                logging.debug(len(message))
            self.read()
            if self.last_ready:
                logging.info('Saving frame...')
                cv2.imwrite('out.jpg', self.last_frame)
        
    def read(self):
        self.last_ready = self.capture.open(self.file.name)
        if self.last_ready:
            self.last_ready, self.last_frame, self.motion_vectors, \
            self.frame_type, self.timestamp = self.capture.read()
        self.capture.release()
    
    def getFrame(self):
        if (self.last_ready is not None) and (self.last_frame is not None):
            return self.last_frame.copy()
        else:
            return None

    def run_forever(self, *args):
        with self.lock:
            logging.info('Running forever')
            self.ws.run_forever()

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from components.facemask_frontend import VideoProcessor
from components.ppe_frontend import VideoProcessor_ppe
from components.facemask_yolov5_frontend import VideoProcessor_facemask_yolov5

RTC_CONFIGURATION = RTCConfiguration(
    {"RTCIceServer": [{
        "urls": ["turn:agora402.klif.slb.com:5349"],
        "username": "username1",
        "credential": "password1"
        },
    ]}
)
"""webrtc configuration for agora gateway
"""

def web_rtc():
    webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)

def web_rtc_ppe():
    webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor_ppe,
    async_processing=True,
)

def web_rtc_facemask_yolov5():
    webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor_facemask_yolov5,
    async_processing=True,
)

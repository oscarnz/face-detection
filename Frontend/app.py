import streamlit as st
from components.facemask_frontend import *
from components.ppe_frontend import *
from components.webrtc_frontend import *
from components.facemask_yolov5_frontend import *
from mycroft.mycroft import *


def main():
    """Streamlit user interface
    """
    st.header("Virtual Reception")
    pages = {
        # "Placeholder": place_holder,
        # "Send to myCroft": let_mycroft_speak,
        # "MyCroft": mycroft,
        "Video detection test": facemask_yolov5_detection_video_2, 
        "Facemask Image Classifier": facemask,
        "Facemask Detection": facemask_video,
        "Facemask detection + webrtc": web_rtc,
        "Facemask Detection - YOLOv5": facemask_yolov5_detection,
        "Facemask Video Detetection - YOLOv5": facemask_yolov5_detection_video,
        "Facemask Detection + webrtc - YOLOv5": web_rtc_facemask_yolov5,
        "PPE Image Classifier": ppe_detection,
        "PPE Detection Video Stream": ppe_detection_video,
        "PPE Detection + webrtc": web_rtc_ppe,
    }
    page_titles = pages.keys()

    page_title = st.sidebar.selectbox(
        "Choose the app mode",
        page_titles,
    )
    st.subheader(page_title)

    page_func = pages[page_title]
    page_func()

main()

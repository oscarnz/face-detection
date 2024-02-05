import streamlit as st
from components.facemask_frontend import facemask, facemask_video
from components.ppe_frontend import ppe_detection, ppe_detection_video
from components.webrtc_frontend import web_rtc, web_rtc_ppe
from mycroft.mycroft import mycroft, place_holder, let_mycroft_speak
import asyncio


async def main():
    """Streamlit user interface
    """
    st.header("Virtual Reception")
    pages = {
        "Placeholder": place_holder,
        "Send to myCroft": let_mycroft_speak,
        "MyCroft": mycroft,
        "Facemask Image Classifier": facemask,
        "Facemask Detection": facemask_video,
        "Facemask detection + webrtc": web_rtc,
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
    await page_func()


if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())

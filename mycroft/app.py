import streamlit as st
from mycroft import mycroft, place_holder, let_mycroft_speak
from mycroft_video import mycroft333
from mycroft_2 import mycroft2
import asyncio

async def main():
    """Streamlit user interface
    """
    st.header("Virtual Reception")
    pages = {
        "Placeholder": place_holder,
        "Send to myCroft": let_mycroft_speak,
        "MyCroft": mycroft,
        "Mycroft2" : mycroft2,
        "Mycroft333" : mycroft333
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

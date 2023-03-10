import streamlit as st
from PIL import Image

st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")

img_logo  = Image.open("images/TTZ_Günzburg_Bild.png")
img_IOT = Image.open("images/IOT_Bild.png")
img_VICE = Image.open("images/VACE_Bild.jpeg")

with st.container():
    st.title("Hi I am Lukas :wave:")
    st.header("A student at HNU currently working at TTZ Günzburg")
    st.write("This app was created using Streamlit")

st.write("[Learn More about Streamlit>](https://docs.streamlit.io/)")

with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("What we do at TTZ Günzburg")
        st.write("##")
        st.write(
            """
            The Technology Transfer Center (TTZ) in Günzburg has the task of carrying out research and development projects in cooperation with the regional economy.
            The TTZ thus fulfills an important task within the framework of the research, technology and innovation strategy of the Bavarian state government.
            The Technology Transfer Center is a branch of the Neu-Ulm University of Applied Sciences and has been in existence since the fall of 2020.
            The TTZ focuses on the topics of data analytics and artificial intelligence.

            """
        )
        st.write("[Learn more about the TTZ >](https://www.hnu.de/forschung/institute-kompetenzzentren/technologie-transfer-zentrum-ttz-guenzburg)")
    with right_column:
        st.image(img_logo)

with st.container():
    st.write("---")
    st.header("Our Projects")
    st.write("##")
    image_column, text_column = st.columns((1,2))
    with image_column:
        st.image(img_IOT)
    with text_column:
        st.subheader("IoT Smart Room Monitoring")
        st.write(
            """
            Using the Smart Room Monitoring solution as an application example, it is shown how a networked Internet of Things architecture can be created
            for a wide variety of use cases using standard single-board computers (Raspberry Pi), high-quality sensor technology and open-source technologies
            (Grafana, MQTT, Python).Furthermore, algorithms are used for intelligent ventilation recommendation. These take into account indoor air quality, 
            the external environment and change dynamics (depending on room size, number of windows, number of people in the room, etc.).
            """
        )
        st.markdown("[Learn more>](https://www.hnu.de/forschung/institute-kompetenzzentren/technologie-transfer-zentrum-ttz-guenzburg/projekte-des-ttz-guenzburg/iot-smart-room-monitoring)")

with st.container():
    image_column, text_column = st.columns((1,2))
    with image_column:
        st.image(img_VICE)
    with text_column:
        st.subheader("VACE - Emotions in speech")
        st.write(
            """
            Develop, train and test an artificial intelligence that detects the smallest changes in human speech.
            The conclusions that can be drawn from this are used to optimize marketing strategies and, in the long term,
            can even be used for the early detection of Alzheimer's disease.
            How can emotions in speech be detected, evaluated and analyzed using AI? The research project was initiated 
            because examining speech in terms of emotional state not only increases customer satisfaction,
            but also enables more targeted use, for example in marketing, while at the same time relieving the burden on customer service.
            """
        )
        st.markdown("[Learn more>](https://www.hnu.de/forschung/institute-kompetenzzentren/technologie-transfer-zentrum-ttz-guenzburg/projekte-des-ttz-guenzburg/vace-emotions-in-speech)")

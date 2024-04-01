import streamlit as st
import langchain_helper as lch
import textwrap
from dotenv import load_dotenv

load_dotenv()


st.title("YouTube Assistant")

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.sidebar.text_area(
            label="What is the YouTube video URL?",
            max_chars=500,
            key="youtube_url"
            )
        user_query = st.sidebar.text_area(
            label="Ask me about the video?",
            max_chars=500,
            key="user_query"
            )
        submit_button = st.form_submit_button(label='Submit')

if user_query and youtube_url:
    db = lch.create_vector_db_from_youtube_url(youtube_url)
    response = lch.get_response_from_query(db, user_query)
    st.subheader("Answer:")
    st.text(textwrap.fill(response, width=85))
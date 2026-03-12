from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
import base64

def get_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg = get_base64("GenAi_image.png")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg}");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

load_dotenv() # OpenAI API key load
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
st.header('Research tool')

user_input = st.text_input('Enter your prompt:')

if st.button('Submit'):
    result = model.invoke(user_input)
    st.write(result.content)
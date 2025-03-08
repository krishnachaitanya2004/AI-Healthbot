import streamlit as st
from model import ask
import time

st.title("Health Chatbot 💪🩺")
st.write("Ask me any health-related question!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask Health Chatbot"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner('Generating response...'):
        response = ask(prompt)
        print(response)
    with st.chat_message("assistant"):
        st.markdown(response)      
        st.session_state.messages.append({"role": "assistant", "content": response})
import streamlit as st
from openai import AzureOpenAI

import os
import sys
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), "..")))
from src.toolbox.data import Data


api_base = "https://oai-esteban.openai.azure.com/"
api_version = "2023-05-15"
api_key = "bf96c6fc58cb4222b614f4a82ef99bea"

openai_client = AzureOpenAI(
    api_key=api_key,  
    api_version=api_version,
    azure_endpoint=api_base
    )

if "chat_history" not in st.session_state:
    with open("./contexts/ts_context.txt", "r") as file:
        contents = file.read()
        train = Data(path="../train.csv").read()
        fcast = Data(path="../src/data/fcast.csv").read()
        train_data = train.to_dict("records")
        fcast_data = fcast.to_dict("records")
        contents = contents.replace("REPLACE_TS", str(train_data))
        contents = contents.replace("REPLACE_FCAST", str(fcast_data))
    st.session_state["chat_history"] = [{"role": "system", "content": contents}]

def generate_response(user_input):
    user_msg = {"role": "user", "content": user_input}
    st.session_state["chat_history"].append(user_msg)

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=st.session_state["chat_history"],
        temperature=.7,
        )
    
    resp = response.choices[0].message.content.strip()
    assistant_resp = {"role": "assistant", "content": resp}
    st.session_state["chat_history"].append(assistant_resp)
    return resp

st.title("Chat with GPT")
user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:        
        response = generate_response(user_input=user_input)
        
st.write("## Conversation")
for chat in st.session_state["chat_history"]:
    if chat["role"] != "system":
        st.write(f"**{chat['role']}**: {chat['content']}")

if st.button("Clear Chat"):
    st.session_state["chat_history"] = []

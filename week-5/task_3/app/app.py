from langsmith import traceable
import streamlit as st
from PIL import Image

from models.openai_client import OpenAIClient
from models.gemini_client import GeminiClient
from models.llama import LLamaModel
from utils.helpers import read_config, encode_image


config = read_config("config.yaml")

st.set_page_config(
    page_title="Finance Assistant Demo - LLM Course 2024",
    page_icon="🚀",
    layout= "wide",
    )

st.title("Finance Assistant Demo with Agentic RAG - LLM Course 2024 ✨")

########### Side Bar  ##################
use_tools = st.sidebar.checkbox("Enable Function Call")
model = st.sidebar.radio(
    "Please choose the AI model.",
    ["openai", "gemini", "llama"],
    captions = ["gpt-4o-mini", "gemini-1.5-flash", "meta-llama/Llama-3.1-8B-Instruct"]
)

if st.sidebar.button('Clear Chat'):
        st.session_state.messages = [{"role": "system", "content": config["system_prompt"]}]
        st.rerun()
############################################    

@traceable
def generate_response(model, use_tools: bool = False):
    if model == "openai":
        client = OpenAIClient(**config["openai"])
        st.session_state.messages[0]["content"] = config["openai"]["system_prompt"]
        return client.generate(st.session_state.messages, use_tools)
    elif model == "gemini":
        client = GeminiClient(**config["gemini"])
        messages=st.session_state.messages
        return messages, client.generate(messages)
    elif model == "llama":
        client = LLamaModel(**config["llama"])
        messages=st.session_state.messages
        return messages, client.generate(messages)

     
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [ {"role": "system", "content": config["system_prompt"]}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if not isinstance(message, dict):
        continue
    if message["role"] == "user" or message["role"] == "assistant":
        with st.chat_message(message["role"]):
            if isinstance(message["content"], str):
                st.markdown(message["content"])
            elif isinstance(message["content"], list):
                st.markdown(message["content"][0]["text"])

if prompt := st.chat_input("who are the top institutional holders of Apple?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        messages, stream = generate_response(model, use_tools)
        response = st.write_stream(stream)
        st.session_state.messages = messages
    st.session_state.messages.append({"role": "assistant", "content": response})

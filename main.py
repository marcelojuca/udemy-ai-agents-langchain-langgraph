import os

import streamlit as st
from dotenv import load_dotenv

from backend.core import run_llm

load_dotenv()
st.header("LangChain Udemy Course - Document Assistant Bot")
prompt = st.text_input("Prompt", placeholder="Enter your prompt here")

if (
    ("user_prompt_history" not in st.session_state)
    and ("chat_answer_history" not in st.session_state)
    and ("chat_history" not in st.session_state)
):
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_answer_history"] = []
    st.session_state["chat_history"] = []


def create_sources_string(sources_urls: set[str]) -> str:
    if not sources_urls:
        return ""
    sources_list = list(sources_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


if prompt:
    with st.spinner("Thinking..."):

        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )
        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )

        formatted_response = (
            f"{generated_response['result']} \n\n {create_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answer_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["result"]))

if st.session_state["chat_answer_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answer_history"], st.session_state["user_prompt_history"]
    ):
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(generated_response)

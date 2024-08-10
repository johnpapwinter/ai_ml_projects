import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from streamlit_float import float_init, float_parent, float_css_helper

from self_rag.sw_qa_bot.qa_bot import get_qa_bot


st.set_page_config(page_title="Prequel QA Bot", layout="wide", page_icon="🤖")
st.header('Prequel QA Bot :sunglasses:', divider='grey')
float_init(theme=True, include_unstable_primary=False)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


with st.container():
    with st.container():
        user_query = st.chat_input("Type your question here...")
        if user_query is not None and user_query != "":
            # response = "Hello"
            response = st.write_stream(
                get_qa_bot().stream({"input": user_query, "chat_history": st.session_state.chat_history})
            )
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))
        button_bottom_position = "1.5rem"
        button_css = float_css_helper(bottom=button_bottom_position, transition=0)
        float_parent(css=button_css)

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
        else:
            continue

    # st.write(st.session_state.chat_history)
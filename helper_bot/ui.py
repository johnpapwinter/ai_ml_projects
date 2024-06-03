import streamlit as st
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage
from helper import GMHelper
from knowledge_base import KnowledgeBase
from loaders import BytesIOPyMuPDFLoader
from streamlit_float import float_init, float_parent, float_css_helper

db = KnowledgeBase()
helper = GMHelper()
file_df = pd.DataFrame({"Filename": list(db.get_filenames())})


# dialog box for document handling
@st.experimental_dialog("Select Action")
def handle_knowledge_base(action):
    match action:
        case "UPLOAD":
            st.text("UPLOAD A NEW DOCUMENT TO THE KNOWLEDGE BASE")
            doc_type = st.text_input(label="doc_type", placeholder="Enter document type")
            uploaded_file = st.file_uploader(
                label="Choose a file", type=["pdf"]
            )
            # CALL TO UPLOAD FUNCTION (params doc_type, uploaded_file)
            loader = BytesIOPyMuPDFLoader(uploaded_file)
            db.create_document(uploaded_file.file_id, loader.load())
        case "UPDATE":
            st.text("UPDATE AN EXISTING DOCUMENT OF THE KNOWLEDGE BASE")
            documents = st.selectbox(
                label="Select document",
                options=("SRD_Spells.pdf", "SRD_Races.pdf", "SRD_Monsters.pdf"),
                index=None,
                placeholder="Select a document"
            )
            st.write(f"We will update {documents}")
            uploaded_file = st.file_uploader(
                label="Choose a file", type=["pdf"]
            )
            # CALL TO UPDATE FUNCTION (param uploaded_file)
            loader = BytesIOPyMuPDFLoader(uploaded_file)
            db.update_document(documents, loader.load())
        case "DELETE":
            st.text("REMOVE AN EXISTING DOCUMENT FROM THE KNOWLEDGE BASE")
            documents = st.selectbox(
                label="Select document",
                options=("SRD_Spells.pdf", "SRD_Races.pdf", "SRD_Monsters.pdf"),
                index=None,
                placeholder="Select a document"
            )
            st.write(f"We will delete {documents}")
            # CALL TO DELETE FUNCTION (param deleted_file)
            db.delete_document(documents)
    if st.button("Submit"):
        st.session_state.document = {"key": "some_key", "value": "some_value"}
        # st.rerun()


# page configs
st.set_page_config(page_title="GM Helper Bot", layout="wide", page_icon="🤖")
st.header('GM Helper :sunglasses:', divider='grey')
float_init(theme=True, include_unstable_primary=False)

chat_tab, docs_tab = st.tabs(["Chat", "Docs"])
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# chat tab
with chat_tab:
    with st.container():
        with st.container():
            user_query = st.chat_input("Type your question here...")
            if user_query is not None and user_query != "":
                response = st.write_stream(helper.get_response(user_query, st.session_state.chat_history))
                # response = helper.get_response(user_input=user_query, chat_history=st.session_state.chat_history)
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
                with st.chat_message("User"):
                    st.write(message.content)


# knowledge base tab
with docs_tab:
    with st.container():
        docs_column, actions_column = st.columns([0.8, 0.2])
        with docs_column:
            st.dataframe(
                file_df,
                hide_index=True,
                use_container_width=True,
            )
        with actions_column:
            st.write("Please select your action")
            if "document" not in st.session_state:
                if st.button("Upload 📤"):
                    handle_knowledge_base("UPLOAD")
                if st.button("Update ✏️"):
                    handle_knowledge_base("UPDATE")
                if st.button("Delete 🗑️"):
                    handle_knowledge_base("DELETE")

with st.sidebar:
    st.write(st.session_state.chat_history)

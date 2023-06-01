import os
import chromadb
import streamlit as st
from dotenv import load_dotenv
from uuid import uuid4
from chromadb.config import Settings
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage, AIMessage

load_dotenv()

project_directory = st.sidebar.text_input(
    "Project Directory", value="", key="project_directory"
)

if not project_directory:
    st.warning("Please enter the project directory.")
    st.stop()

persist_directory = os.path.join(project_directory, "db.chromadb")

api_key = st.sidebar.text_input("API Key", type="password", key="api_key")

if not api_key:
    st.warning("Please enter the OpenAI API Key.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet", persist_directory=persist_directory
    )
)
embeddings = OpenAIEmbeddings()

st.title("AI Assistant with Memory")

collection_name = "conversation_history"

try:
    collection = client.get_collection(collection_name)
except ValueError:
    collection = client.create_collection(collection_name)

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "The following is a friendly conversation between a human and an AI. "
            "The AI is talkative and provides lots of specific details from its context. "
            "If the AI does not know the answer to a question, it truthfully says it does not know."
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
chain = LLMChain(llm=chat, prompt=prompt)

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You: ", placeholder="Ask anything...")

if st.button("Send"):
    response = chain.run(input=user_input, history=st.session_state.history)
    st.session_state.history.append(HumanMessage(content=user_input))
    st.session_state.history.append(AIMessage(content=response))

    full_conversation = "\n".join(
        [
            f"You: {message.content}"
            if isinstance(message, HumanMessage)
            else f"AI: {message.content}"
            for message in st.session_state.history
        ]
    )
    embeddings_full_conversation = embeddings.embed_documents(
        [full_conversation]
    )[0]
    collection.add(
        ids=[str(uuid4())],
        embeddings=[embeddings_full_conversation],
        metadatas=[{"type": "conversation"}],
        documents=[full_conversation],
    )

if st.button("Reset"):
    st.session_state.history = []

for i in range(0, len(st.session_state.history), 2):
    st.text(f"You: {st.session_state.history[i].content}")
    if i + 1 < len(st.session_state.history):
        st.text(f"AI: {st.session_state.history[i + 1].content}")

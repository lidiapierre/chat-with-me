import streamlit as st
import pinecone
import os
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from models import get_llm, embeddings
import openai

from config import config

if not load_dotenv():
    print("Missing .env file")
    exit(1)

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_ENVIRONMENT')

index_name = os.environ.get('INDEX_NAME')

openai.api_key = os.environ["OPENAI_API_KEY"]


def display_existing_messages():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant", avatar='profile.png'):
                st.write(message["content"])


def add_user_message_to_session(prompt):
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)


system_prompt = f"""
Your name is {config['name']}. You are a {config['role']}.
Your task is to answer user questions about you.
You will use the information given above each question, do not make up answers.
If the information is missing, say "Sorry, I do not have this information.", then offer to ask questions regarding
your professional experience or skills.
Answer concisely and precisely, do not be vague. Sound professional and friendly.
End each sentence with a period.
"""


def generate_assistant_response(augmented_query):
    with st.chat_message("assistant", avatar='profile.png'):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # TODO!!!!
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query},
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

        st.session_state["messages"].append(
            {"role": "assistant", "content": full_response}
        )
    return full_response


def get_relevant_contexts(query, embeddings):
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV
    )
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    docs = docsearch.similarity_search(query)
    context = ""
    for doc in docs:
        context += doc.page_content
    return context


def augment_query(contexts, query):
    augmented_query = (
        f"###Search Results: \n{contexts} #End of Search Results\n\n-----\n\n {query}"
    )
    return augmented_query


def print_markdown_from_file(file_path):
    with open(file_path, "r") as f:
        markdown_content = f.read()
        st.markdown(markdown_content)


def hide_streamlit_header_footer():
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)


def page_setup():
    hide_streamlit_header_footer()
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        st.title("")
        st.image('profile.png', width=100)
    with col2:
        st.title(f"Hi, I am {config['name']}")
        st.subheader(config['role'])


def main():
    page_setup()
    display_existing_messages()
    query = st.chat_input("Ask me about my skills and expertise!")
    if query:

        add_user_message_to_session(query)
        contexts = get_relevant_contexts(query, embeddings)
        augmented_query = augment_query(contexts, query)
        generate_assistant_response(augmented_query)


if __name__ == "__main__":
    main()

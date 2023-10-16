import base64
import os
import pickle

import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub

from config import config

if not load_dotenv():
    print("Missing .env file")
    exit(1)

embeddings_model_name = os.environ.get("HF_EMBEDDING_MODEL_NAME")

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

persist_directory = os.environ.get('PERSIST_DIRECTORY')
llm = HuggingFaceHub(  # TODO parametrize
    repo_id="google/flan-t5-xxl",
    model_kwargs={"temperature": 0.5, "max_length": 512}
)

target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

openai.api_key = os.environ["OPENAI_API_KEY"]


def display_existing_messages():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    for message in st.session_state["messages"]:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar='profile.jpg'):
                st.markdown(message["content"])


def add_user_message_to_session(prompt):
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)


def generate_assistant_response(augmented_query):
    system_prompt = f"""
You are impersonating {config['name']}. This person is a {config['role']}.
Your task is to answer user questions about you. You will answer in the first person ('I ..'),
using the information given above each question. 
If the information is missing, say "Sorry, I do not have this information.", then offer to ask questions regarding
your professional experience or skills.
Answer concisely and precisely, do not be vague.
End each sentence with a period.
"""
    with st.chat_message("assistant", avatar='profile.jpg'):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(  # TODO change
                model="gpt-3.5-turbo",
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


def get_relevant_contexts(query):  # TODO directly use RetrievalConversational chain?
    with open(f"persist/data.pkl", "rb") as f:
        db = pickle.load(f)
    # retriever = db.as_retriever(search_kwargs={"k": target_source_chunks}) TO USE WITH RETRIEVAL QA CHAIN
    result = db.similarity_search(query)
    # res = re.query(query_embedding, top_k=8, include_metadata=True)
    # contexts = []
    # for item in res["matches"]:
    #     metadata = item["metadata"]
    #     text = metadata.get("text", "")
    #     url = metadata.get("url", "")
    #     title = metadata.get("title", "")
    #     relevance_score = item.get("score", "")
    #     context = {
    #         "search_results_text": text,
    #         "search_results_url": url,
    #         "search_results_title": title,
    #         "search_relevance_score": relevance_score,
    #     }
    #     contexts.append(context)

    # contexts = str(contexts)
    # return contexts
    print(f"using {len(result)} documents")
    context = ""
    for doc in result:
        context += doc.page_content
    print(context)
    return context


def augment_query(contexts, query):
    augmented_query = (
        f"###Search Results: \n{contexts} #End of Search Results\n\n-----\n\n {query}"
    )
    return augmented_query


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


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
        unsafe_allow_html=True
    )


def main():
    add_bg_from_local("abstract_6.jpg")
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        st.title("")
        st.image('profile.png', width=100)
    with col2:
        st.title(f"Hi, I am {config['name']}")
        st.subheader(config['role'])

    # db = FAISS(persist_directory=persist_directory, embedding_function=embeddings)
    # retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    #
    # qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=retriever,
    #     return_source_documents=True
    # )

    hide_streamlit_header_footer()
    display_existing_messages()

    # query = st.text_input(label="ama", label_visibility='hidden', placeholder="Ask me anything !")
    query = st.chat_input("Ask me anything!")

    if query:
        add_user_message_to_session(query)
        contexts = get_relevant_contexts(query)
        augmented_query = augment_query(contexts, query)
        generate_assistant_response(augmented_query)
        # add_to_database(query, response)
    # with st.sidebar: # TODO maybe add resume ?
    #     print_markdown_from_file("case_studies.md")


if __name__ == '__main__':
    main()

import os

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import HuggingFaceHub

HF_EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
HF_LLM = 'google/flan-t5-xxl'

if not load_dotenv():
    print("Missing .env file")
    exit(1)


def get_embeddings():
    if os.getenv("OPENAI_EMBEDDINGS", 'False').lower() in ('true', '1', 't'):
        print("using openAI embeddings")
        return OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    else:
        print("using HF embeddings")
        return HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL_NAME)


def get_llm():
    model_temperature = os.environ["MODEL_TEMPERATURE"] or 0.2
    if os.getenv("OPENAI_LLM", 'False').lower() in ('true', '1', 't'):
        print("using openAI chat LLM")
        return ChatOpenAI(temperature=model_temperature)
    else:
        print("using HF LLM")
        return HuggingFaceHub(repo_id=HF_LLM, model_kwargs={
            "temperature": model_temperature,
            "max_length": 512}
                              )


embeddings = get_embeddings()

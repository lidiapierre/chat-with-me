import logging
import os

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import HuggingFaceHub

HF_EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
HF_LLM = 'google/flan-t5-xxl'

if not load_dotenv():
    logging.error("Missing .env file")
    exit(1)


def get_embeddings():
    if os.getenv("OPENAI_EMBEDDINGS", 'False').lower() in ('true', '1', 't'):
        logging.info("Using openAI embeddings")
        return OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    else:
        logging.info(f"Using HF embeddings {HF_EMBEDDING_MODEL_NAME}")
        return HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL_NAME)


def get_llm():
    model_temperature = os.environ["MODEL_TEMPERATURE"] or 0.2
    if os.getenv("OPENAI_LLM", 'False').lower() in ('true', '1', 't'):
        logging.info("Using openAI chat LLM")
        return ChatOpenAI(temperature=model_temperature)
    else:
        logging.info(f"using HF LLM {HF_LLM}")
        return HuggingFaceHub(repo_id=HF_LLM, model_kwargs={
            "temperature": model_temperature,
            "max_length": 512}
                              )


embeddings = get_embeddings()

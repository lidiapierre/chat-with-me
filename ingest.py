import glob
import logging
import os

import pinecone
from dotenv import load_dotenv
from langchain.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

from config import config
from models import get_embeddings

if not load_dotenv():
    logging.error("Missing .env file")
    exit(1)

source_directory = 'source_documents'
chunk_size = os.environ.get('CHUNK_SIZE')
chunk_overlap = os.environ.get('CHUNK_OVERLAP')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

embeddings = get_embeddings()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_ENVIRONMENT')

index_name = os.environ.get('INDEX_NAME')

LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


def load_single_document(file_path):
    ext = "." + file_path.rsplit(".", 1)[-1].lower()
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")


def get_documents_chunks_from_files():
    """
    Loads all documents from the source documents directory
    #TODO check file extensions, add pooling
    """
    all_files = glob.glob(f"{source_directory}/*")
    docs = []
    for f in all_files:
        docs.extend(load_single_document(f))

    chunks = text_splitter.split_documents(docs)
    return chunks


def get_documents_chunks_from_urls():
    urls = []
    if config['github']:
        urls.append(config['github'])
    if config['linkedin']:
        urls.append(config['linkedin'])
    if config['other_urls']:
        urls.extend(config['other_urls'])
    loader = UnstructuredURLLoader(urls=urls)
    chunks = text_splitter.split_documents(loader.load())
    return chunks


def main():
    """
    Load documents and split in chunks
    #TODO add batching
    """

    pinecone.init(api_key=PINECONE_API_KEY,
                  environment=PINECONE_API_ENV)

    if index_name not in pinecone.list_indexes():
        test_query = embeddings.embed_query("test")
        dimension = len(test_query)
        logging.info(f"Creating new Pinecone index {index_name} of dimension {dimension}")
        # we create a new index
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=dimension
        )

    documents = []
    logging.info(f"Loading documents from {source_directory}")
    documents.extend(get_documents_chunks_from_files())
    logging.info(f"Reading `config.py`")
    documents.extend(get_documents_chunks_from_urls())
    Pinecone.from_documents(documents, embeddings, index_name=index_name)

    # TODO show progress bar with tqdm 'show_progress=True'


if __name__ == "__main__":
    main()

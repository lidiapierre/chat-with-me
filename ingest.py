import glob
import os
import pickle

from dotenv import load_dotenv
from langchain.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import \
    FAISS  # TODO use Pinecone and https://github.com/lidiapierre/seo-chat-bot/blob/master/scraper-embedder.py

if not load_dotenv():
    print("Missing .env file")
    exit(1)

persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = 'source_documents'
embeddings_model_name = os.environ.get('HF_EMBEDDING_MODEL_NAME')
chunk_size = 500  # TODO move to env
chunk_overlap = 50

text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

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


def get_documents_chunks(source_dir, existing_docs):
    """
    Loads all documents from the source documents directory
    #TODO check file extensions, add pooling, add list of files to ignore
    """
    all_files = glob.glob(f"{source_dir}/*")
    all_docs = []
    for f in all_files:
        if f not in existing_docs:
            all_docs.extend(load_single_document(f))

    chunks = text_splitter.split_documents(all_docs)
    return chunks


def main():
    """
    Load documents and split in chunks
    #TODO add batching
    """
    print(f"Loading documents from {source_directory}")  # TODO add support to read from config

    print("Creating new vectorstore")
    documents = get_documents_chunks(source_directory, [])
    vector_store = FAISS.from_documents(documents, embeddings)  # TODO add persist
    with open(f"persist/data.pkl", "wb") as f:
        pickle.dump(vector_store, f)

    # if not os.path.exists(persist_directory):
    # print("Creating new vectorstore")
    # documents = get_documents_chunks(source_directory, [])
    # vector_store = FAISS.from_documents(documents, embeddings) # TODO add persist
    # with open(f"persist/data.pkl", "rb") as f:
    #     pickle.dump(vector_store, f)

    # else:
    # db = FAISS(persist_directory=persist_directory, embedding_function=embeddings)
    # collection = db.get()
    # existing_docs = [metadata['source'] for metadata in collection['metadatas']]
    # documents = get_documents_chunks(source_directory, existing_docs)
    # vector_store = FAISS.from_documents(documents, embeddings)

    if not documents:
        print("No new documents to load")
        exit(0)

    print(f"Loaded new documents from {source_directory}")  # TODO logging instead of print


if __name__ == "__main__":
    main()

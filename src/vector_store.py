from pathlib import Path

# FAISS use the vector to calculate the similarity between the query and the document, and return the most similar documents to the query
# Choroma is a light weight vector database, which is a wrapper of FAISS
from langchain_community.vectorstores import FAISS

from src.embeddings import get_embeddings

FAISS_PATH = "data/faiss_index"

def build_vector_store(documents):
    """
        build a vector store from the specified documents
        the input documents is a list of document objects
        which is the output of the split_documents function in ingestion.py
        it is also the chunks of text after the documents are split into smaller chunks
    """
    # use it to embed the chunks of text into vectors
    embeddings = get_embeddings()

    # build the vector store from the documents and the embeddings
    vector_store = FAISS.from_documents(
        documents,
        embeddings
    )

    # build the folder of the vector store if it does not exist
    Path(FAISS_PATH).parent.mkdir(parents = True, exist_ok = True)


    vector_store.save_local(FAISS_PATH)

    return vector_store

def load_vector_store():
    """
        load the vector store from the specified path
    """

    embeddings = get_embeddings()
    if Path(FAISS_PATH).exists():
        try:
            return FAISS.load_local(
                FAISS_PATH,
                embeddings,
                allow_dangerous_deserialization = True  # allow some dangerous information to be deserialized
            )
        except Exception as e:
            return None
    else:
        return None

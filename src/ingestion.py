
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(directory = "data/documents"):
    """"
        load documents from the specified directory
        and split them into chunks of text
    """

    documents = []

    for file in Path(directory).glob("*"):
        if file.suffix.lower() == ".pdf":
            # loader is a list of documents
            loader = PyPDFLoader(file)
            documents.extend(loader.load())
        elif file.suffix.lower() in [".txt", ".md"]:
            # loader is a list of documents
            loader = TextLoader(
                str(file),           # file path object to the text file string
                encoding = "utf-8"   # enforce UTF-8 encoding for text files
            )
            documents.extend(loader.load())
    return documents

def split_documents(documents, chunk_size = 300, chunk_overlap = 70):
    """
        split documents into chunks of text with the specified chunk size
        and  overlap
    """

    # recursive character text splitter is used to split the documents
    # into chunks of text with the specified chunk size and overlap
    # recursive means splitter will split the text into smaller chunks
    # throught the level of the text hierachy (paragraph, sentence, word)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    # format of return
    # [
    #     Document(page_content='text of the document', metadata={'source': 'path/to/document'}),
    #     Document(page_content='text of the document', metadata={'source': 'path/to
    #  ]
    # let the bigger document be smaller chunks of text
    return splitter.split_documents(documents)

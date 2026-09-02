from src.ingestion import load_documents, split_documents
from src.generation import generate_answer
from src.hybrid import HybridRetriever
from src.reranker import Reranker
from src.vector_store import load_vector_store, build_vector_store
from src.bm25 import BM25Retriever
# 0 - faiss
# 1 - bm25
# 2 - hybrid
# 3 - hybrid + reranker
RETRIEVAL_METHOD = 0
RETRIEVAL_K = 8
RERANK_TOP_K = 4

def main():
    """
        main function to run the RAG pipeline

        1. load document
        2. split document into chunks of text
        3. build vector store from the chunks of text
        4. load vector store from the specified path
        5. input the query and get the most similar documents

        RAG pipeline:

        Document
            ↓
        Chunking
            ↓
        Retrieval
            ↓
        Optional Reranking
            ↓
        LLM
            ↓
        Answer
    """



    # load document
    documents = load_documents()
    print(f"Loaded {len(documents)} documents from the specified directory.")
    print(f"Document source: {[doc.metadata['source'] for doc in documents]}")

    # split document
    chunks = split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks of text.")

    # create retriever
    vector_store = None
    vector_store = load_vector_store()
    if vector_store is not None:
        print("FAISS index loaded.")
    else:
        # build vector store
        vector_store = build_vector_store(documents)
        print("FAISS index created.")

    faiss_retriever = vector_store.as_retriever(search_kwargs = {"k": RERANK_TOP_K})
    bm25_retriever = BM25Retriever(chunks, k = RERANK_TOP_K)

    hybrid_retriever = HybridRetriever(chunks, k = RETRIEVAL_K, rrf_k = 60)
    reranker = Reranker()


    # input the query
    while True:
        query = input("\nPlease input your query (or type 'q' to quit):")

        if query.lower() == "q":
            print("Exiting the program.")
            break

        # =========================
        # Retrieval
        # =========================
        if RETRIEVAL_METHOD == 0:
            # FAISS retrieval
            results = faiss_retriever.invoke(query)
        elif RETRIEVAL_METHOD == 1:
            # BM25 retrieval
            results = bm25_retriever.retrieve(query)
        elif RETRIEVAL_METHOD == 2:
            # Hybrid retrieval
            results = hybrid_retriever.hybrid_retrieve(query)
        elif RETRIEVAL_METHOD == 3:
            # Hybrid retrieval + Reranking
            results = hybrid_retriever.hybrid_retrieve(query)
            results = reranker.rerank(query, results, top_k = RERANK_TOP_K)
        else:
            print("Invalid retrieval method. Please choose from 0, 1, 2, or 3.")
            continue

        print("\n Generating answer from the most similar documents...")

        answer = generate_answer(query, results)

        print(f"\nAnswer:\n{answer}\n")

if __name__ == "__main__":
    main()

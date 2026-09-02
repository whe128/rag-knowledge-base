from collections import defaultdict

from src.bm25 import BM25Retriever
from src.vector_store import load_vector_store, build_vector_store

class HybridRetriever:
    def __init__(self, documents, k = 5, rrf_k = 60):
        """
            input:
                documents: the documents to be retrieved - have been split into chunks of text
                k: the number of documents to return
                rrf_k: the k value for RRF, it is used to adjust the weight of the score of each document
        """
        self.documents = documents
        self.k = k
        self.rrk_k = rrf_k

        vector_store = None
        vector_store = load_vector_store()
        if vector_store is not None:
            print("FAISS index loaded.")
        else:
            # build vector store
            vector_store = build_vector_store(documents)
            print("FAISS index created.")

        self.vector_retriever = vector_store.as_retriever(search_kwargs = {"k": self.k})
        self.bm25_retriever = BM25Retriever(documents, k = self.k)


    def hybrid_retrieve(self, query):
        """
            use the query to get the score of each document
            then get the content from original documents by the index of the score
        """

        # FAISS retrieval
        # use the cosine similarity, but it cannot see the content of each other, rough similarity, but it is fast
        vector_results = self.vector_retriever.invoke(query)

        # BM25 retrieval
        bm25_results = self.bm25_retriever.retrieve(query)

        # RRF
        scores = defaultdict(float)
        documents = {}                  # store the map id -> document object

        for rank, doc in enumerate(vector_results):
            doc_id = doc.metadata['source']

            scores[doc_id] += 1 / (rank + 1 + self.rrf_k)

            documents[doc_id] = doc

        for rank, doc in enumerate(bm25_results):
            doc_id = doc.metadata['source']

            scores[doc_id] += 1 / (rank + 1 + self.rrf_k)

            documents[doc_id] = doc

        # sort the documents by score
        sorted_docs = sorted(
            scores,
            key = lambda x: scores[x],
            reverse = True
        )

        return [documents[doc_id] for doc_id in sorted_docs[:self.k]]

from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name = "BAAI/bge-reranker-v2-m3"):
        """
            initialize the reranker with the specified model
        """

        # use the CrossEncoder model to rerank the documents based on the query
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_k = 3):
        """
            rerank the documents based on the query and return the top k documents
            compare the query with each document and get the score of each document
            FAISS retrieval is rough similarity, its use the bi-encoder based on bge-m3
            rerank is more accurate, use the cross-encoder based on bge-reranker-v2-m3, it can see the content of each document and the query, and get the score of each document
                send query + document to the cross-encoder model - > they can see each other
                use transformer to calculate the correlation between the query and the document
                input [CLS] query [SEP] document [SEP] doc_text [SEP]
                get the score of each document

                advantage: sufficiently understand the sematic correlation
            input:
                query: the query to compare with the documents
                documents: the documents to be reranked (have hybrid retrieval, BM25 and FAISS)
                top_k: the number of documents to return
        """

        # get the score of each document
        pairs = [ [query, doc.page_content] for doc in documents ]

        scores = self.model.predict(pairs)

        ranked_docs = sorted(
            zip(documents, scores),
            key = lambda x: x[1],
            reverse = True
        )


        return [doc for doc, _ in ranked_docs[:top_k]]

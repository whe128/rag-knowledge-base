from rank_bm25 import BM25Okapi

from src.ingestion import load_documents, split_documents

class BM25Retriever:
    """
    BM25Retriever is a class that implements the
    BM25 algorithm for information retrieval.
    """

    def __init__(self, documents, k = 5):
        self.documents = documents
        self.k = k

        # build the corpus from the documents
        corpus = [doc.page_content for doc in documents]

        # tokenize the corpus
        tokenized_corpus = [
            # to lower and split the text into words
            # example "Hello World" -> ["hello", "world"]
            # tokenized corpus is a list of lists of words
            # example [
            #           ["hello", "world"],
            #           ["this", "is", "a", "test"]
            #         ]
            text.lower().split() for text in corpus
        ]

        # build the BM25 model from the tokenized corpus
        # it is the knowledge base of the BM25 algorithm
        # we can input query to get the most similar documents from the knowledge base
        # use the query, we can get the score of each document
        # here we just get the index of each document
        # we need extract the document from the index to get the document object
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query):
        """
            use the query to get the score of each document
            then get the content from original documents by the index of the score
        """

        # tokenize the query
        tokenized_query = query.lower().split()

        # use the tokenized query to extract the top k most similar documents from the knowledge base
        result = self.bm25.get_top_n(
            tokenized_query,        # query
            self.documents,         # original documents
            n = self.k              # number of documents to return
        )
        return result

if __name__ == "__main__":
    # load documents from the specified directory
    documents = load_documents()

    # get the file name of each document
    print( "document source: ", [doc.metadata['source'] for doc in documents])

    # split documents into chunks of text
    chunks = split_documents(documents)

    # build the BM25 retriever from the chunks of text
    retriever = BM25Retriever(chunks)

    # input the query
    while True:
        query = input("\nPlease input your query (or type 'q' to quit):")

        if query.lower() == "q":
            print("Exiting the program.")
            break

        # get the most similar documents
        similar_docs = retriever.retrieve(query)

        print(f"\nThe top {retriever.k} most similar documents are:")
        for i, doc in enumerate(similar_docs):
            print(f"\nDocument {i + 1}:")
            print(f"Source: {doc.metadata['source']}")
            print(f"Content: {doc.page_content}\n")

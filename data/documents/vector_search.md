# Vector Search and FAISS

## What is Vector Search?

Vector search retrieves documents by comparing numerical vector representations.

A document collection is first converted into embeddings. These embeddings are stored in a vector index.

When a user submits a query, the query is also converted into an embedding. The vector index then searches for document vectors that are most similar to the query vector.

The basic process is:

Documents
→ Embeddings
→ Vector Index

Query
→ Query Embedding
→ Similarity Search
→ Top-K Documents

## FAISS

FAISS is a library developed for efficient similarity search over dense vectors.

In a RAG system, FAISS can be used as the vector index for document embeddings.

Each indexed chunk can also contain metadata such as:

- Source file
- Page number
- Document ID
- Chunk ID

The metadata allows the RAG system to identify where a retrieved piece of information came from.

## FAISS Retrieval

Suppose a knowledge base contains 10,000 document chunks.

A user asks:

"What is BM25?"

The query is converted into an embedding using BGE-M3.

FAISS compares the query vector with the vectors stored in the index and returns the most similar chunks.

For example:

Top 1 → BM25 definition
Top 2 → Lexical retrieval
Top 3 → Hybrid retrieval
Top 4 → Information retrieval

The retrieved chunks can then be passed to an LLM.

## Vector Search Does Not Generate Answers

FAISS is a retrieval system, not a language model.

FAISS answers the question:

"Which documents are relevant to this query?"

It does not answer:

"What should I say to the user?"

The LLM performs the generation stage.

Therefore, the architecture is:

Query
→ Embedding
→ FAISS
→ Retrieved Documents
→ LLM
→ Answer

## Retrieval and Generation

Retrieval and generation should be considered separate components.

A retrieval system can be evaluated using retrieval metrics such as Recall@K and MRR.

The language model can be evaluated using metrics such as answer correctness and faithfulness.

This separation makes it possible to identify whether a bad final answer was caused by poor retrieval or poor generation.

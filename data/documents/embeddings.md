# Text Embeddings

## What is a Text Embedding?

A text embedding is a numerical vector representation of text.

An embedding model maps a sentence, paragraph, or document into a vector space. Texts with similar meanings should ideally have similar vector representations.

For example, the following two sentences have different words but similar meanings:

"How does vector search work?"

"How can documents be retrieved using embeddings?"

A good embedding model should place these sentences relatively close to each other in the embedding space.

## Semantic Similarity

Embedding-based retrieval is based on semantic similarity rather than exact keyword matching.

A query is first converted into a vector. Document chunks are also represented as vectors. The retrieval system then searches for document vectors that are close to the query vector.

This allows dense retrieval to find relevant documents even when the query and document do not contain exactly the same words.

## Cosine Similarity

Cosine similarity measures the similarity between two vectors based on the angle between them.

For two vectors A and B:

cosine_similarity(A, B) = (A · B) / (||A|| ||B||)

If both vectors are normalized to unit length, their dot product becomes equivalent to cosine similarity:

A · B = cosine_similarity(A, B)

This is why embedding systems often normalize vectors before performing similarity search.

## BGE-M3

BGE-M3 is a multilingual embedding model that can be used for information retrieval.

In this RAG project, BGE-M3 is responsible for converting document chunks and queries into vector representations.

The workflow is:

Document Chunk
→ BGE-M3
→ Embedding Vector
→ FAISS

At query time:

User Query
→ BGE-M3
→ Query Vector
→ FAISS Search

## Embedding Model Consistency

The same embedding model should normally be used when building and querying a vector index.

If an index was created using one embedding model and queries are encoded using an incompatible embedding model, the vectors may belong to different embedding spaces.

In that situation, the vector index should generally be rebuilt.

## Dense Retrieval

Dense retrieval uses embedding vectors to retrieve semantically similar documents.

Its major advantage is semantic matching.

For example, a query containing:

"How can an LLM access external knowledge?"

may retrieve a document discussing:

"Retrieval-Augmented Generation provides language models with information from external documents."

Even though the exact wording is different, the semantic meaning is similar.

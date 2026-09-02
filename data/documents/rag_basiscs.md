# Retrieval-Augmented Generation (RAG)

## What is RAG?

Retrieval-Augmented Generation, commonly called RAG, is a system architecture that combines information retrieval with large language model generation. Instead of relying only on the knowledge stored in a language model's parameters, a RAG system retrieves relevant external information and provides it to the language model as context.

RAG is useful when the required information is domain-specific, frequently updated, private, or too large to be reliably stored in the model parameters.

## RAG Pipeline

A typical RAG pipeline contains several stages:

1. Document loading
2. Document chunking
3. Text embedding
4. Vector indexing
5. Information retrieval
6. Optional reranking
7. Context construction
8. LLM generation

First, documents are loaded from files such as PDF, Markdown, or plain text. The documents are then divided into smaller chunks. Each chunk is converted into an embedding vector and stored in a vector index.

When a user submits a query, the query is converted into an embedding and used to retrieve relevant document chunks. A lexical retriever such as BM25 can also be used. The retrieved candidates may then be reranked before they are passed to the language model.

## Why Retrieval Matters

Retrieval quality has a strong influence on the final answer quality of a RAG system. A language model cannot reliably answer a question using external knowledge if the relevant information was never retrieved.

Therefore, retrieval systems usually have two goals:

- High recall during the candidate retrieval stage
- High precision during the final ranking stage

The first-stage retriever should retrieve enough potentially relevant documents. A reranker can then select the most relevant documents from those candidates.

## RAG versus Fine-Tuning

RAG and fine-tuning solve different problems.

RAG provides external information to the model at inference time. The underlying model parameters do not need to be changed.

Fine-tuning changes model parameters using additional training data. It is useful when the goal is to change model behavior, style, or task-specific capabilities.

RAG is often preferred when the main requirement is to provide the model with an up-to-date or private knowledge base.

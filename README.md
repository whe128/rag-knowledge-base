# RAG-knowledge-base

An end-to-end Retrieval-Augmented Generation (RAG) system built with LangChain.

The project demonstrates document ingestion, text chunking, BGE-M3 embeddings, FAISS vector retrieval, BM25 lexical retrieval, hybrid retrieval, BGE reranking, and LLM-based answer generation.

## Architecture

```text
Documents
   ↓
LangChain Loader
   ↓
Text Chunking
   ↓
BGE-M3 Embedding
   ↓
FAISS
   ↓
Vector Retrieval
        +
       BM25
        ↓
Hybrid Retrieval
        ↓
BGE Reranker
        ↓
Top-K Context
        ↓
Qwen / DeepSeek
        ↓
Answer + Sources

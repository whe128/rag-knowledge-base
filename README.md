# RAG Knowledge Base

An end-to-end Retrieval-Augmented Generation (RAG) knowledge base QA system built with **LangChain, BGE-M3, FAISS, BM25, RRF, BGE Reranker, and Ollama**.

The project implements and compares multiple retrieval strategies to study how different retrieval components affect the final RAG answer quality.

## Architecture

```text
                    Documents
                        │
                        ▼
                Document Loading
                        │
                        ▼
                    Chunking
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
        BGE-M3 Embedding          BM25
             │                     │
             ▼                     ▼
           FAISS              Lexical Retrieval
             │                     │
             └──────────┬──────────┘
                        ▼
                 Hybrid Retrieval
                        │
                       RRF
                        │
                        ▼
                 Candidate Chunks
                        │
                        ▼
              BGE Reranker (Optional)
                        │
                        ▼
                   Top-K Chunks
                        │
                        ▼
                  Local LLM
                    Ollama
                        │
                        ▼
                     Answer
```

## Features

* PDF / TXT / Markdown document loading
* Recursive text chunking
* Dense retrieval with **BGE-M3**
* Vector search with **FAISS**
* Lexical retrieval with **BM25**
* Hybrid retrieval using **Reciprocal Rank Fusion (RRF)**
* Second-stage reranking with **BGE-Reranker-v2-M3**
* Local LLM generation with **Ollama**
* Retrieval method comparison
* Retrieval evaluation with **Recall@K** and **MRR@K**
* Generation evaluation for answer correctness and faithfulness

## Retrieval Methods

The project supports four retrieval configurations:

| Method            | Pipeline                                    |
| ----------------- | ------------------------------------------- |
| FAISS             | FAISS → Top-K → LLM                         |
| BM25              | BM25 → Top-K → LLM                          |
| Hybrid            | FAISS + BM25 → RRF → Top-K → LLM            |
| Hybrid + Reranker | FAISS + BM25 → RRF → Reranker → Top-K → LLM |

The retrieval method can be selected in `run_rag.py`:

```python
# 0 - FAISS
# 1 - BM25
# 2 - Hybrid
# 3 - Hybrid + Reranker

RETRIEVAL_METHOD = 0
```

## Why Hybrid Retrieval?

Dense retrieval and lexical retrieval have different strengths.

**Dense retrieval**

BGE-M3 converts documents and queries into embeddings and retrieves semantically similar chunks.

It is useful when the query and document use different wording but have similar meanings.

```text
Query:
"What is the purpose of vector embeddings?"

Document:
"Embeddings represent text as dense numerical vectors..."
```

**BM25**

BM25 focuses on lexical matching and term importance.

It can perform well when the query contains important technical terms, names, or exact keywords.

**Hybrid Retrieval**

The project combines both approaches:

```text
FAISS
  │
  ├── semantic relevance
  │
  ▼
Top-K documents
       \
        \
         → RRF → Final ranking
        /
       /
BM25
  │
  └── lexical relevance
```

## Reciprocal Rank Fusion

FAISS and BM25 produce scores with different scales, so their raw scores are not directly comparable.

Instead, this project uses **Reciprocal Rank Fusion (RRF)**.

For a document with rank `r`:

```text
RRF(d) = 1 / (k + r)
```

where `k` is typically set to `60`.

If a document appears in both retrieval results, its scores are accumulated:

```text
RRF(d) =
    1 / (60 + rank_FAISS)
  + 1 / (60 + rank_BM25)
```

This allows the system to combine multiple ranked lists without requiring their original scores to be on the same scale.

## Reranking

The hybrid retriever first retrieves a larger candidate set:

```text
FAISS Top-8
      +
BM25 Top-8
      ↓
     RRF
      ↓
Hybrid candidates
      ↓
BGE-Reranker-v2-M3
      ↓
Top-4
      ↓
LLM
```

The reranker uses a cross-encoder to directly evaluate the relevance between:

```text
(query, document)
```

This is different from the embedding-based retrieval stage.

The general design is:

```text
First-stage retrieval
        ↓
High recall
        ↓
Candidate documents
        ↓
Cross-encoder reranking
        ↓
High precision
        ↓
LLM
```

## Project Structure

```text
langchain-rag-knowledge-base/
│
├── data/
│   └── documents/
│       └── *.pdf / *.txt / *.md
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── bm25.py
│   ├── hybrid.py
│   ├── reranker.py
│   └── generation.py
│
├── evaluation/
│   └── questions.json
│
├── tests/
│
├── run_rag.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Installation

### 1. Create a Python environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama

Install Ollama and pull a local LLM:

```bash
ollama pull qwen2.5:1.5b
```

Start Ollama if it is not already running:

```bash
ollama serve
```

The default Ollama endpoint is:

```text
http://localhost:11434
```

## Model Setup

The project uses:

### Embedding Model

```text
BAAI/bge-m3
```

### Reranker

```text
BAAI/bge-reranker-v2-m3
```

### Generation Model

```text
Qwen 2.5 1.5B
```

through Ollama.

### Hugging Face Mirror

If Hugging Face access is unavailable, a mirror can be configured:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

On Windows PowerShell:

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

The Python process will then use the configured Hugging Face endpoint when downloading models.

## Add Documents

Place knowledge-base documents inside:

```text
data/documents/
```

For example:

```text
data/documents/
├── rag_basics.md
├── vector_database.md
├── bm25.md
├── embedding.md
├── reranking.md
└── transformer.md
```

The current ingestion pipeline supports:

```text
.pdf
.txt
.md
```

## Run

Run:

```bash
python run_rag.py
```

The program will:

```text
1. Load documents
2. Split documents into chunks
3. Load/build the FAISS index
4. Initialize BM25
5. Initialize the hybrid retriever
6. Initialize the reranker
7. Retrieve relevant chunks
8. Generate an answer with the local LLM
```

Then enter a question:

```text
Please input your query:
What is BM25?
```

Type:

```text
q
```

to exit.

## Configuration

The main retrieval configuration is controlled by:

```python
RETRIEVAL_METHOD = 0
RETRIEVAL_K = 8
RERANK_TOP_K = 4
```

### FAISS

```text
FAISS → Top-4 → LLM
```

### BM25

```text
BM25 → Top-4 → LLM
```

### Hybrid

```text
FAISS + BM25
      ↓
     RRF
      ↓
   Top-4
      ↓
     LLM
```

### Hybrid + Reranker

```text
FAISS + BM25
      ↓
     RRF
      ↓
   Top-8
      ↓
 BGE Reranker
      ↓
   Top-4
      ↓
     LLM
```

Using the same number of final chunks for the four methods makes the comparison of final LLM answers more controlled.

## Evaluation

The project evaluates the retrieval pipeline independently from the generation pipeline.

### Retrieval Evaluation

Two primary metrics are used:

#### Recall@K

Measures whether at least one relevant document/chunk appears in the top K results.

```text
Recall@K =
# queries with relevant result in Top-K
---------------------------------------
        # total queries
```

#### MRR@K

Measures how highly the first relevant result is ranked.

```text
MRR = 1 / rank_of_first_relevant_document
```

The retrieval experiment can compare:

| Method            | Recall@5 | MRR@5 |
| ----------------- | -------: | ----: |
| FAISS             |        - |     - |
| BM25              |        - |     - |
| Hybrid            |        - |     - |
| Hybrid + Reranker |        - |     - |

The goal is to determine whether each retrieval component improves retrieval quality.

### Generation Evaluation

Generation quality can be evaluated using:

* Answer correctness
* Faithfulness
* Context relevance
* LLM-as-a-Judge

A useful experiment is:

```text
FAISS
   vs
BM25
   vs
Hybrid
   vs
Hybrid + Reranker
```

while keeping the same:

* Questions
* Knowledge base
* LLM
* Prompt
* Number of final context chunks

This makes the comparison more meaningful.

## Evaluation Dataset

The evaluation questions should be constructed from the project's own knowledge base.

Example:

```json
[
  {
    "id": 1,
    "question": "What is BM25?",
    "relevant_chunks": [37],
    "reference_answer": "BM25 is a classical lexical information retrieval algorithm."
  }
]
```

Questions should include:

* Exact keyword queries
* Semantic/paraphrased queries
* Queries requiring hybrid retrieval
* Queries where multiple chunks are similar
* Unanswerable questions

This helps evaluate the strengths and weaknesses of different retrieval strategies.

## Example Experiment

A typical experiment can use:

```text
6 documents
      ↓
86 chunks
      ↓
     ┌──────────────────────────────┐
     │                              │
   FAISS                          BM25
     │                              │
     └──────────────┬───────────────┘
                    │
                   RRF
                    │
              Hybrid Retrieval
                    │
             BGE Reranker
                    │
                 Top-4
                    │
               Qwen 2.5
                    │
                  Answer
```

The experiment can then measure how retrieval quality changes when adding:

```text
FAISS
  ↓
BM25
  ↓
Hybrid / RRF
  ↓
Reranker
```

## Tech Stack

```text
Python
LangChain
BGE-M3
BGE-Reranker-v2-M3
FAISS
BM25
RRF
Ollama
Qwen 2.5
```

## Key Concepts Demonstrated

This project demonstrates the following RAG concepts:

```text
Document Processing
        ↓
Chunking
        ↓
Embedding
        ↓
Dense Retrieval
        ↓
Sparse Retrieval
        ↓
Hybrid Retrieval
        ↓
Rank Fusion
        ↓
Reranking
        ↓
Context Construction
        ↓
LLM Generation
        ↓
Evaluation
```

The project therefore covers the major components of a modern retrieval-augmented generation system from document ingestion to final answer evaluation.

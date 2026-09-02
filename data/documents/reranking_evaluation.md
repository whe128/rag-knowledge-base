# Reranking and RAG Evaluation

## What is Reranking?

Reranking is a second-stage retrieval process.

The first-stage retriever retrieves a relatively large candidate set.

A reranker then evaluates the query and candidate documents more carefully and produces a refined ranking.

The typical architecture is:

Query
→ First-stage Retrieval
→ Top-N Candidates
→ Reranker
→ Top-K Documents
→ LLM

## Why Use a Reranker?

The first-stage retriever is usually optimized for efficient retrieval and high recall.

It needs to search a large collection quickly.

A reranker can use a more expensive model because it only needs to process a small candidate set.

The goal is to improve precision among the retrieved candidates.

For example:

FAISS → Top 20
BM25 → Top 20
Hybrid RRF → Top 20
Reranker → Top 5
LLM

## Cross-Encoder

A cross-encoder receives the query and document together.

Instead of independently embedding the query and document, it directly models the interaction between them.

For example:

Query:
"What is the purpose of RRF?"

Document:
"RRF combines ranked lists from multiple retrieval systems."

The cross-encoder evaluates the relevance of this pair.

Cross-encoders can provide high-quality relevance scores, but they are more computationally expensive than embedding-based retrieval.

## BGE-Reranker-v2-M3

BGE-Reranker-v2-M3 is a multilingual reranking model.

In this project, it is used after hybrid retrieval.

The pipeline is:

FAISS

- BM25
  ↓
  RRF
  ↓
  Candidate Documents
  ↓
  BGE-Reranker-v2-M3
  ↓
  Final Top-K Documents
  ↓
  LLM

## Recall@K

Recall@K evaluates whether a relevant document appears in the top K retrieved results.

For example, suppose the relevant chunk for a question is chunk 17.

If the retrieval results are:

1. chunk 5
2. chunk 9
3. chunk 17
4. chunk 30
5. chunk 42

then Recall@5 for this query is 1 because the relevant chunk appears within the top five results.

If the relevant chunk does not appear within the top five results, Recall@5 is 0.

## MRR

MRR stands for Mean Reciprocal Rank.

It evaluates how highly the first relevant result appears.

For one query:

First relevant result at rank 1:

MRR = 1 / 1 = 1.0

First relevant result at rank 2:

MRR = 1 / 2 = 0.5

First relevant result at rank 5:

MRR = 1 / 5 = 0.2

The final MRR is the average reciprocal rank across all evaluation questions.

## Retrieval Evaluation versus Generation Evaluation

Retrieval evaluation and generation evaluation measure different things.

Retrieval evaluation asks:

"Did the system retrieve the relevant information?"

Useful metrics include:

- Recall@K
- MRR@K

Generation evaluation asks:

"Did the LLM produce a correct answer based on the retrieved information?"

Useful criteria include:

- Answer correctness
- Faithfulness
- Context relevance

These two evaluation stages should be separated.

A bad answer may result from poor retrieval even when the LLM is capable of answering the question.

Similarly, the correct evidence may be retrieved while the LLM still produces an incorrect answer.

## Faithfulness

Faithfulness measures whether the generated answer is supported by the retrieved context.

A faithful RAG system should avoid inventing information that does not appear in the retrieved documents.

For questions that cannot be answered using the knowledge base, the system should indicate that the answer is unknown rather than generating unsupported information.

## RAG Ablation Study

A useful experiment is to compare four retrieval configurations:

FAISS
vs.
BM25
vs.
Hybrid
vs.
Hybrid + Reranker

The same questions, documents, LLM, prompt, and final context size should be used when comparing the final generated answers.

This makes it easier to determine whether each additional retrieval component improves the system.

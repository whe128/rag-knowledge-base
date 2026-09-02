# Hybrid Retrieval and Reciprocal Rank Fusion

## What is Hybrid Retrieval?

Hybrid retrieval combines multiple retrieval methods.

A common design combines:

- Dense retrieval
- Sparse or lexical retrieval

In this project, dense retrieval is implemented using FAISS and BGE-M3, while lexical retrieval is implemented using BM25.

The two retrievers have different strengths.

FAISS focuses on semantic similarity.

BM25 focuses on lexical matching.

Combining them can improve retrieval robustness.

## Why Combine FAISS and BM25?

Consider two queries.

Query A:

"What does RRF mean?"

The query contains a specific technical term. BM25 may perform well because it can match the exact term.

Query B:

"How can several search rankings be combined into one ranking?"

This query expresses the concept without using the exact term "RRF".

Dense retrieval may perform better because the semantic meaning is similar to documents discussing Reciprocal Rank Fusion.

Therefore, the two retrieval methods can complement each other.

## Score Scale Problem

FAISS and BM25 produce relevance scores using different algorithms.

A FAISS similarity score and a BM25 score generally have different numerical scales.

For example:

FAISS:
0.82
0.79
0.75

BM25:
12.7
10.3
8.5

Directly adding these values is not necessarily meaningful.

The score distributions are different.

## Reciprocal Rank Fusion

Reciprocal Rank Fusion, commonly called RRF, combines ranked lists instead of directly combining raw scores.

For a document at rank r:

RRF_score = 1 / (k + r)

where k is a constant.

The total RRF score is the sum of the contributions from different retrieval systems.

For example, if a document is ranked first by FAISS and third by BM25:

RRF_score =
1 / (k + 1)

- 1 / (k + 3)

A document that appears near the top of multiple retrieval lists receives a larger combined score.

## RRF Constant

A commonly used RRF constant is:

k = 60

The constant reduces the influence of the exact rank difference while still rewarding documents that appear near the top of multiple lists.

## Hybrid Retrieval Pipeline

The hybrid retrieval pipeline is:

Query
→ FAISS
→ Dense Candidates

Query
→ BM25
→ Lexical Candidates

Dense Candidates

- Lexical Candidates
  → RRF
  → Hybrid Ranking
  → Top-K Documents

The final ranked documents can then be sent to a reranker.

## Benefits of RRF

RRF has several practical advantages:

1. It does not require retrieval scores to have the same scale.
2. It is simple to implement.
3. It combines multiple ranked lists.
4. It rewards documents that are consistently highly ranked.
5. It can be used without training an additional ranking model.

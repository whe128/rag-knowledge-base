import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json

from src.ingestion import load_documents, split_documents
from src.vector_store import load_vector_store
from src.bm25 import BM25Retriever
from src.hybrid import HybridRetriever
from src.reranker import Reranker
from src.generation import generate_answer

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

def get_judge_llm():
    return ChatOllama( model="qwen2.5:1.5b", temperature=0)

judge_prompt = ChatPromptTemplate.from_template(
    """
    You are an evaluator for a RAG question-answering system.
    Determine whether the generated answer correctly answers the question.
    Question: {question}
    Reference Answer: {reference_answer}
    Generated Answer: {generated_answer}
    Rules:
        - If the generated answer contains the main correct information, return CORRECT.
        - If the generated answer is factually wrong or misses the key information, return INCORRECT.
        - Different wording is acceptable.
        - Do not require the generated answer to exactly match the reference answer.
    Return ONLY one word:
    CORRECT
    or
    INCORRECT
    """
)
def evaluate_answer( question, reference_answer, generated_answer):
    llm = ChatOllama( model="qwen2.5:1.5b", temperature=0)
    chain = judge_prompt | llm
    response = chain.invoke({
        "question": question,
        "reference_answer": reference_answer,
        "generated_answer": generated_answer
    })
    return response.content.strip().upper() == "CORRECT"

QUESTIONS_FILE = "evaluation/questions.json"

# 0 - FAISS
# 1 - BM25
# 2 - Hybrid
# 3 - Hybrid + Reranker
RETRIEVAL_METHOD = 2

RETRIEVAL_K = 8
RERANK_TOP_K = 4


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_retrieval_results(
    query,
    faiss_retriever,
    bm25_retriever,
    hybrid_retriever_only,
    hybrid_retriever_plus_rerank,
    reranker,
):
    if RETRIEVAL_METHOD == 0:
        return faiss_retriever.invoke(query)

    elif RETRIEVAL_METHOD == 1:
        return bm25_retriever.retrieve(query)

    elif RETRIEVAL_METHOD == 2:
        return hybrid_retriever_only.hybrid_retrieve(query)

    elif RETRIEVAL_METHOD == 3:
        results = hybrid_retriever_plus_rerank.hybrid_retrieve(query)

        return reranker.rerank(
            query,
            results,
            top_k=RERANK_TOP_K,
        )

    else:
        raise ValueError("Invalid retrieval method")


def main():

    # =========================
    # 1. Load knowledge base
    # =========================

    documents = load_documents()
    chunks = split_documents(documents)

    print(f"Loaded {len(documents)} documents")
    print(f"Created {len(chunks)} chunks")

    # =========================
    # 2. Load FAISS
    # =========================

    vector_store = load_vector_store()

    # =========================
    # 3. Initialize retrievers
    # =========================
    faiss_retriever = vector_store.as_retriever(
            search_kwargs={"k": RERANK_TOP_K}
        )

    bm25_retriever = BM25Retriever(
        chunks,
        k=RERANK_TOP_K,
    )

    hybrid_retriever_only = HybridRetriever(
        chunks,
        k=RERANK_TOP_K,     # 4
        rrf_k=60,
    )

    hybrid_retriever_plus_rerank = HybridRetriever(
        chunks,
        k=RETRIEVAL_K,      # 8
        rrf_k=60,
    )

    reranker = Reranker()

    # =========================
    # 4. Load test dataset
    # =========================

    questions = load_questions()

    print(f"Loaded {len(questions)} test questions")

    # =========================
    # 5. Run evaluation
    # =========================

    correct = 0

    for i, item in enumerate(questions, start=1):

        question = item["question"]
        reference_answer = item["reference_answer"]

        print("\n" + "=" * 60)
        print(f"Question {i}/{len(questions)}")
        print(f"Q: {question}")

        # Retrieval
        results = get_retrieval_results(
            question,
            faiss_retriever,
            bm25_retriever,
            hybrid_retriever_only,
            hybrid_retriever_plus_rerank,
            reranker,
        )

        # Generation
        answer = generate_answer(
            question,
            results,
        )

        print(f"\nReference Answer:\n{reference_answer}")
        print(f"\nGenerated Answer:\n{answer}")

        # =========================
        # evaluation
        # =========================

        is_correct = evaluate_answer( question, reference_answer, answer)
        if is_correct:
            correct += 1
            print("Evaluation: CORRECT")
        else:
            print("Evaluation: INCORRECT")

    # =========================
    # 6. Calculate accuracy
    # =========================

    accuracy = correct / len(questions)

    print("\n" + "=" * 60)
    print("Evaluation Result")
    print("=" * 60)

    print(f"Correct: {correct}/{len(questions)}")
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()



from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

def get_llm(model_name = "qwen2.5:1.5b", temperature = 0):
    return ChatOllama(
        model = model_name,
        temperature = temperature,
    )

"""
build a prompt template for the RAG pipeline
the prompt template is used to generate the answer from the specified question and documents
"""
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful knowledge base assistant.
Answer the question using ONLY the information
provided in the context.
If the answer cannot be found in the context,
say "I don't know."
Context:
{context}
Question:
{question}
Answer:
"""
)

def generate_answer(question, documents):
    """
    generate answer from the specified question and documents
    """

    # crete the context from the documents
    context = "\n\n".join(
        doc.page_content for doc in documents
    )


    llm = get_llm()

    # | is the pipe operator, let the output of the prompt be the input of the llm
    chain = prompt | llm

    # invoke is the unified interface for the chain
    # prompt | llm is the runnable chain, which is a callable object
    # it can call the invoke method to generate the answer from the specified question and documents
    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return response.content

if __name__ == "__main__":
    print(generate_answer("What is the hewang's email adress?", []))

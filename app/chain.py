import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

def build_qa_chain(retriever):
    """Build the full RAG chain
    User question -> Retrieve chunks -> Fill prompt -> LLM -> Answer
    """
    # 1. Define the LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,   # t=0 for deterministic, factual 
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Define strict prompt template
    prompt_text = """
    You are a helpful customer service assistant for a local business.
    Use only the context provided below to answer the customers questions.
    If the answer is not found in the context, say:
    I'm sorry, I don't have that information. Please contact us directly.

    Do not make up answers. Do not use outside knowledge.
    Be friendly, concise and professional.

    context = {context}
    customer question = {question}
    Answer:"""

    prompt = PromptTemplate(
        template= prompt_text,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        retriever = retriever,
        chain_type = "stuff",
        chain_type_kwargs = {"prompt": prompt},
        return_source_documents = True
    )

    return qa_chain


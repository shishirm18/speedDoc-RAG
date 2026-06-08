import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def get_embeddings():
    """Initialize the open AI embeddings model
    This converts the text into numerical vectors (embeddings)
    """

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    return embeddings 

def create_vectorstore(chunks):
    """Take the document chunks, conver them to vectors
    and store them in FAISS index saved in disk
    """
    embeddings = get_embeddings()
    # Sends each chunks to openAI embeddings model
    # Gets back the 1536 D vector from each chunk
    # store all vectors in FAISS index in memory
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("vectorstore/faiss_index")
    print("Vector store created and saved to disk!")

    return vector_store


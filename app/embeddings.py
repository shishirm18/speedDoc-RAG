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

    return vector_store

def load_vectorstore():
    """Rather than processing the documents every time
    Just load the index from the vector store
    """
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        "vectorstore/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Vector store loaded from the disk")

    return vectorstore 

def vectorstore_exists():
    return os.path.exists("vectorstore/faiss_index")

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(file_path: str):
    """ load a document from a file path,
    supports pdf and text files,
    returns a list of langchain document objects
    """
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}. Accepted .pdf or .txt")
    
    documents = loader.load()
    return documents

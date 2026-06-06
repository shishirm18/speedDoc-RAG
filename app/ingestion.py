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
    print(f"Number of documents {len(documents)}")
    return documents

def split_documents(documents):
    """Split the document into smaller chunks,
    This is critial - LLMs have token limits,
    so we only send relevant chunks, not entire document.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500, # chunks = 500
        chunk_overlap = 50, # 50 char overlap between chunks(preserve context)
        separators = ["\n\n", "\n", ".", " ", ""] 
    )
    chunks = splitter.split_documents(documents)

    print(f"The document is split into {len(chunks)} chunks")
    return chunks

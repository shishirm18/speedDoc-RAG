# speedDoc-RAG
speedDoc-RAG is a RAG (Retrieval Augmented Generation) based application which helps to query the docs faster and gets instance reliable responses in seconds. 

# Dependencies
- langchain: The main RAG orchestration framework
- langchain-openai: Langchain's connector to OpenAI models
- langchain-community: Community loaders (PDF, text, etc)
- faiss-cpu: local vector store - stores & searches embedding
- pypdf: Reads & parses pdf files
- python-dotenv: Loads you .env API secret keys into app
- streamlit: Turns the python into webUI
- tiktoken: OpenAI's tokenizer - used internally by langchain

# RAG Architecture
* Step 1: Upload the documents (pdf or text)
          |
          V
* Step 2: Breaking down the document into smaller chunks (using RecursiveCharacterTextSplitting)
          |
          V
* Step 3: Convert the smaller chunks into vector embeddings (using openAI embeddings model: text-embedding-3-small)
          |
          V
* Step 4: 
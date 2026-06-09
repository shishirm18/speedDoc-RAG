from app.ingestion import load_documents, split_documents
from app.embeddings import create_vectorstore
# docs = load_documents("data/uploads/AngularQuestions.pdf")
docs = load_documents("data/uploads/test.txt")
# print(len(docs))
# print(docs)

chunks = split_documents(docs)
# vector_store = create_vectorstore(chunks)


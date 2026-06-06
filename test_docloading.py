from app.ingestion import load_documents

docs = load_documents("data/uploads/test.txt")
print(len(docs))
print(docs)
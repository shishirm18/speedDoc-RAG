from app.ingestion import load_documents, split_documents

# docs = load_documents("data/uploads/AngularQuestions.pdf")
docs = load_documents("data/uploads/test.txt")
# print(len(docs))
# print(docs)

chunks = split_documents(docs)
for i, chunk in enumerate(chunks):
    print(i, chunk.page_content)
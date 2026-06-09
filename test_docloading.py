from app.ingestion import load_documents, split_documents
from app.embeddings import create_vectorstore, load_vectorstore, vectorstore_exists
from app.retriever import get_retriever
from app.chain import build_qa_chain

# docs = load_documents("data/uploads/AngularQuestions.pdf")
print("Step1: Load documents and split into chunks")
docs = load_documents("data/uploads/gym.txt")
chunks = split_documents(docs)
print(len(chunks))

print("Step2: create vector store and save to disk")
vector_store = load_vectorstore()

print("Step3: retrieve top k, from this vector store")
retriever = get_retriever(vector_store, k=3)

print("Step4: Build the RAG chain with retrieved vector store!")
qa_chain = build_qa_chain(retriever)
print(qa_chain)
# print("Step5: Passing the question")
# qa_chain

result = qa_chain.invoke({"query": "What is the cost of student membership per month?"})
print(result['result'])

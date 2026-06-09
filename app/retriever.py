def get_retriever(vectorstore, k=2):
    """
    Create a retriever from vector store
    k=3 means, retrive the top 3 similar chunks
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={'k': k}
    )

    return retriever

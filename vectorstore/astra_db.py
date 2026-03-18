from langchain_community.vectorstores import FAISS

def init_db(embedding, docs):
    return FAISS.from_documents(docs, embedding)
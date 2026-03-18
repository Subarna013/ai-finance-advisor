from langchain_community.vectorstores import Chroma

def init_db(embedding, docs):
    return Chroma.from_documents(docs, embedding)
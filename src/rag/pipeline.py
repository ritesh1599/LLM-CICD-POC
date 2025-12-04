from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from .loader import load_documents
from .embeddings import get_embeddings

class PizzaRAG:
    def __init__(self):
        docs = load_documents()
        self.emb = get_embeddings()
        self.db = FAISS.from_texts(docs, self.emb)
        self.llm = OpenAI()

    def query(self, question):
        relevant_docs = self.db.similarity_search(question, k=3)
        context = "\n".join([d.page_content for d in relevant_docs])

        prompt = f"""
You are answering questions about a pizza company using the context below.
Context:
{context}

Question: {question}
Answer:
"""
        return self.llm(prompt)

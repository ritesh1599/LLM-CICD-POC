from fastapi import FastAPI
from pydantic import BaseModel
from rag.pipeline import PizzaRAG

app = FastAPI()
rag = PizzaRAG()

class Query(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask_question(q: Query):
    answer = rag.query(q.question)
    return {"answer": answer}

from rag.pipeline import PizzaRAG

def test_rag_basic():
    rag = PizzaRAG()
    response = rag.query("What sizes of pizza do you have?")
    assert len(response) > 0

import sys, os
sys.path.append(os.path.dirname(__file__))
from graph import build_graph

if __name__ == "__main__":
    app = build_graph()
    query = input("Ask a question about httpx: ")

    initial_state = {
        "query": query,
        "retrieved": [],
        "grade": "",
        "grade_reasoning": "",
        "answer": "",
        "retry_count": 0,
    }

    final_state = app.invoke(initial_state)

    print("\n" + "="*60)
    print("GRADE:", final_state["grade"], "-", final_state["grade_reasoning"])
    print("="*60)
    print("ANSWER:")
    print(final_state["answer"])
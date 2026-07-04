import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "agent"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))

from naive_rag import answer_question as naive_answer
from graph import build_graph

def get_naive_result(query):
    answer, retrieved = naive_answer(query, top_k=5)
    contexts = [str(r["chunk"].get("code") or r["chunk"].get("content")) for r in retrieved]
    return answer, contexts

def get_agentic_result(query, app):
    initial_state = {
        "query": query, "original_query": query, "retrieved": [],
        "grade": "", "grade_reasoning": "", "answer": "",
        "retry_count": 0, "max_retries": 2,
    }
    final_state = app.invoke(initial_state)
    contexts = [str(r["chunk"].get("code") or r["chunk"].get("content")) for r in final_state["retrieved"]]
    return final_state["answer"], contexts

if __name__ == "__main__":
    qa_path = os.path.join(os.path.dirname(__file__), "qa_set.json")
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_set = json.load(f)

    app = build_graph()

    naive_results = []
    agentic_results = []

    for i, item in enumerate(qa_set):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"\n[{i+1}/{len(qa_set)}] {q}")

        print("  Running naive RAG...")
        naive_ans, naive_ctx = get_naive_result(q)
        naive_results.append({
            "question": q, "answer": naive_ans, "contexts": naive_ctx, "ground_truth": gt
        })

        print("  Running agentic RAG...")
        agentic_ans, agentic_ctx = get_agentic_result(q, app)
        agentic_results.append({
            "question": q, "answer": agentic_ans, "contexts": agentic_ctx, "ground_truth": gt
        })

    out_dir = os.path.dirname(__file__)
    with open(os.path.join(out_dir, "naive_results.json"), "w", encoding="utf-8") as f:
        json.dump(naive_results, f, indent=2)
    with open(os.path.join(out_dir, "agentic_results.json"), "w", encoding="utf-8") as f:
        json.dump(agentic_results, f, indent=2)

    print("\nDone. Saved naive_results.json and agentic_results.json")
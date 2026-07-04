import json, os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

def load_for_ragas(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_dict({
        "question": [d["question"] for d in data],
        "answer": [d["answer"] for d in data],
        "contexts": [d["contexts"] for d in data],
        "ground_truth": [d["ground_truth"] for d in data],
    })

eval_dir = os.path.dirname(__file__)

print("Scoring naive RAG...")
naive_ds = load_for_ragas(os.path.join(eval_dir, "naive_results.json"))
naive_scores = evaluate(naive_ds, metrics=[faithfulness, answer_relevancy, context_precision])

print("Scoring agentic RAG...")
agentic_ds = load_for_ragas(os.path.join(eval_dir, "agentic_results.json"))
agentic_scores = evaluate(agentic_ds, metrics=[faithfulness, answer_relevancy, context_precision])

print("\n" + "="*60)
print("COMPARISON: Naive RAG vs Agentic (Self-Correcting) RAG")
print("="*60)
print(f"{'Metric':<25}{'Naive':<15}{'Agentic':<15}")
naive_df = naive_scores.to_pandas()
agentic_df = agentic_scores.to_pandas()

for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
    naive_avg = naive_df[metric].mean()
    agentic_avg = agentic_df[metric].mean()
    print(f"{metric:<25}{naive_avg:<15.4f}{agentic_avg:<15.4f}")

naive_df.to_csv(os.path.join(eval_dir, "naive_scores.csv"), index=False)
agentic_df.to_csv(os.path.join(eval_dir, "agentic_scores.csv"), index=False)
print("\nDetailed scores saved to naive_scores.csv and agentic_scores.csv")
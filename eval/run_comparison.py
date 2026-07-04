import sys, os, json
sys.path.append(os.path.dirname(__file__))
from custom_eval import evaluate_results

eval_dir = os.path.dirname(__file__)

with open(os.path.join(eval_dir, "naive_results.json"), "r", encoding="utf-8") as f:
    naive_results = json.load(f)
with open(os.path.join(eval_dir, "agentic_results.json"), "r", encoding="utf-8") as f:
    agentic_results = json.load(f)

print("Evaluating naive RAG...")
naive_faith, naive_rel, _, _ = evaluate_results(naive_results)

print("\nEvaluating agentic RAG...")
agentic_faith, agentic_rel, _, _ = evaluate_results(agentic_results)

print("\n" + "="*60)
print("FINAL COMPARISON: Naive RAG vs Self-Correcting Agentic RAG")
print("="*60)
print(f"{'Metric':<20}{'Naive':<15}{'Agentic':<15}{'Improvement':<15}")
print(f"{'Faithfulness':<20}{naive_faith:<15.4f}{agentic_faith:<15.4f}{(agentic_faith-naive_faith):+.4f}")
print(f"{'Answer Relevancy':<20}{naive_rel:<15.4f}{agentic_rel:<15.4f}{(agentic_rel-naive_rel):+.4f}")
import os, json, re
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "agent"))
from llm_utils import safe_invoke
import time
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))

FAITHFULNESS_PROMPT = """Given a context and an answer, determine what fraction of claims in the answer are actually supported by the context.

Context:
{context}

Answer:
{answer}

Respond in EXACTLY this format:
SCORE: <a number between 0.0 and 1.0>
REASONING: <one sentence>
"""

RELEVANCY_PROMPT = """Given a question and an answer, rate how well the answer actually addresses the question (regardless of factual correctness).

Question: {question}
Answer: {answer}

Respond in EXACTLY this format:
SCORE: <a number between 0.0 and 1.0>
REASONING: <one sentence>
"""

def extract_score(text):
    match = re.search(r"SCORE:\s*([\d.]+)", text)
    return float(match.group(1)) if match else 0.0

def score_faithfulness(answer, contexts):
    context_str = "\n\n".join(contexts)[:4000]
    prompt = FAITHFULNESS_PROMPT.format(context=context_str, answer=answer)
    response = safe_invoke(llm, [("user", prompt)])
    time.sleep(2)
    return extract_score(response.content)

def score_relevancy(question, answer):
    prompt = RELEVANCY_PROMPT.format(question=question, answer=answer)
    response = safe_invoke(llm, [("user", prompt)])
    time.sleep(2)
    return extract_score(response.content)

def evaluate_results(results):
    faithfulness_scores = []
    relevancy_scores = []
    for item in results:
        f_score = score_faithfulness(item["answer"], item["contexts"])
        r_score = score_relevancy(item["question"], item["answer"])
        faithfulness_scores.append(f_score)
        relevancy_scores.append(r_score)
        print(f"  Q: {item['question'][:50]}... | Faithfulness: {f_score:.2f} | Relevancy: {r_score:.2f}")

    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
    avg_relevancy = sum(relevancy_scores) / len(relevancy_scores)
    return avg_faithfulness, avg_relevancy, faithfulness_scores, relevancy_scores
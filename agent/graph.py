import sys, os
from llm_utils import safe_invoke
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))

from langgraph.graph import StateGraph, END
from state import RAGState
from hybrid_search import hybrid_search
from grading import grade_retrieval
from query_rewriter import rewrite_query
from prompt_templates import RAG_SYSTEM_PROMPT, build_rag_prompt

from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))

def retrieve_node(state: RAGState) -> RAGState:
    print(f"[retrieve] Searching for: {state['query']}")
    results = hybrid_search(state["query"], top_k=5)
    state["retrieved"] = results
    return state

def grade_node(state: RAGState) -> RAGState:
    print("[grade] Evaluating retrieval quality...")
    grade, reasoning = grade_retrieval(state["query"], state["retrieved"])
    state["grade"] = grade
    state["grade_reasoning"] = reasoning
    print(f"[grade] Result: {grade} — {reasoning}")
    return state

def correct_node(state: RAGState) -> RAGState:
    print(f"[correct] Retry {state['retry_count'] + 1}/{state['max_retries']} — rewriting query...")
    new_query = rewrite_query(state["query"], state["grade_reasoning"])
    print(f"[correct] Rewritten query: {new_query}")
    state["query"] = new_query
    state["retry_count"] += 1
    return state

def answer_node(state: RAGState) -> RAGState:
    print("[answer] Generating final answer...")
    prompt = build_rag_prompt(state["original_query"], state["retrieved"])
    messages = [("system", RAG_SYSTEM_PROMPT), ("user", prompt)]
    response = safe_invoke(llm, messages)
    state["answer"] = response.content
    return state

def fallback_node(state: RAGState) -> RAGState:
    print("[fallback] Could not find sufficient context after retries.")
    state["answer"] = (
        f"I wasn't able to find sufficient information in the httpx codebase/docs "
        f"to confidently answer: \"{state['original_query']}\". "
        f"The closest context found was judged as '{state['grade']}': {state['grade_reasoning']}"
    )
    return state

def route_after_grade(state: RAGState) -> str:
    if state["grade"] == "relevant":
        return "answer"
    if state["retry_count"] >= state["max_retries"]:
        if state["grade"] == "partial":
            return "answer"  # best-effort answer using what we have, rather than refusing
        return "fallback"  # only truly irrelevant context triggers fallback
    return "correct"

def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("correct", correct_node)
    graph.add_node("answer", answer_node)
    graph.add_node("fallback", fallback_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"answer": "answer", "correct": "correct", "fallback": "fallback"},
    )
    graph.add_edge("correct", "retrieve")  # loop back after rewriting
    graph.add_edge("answer", END)
    graph.add_edge("fallback", END)

    return graph.compile()
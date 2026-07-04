import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))

from langgraph.graph import StateGraph, END
from state import RAGState
from hybrid_search import hybrid_search
from grading import grade_retrieval
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

def answer_node(state: RAGState) -> RAGState:
    print("[answer] Generating final answer...")
    prompt = build_rag_prompt(state["query"], state["retrieved"])
    messages = [("system", RAG_SYSTEM_PROMPT), ("user", prompt)]
    response = llm.invoke(messages)
    state["answer"] = response.content
    return state

def route_after_grade(state: RAGState) -> str:
    # Placeholder routing for today — tomorrow this will trigger query rewriting
    if state["grade"] == "irrelevant":
        return "answer"  # tomorrow: "correct" instead
    return "answer"

def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, {"answer": "answer"})
    graph.add_edge("answer", END)

    return graph.compile()
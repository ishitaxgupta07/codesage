import sys, os
from llm_utils import safe_invoke
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))

from hybrid_search import hybrid_search
from prompt_templates import RAG_SYSTEM_PROMPT, build_rag_prompt

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))

def answer_question(query, top_k=5):
    retrieved = hybrid_search(query, top_k=top_k)
    prompt = build_rag_prompt(query, retrieved)

    messages = [
        ("system", RAG_SYSTEM_PROMPT),
        ("user", prompt),
    ]

    response = safe_invoke(llm, messages)
    return response.content, retrieved
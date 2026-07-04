import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))

GRADE_PROMPT = """You are a strict grader evaluating whether retrieved context is sufficient to answer a question.

Question: {query}

Retrieved context summaries:
{context_summaries}

Judge the retrieved context and respond in EXACTLY this format (no extra text):
GRADE: <relevant|partial|irrelevant>
REASONING: <one sentence explaining why>

- "relevant": the context clearly contains enough information to fully answer the question
- "partial": the context is somewhat related but missing key details
- "irrelevant": the context does not meaningfully address the question
"""

def summarize_chunk(chunk):
    if chunk.get("type") == "doc_section":
        return chunk.get("content", "")[:200]
    else:
        return f"{chunk.get('name','')}: {chunk.get('docstring','')[:150]}"

def grade_retrieval(query, retrieved_chunks):
    summaries = "\n".join(
        f"- {summarize_chunk(item['chunk'])}" for item in retrieved_chunks
    )
    prompt = GRADE_PROMPT.format(query=query, context_summaries=summaries)

    response = llm.invoke([("user", prompt)])
    text = response.content.strip()

    grade = "irrelevant"
    reasoning = text
    for line in text.splitlines():
        if line.upper().startswith("GRADE:"):
            grade = line.split(":", 1)[1].strip().lower()
        if line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()

    return grade, reasoning
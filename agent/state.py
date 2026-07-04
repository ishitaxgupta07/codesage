from typing import TypedDict, List, Dict, Any

class RAGState(TypedDict):
    query: str
    retrieved: List[Dict[str, Any]]
    grade: str          # "relevant", "irrelevant", "partial"
    grade_reasoning: str
    answer: str
    retry_count: int
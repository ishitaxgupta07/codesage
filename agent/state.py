from typing import TypedDict, List, Dict, Any

class RAGState(TypedDict):
    query: str
    original_query: str
    retrieved: List[Dict[str, Any]]
    grade: str
    grade_reasoning: str
    answer: str
    retry_count: int
    max_retries: int
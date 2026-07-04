import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))

REWRITE_PROMPT = """The following query did not retrieve good results from a codebase/documentation search:

Original query: {query}
Why it failed: {reasoning}

Rewrite this query to be more likely to retrieve relevant results. Consider:
- Using more specific technical terms
- Breaking down vague phrasing into concrete keywords
- Removing conversational filler

Respond with ONLY the rewritten query, nothing else.
"""

def rewrite_query(original_query, reasoning):
    prompt = REWRITE_PROMPT.format(query=original_query, reasoning=reasoning)
    response = llm.invoke([("user", prompt)])
    return response.content.strip().strip('"')
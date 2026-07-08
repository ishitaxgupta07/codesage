import os

def clean_path(filepath):
    normalized = os.path.normpath(filepath)
    parts = normalized.split(os.sep)
    if "target_repo" in parts:
        idx = parts.index("target_repo")
        return "/".join(parts[idx+1:])
    return os.path.basename(filepath)

RAG_SYSTEM_PROMPT = """You are CodeSage, an assistant that answers questions about the httpx Python library using ONLY the provided context chunks.

Rules:
- Only use information from the provided context. Do not use outside knowledge about httpx.
- If the context doesn't contain enough information to answer, say so clearly instead of guessing.
- After each claim, cite the source using the format [Source: <file>, line <start_line>] for code, or [Source: <file>] for docs.
- Be concise and technical. This is for a developer audience.
"""

def build_rag_prompt(query, retrieved_chunks):
    context_blocks = []
    for i, item in enumerate(retrieved_chunks):
        chunk = item["chunk"]
        if chunk.get("type") == "doc_section":
            source = clean_path(chunk.get("file", "unknown"))
            text = chunk.get("content", "")
            context_blocks.append(f"[Chunk {i+1}] Source: {source}\n{text}")
        else:
            source = f"{clean_path(chunk.get('file','unknown'))}, line {chunk.get('start_line','?')}"
            text = chunk.get("code", "")
            docstring = chunk.get("docstring", "")
            context_blocks.append(f"[Chunk {i+1}] Source: {source}\nDocstring: {docstring}\nCode:\n{text}")

    context_str = "\n\n---\n\n".join(context_blocks)

    user_prompt = f"""Context chunks retrieved from the httpx codebase and docs:

{context_str}

---

Question: {query}

Answer the question using only the context above, with citations."""

    return user_prompt
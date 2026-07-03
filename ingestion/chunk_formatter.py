def chunk_to_text(chunk):
    if chunk.get("type") == "doc_section":
        return chunk.get("content", "")
    else:
        # Code chunk: combine docstring + code for richer semantic meaning
        name = chunk.get("name", "")
        docstring = chunk.get("docstring", "")
        code = chunk.get("code", "")
        return f"{name}\n{docstring}\n{code}"
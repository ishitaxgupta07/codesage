import ast

def chunk_python_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    chunks = []
    lines = source.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = node.end_lineno
            code_snippet = "\n".join(lines[start:end])
            docstring = ast.get_docstring(node) or ""

            chunks.append({
                "file": filepath,
                "type": type(node).__name__,
                "name": node.name,
                "start_line": node.lineno,
                "end_line": node.end_lineno,
                "code": code_snippet,
                "docstring": docstring,
            })
    return chunks
import re

def chunk_markdown_file(filepath, max_chars=1500):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    sections = re.split(r'\n(?=#{1,3} )', content)
    chunks = []
    for section in sections:
        if len(section.strip()) < 20:
            continue
        chunks.append({
            "file": filepath,
            "type": "doc_section",
            "content": section.strip()[:max_chars],
        })
    return chunks
import json
import sys
import os

sys.path.append(os.path.dirname(__file__))

from file_filter import get_source_files
from chunker import chunk_python_file
from doc_chunker import chunk_markdown_file

repo_path = os.path.join(os.path.dirname(__file__), "..", "target_repo")

all_chunks = []

py_files = get_source_files(repo_path, (".py",))
print(f"Found {len(py_files)} Python files")
for f in py_files:
    all_chunks.extend(chunk_python_file(f))

md_files = get_source_files(repo_path, (".md",))
md_files = [f for f in md_files if "CHANGELOG" not in f.upper()]  # add this line
print(f"Found {len(md_files)} Markdown files")
for f in md_files:
    all_chunks.extend(chunk_markdown_file(f))

print(f"Total chunks created: {len(all_chunks)}")

output_path = os.path.join(os.path.dirname(__file__), "..", "chunks.json")
with open(output_path, "w", encoding="utf-8") as out:
    json.dump(all_chunks, out, indent=2)

print(f"Saved to {output_path}")
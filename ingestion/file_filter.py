import os

EXCLUDE_DIRS = {"tests", "test", "__pycache__", ".git", "examples", ".github", "scripts"}

def get_source_files(repo_path, extensions=(".py",)):
    files = []
    for root, dirs, filenames in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in filenames:
            if f.endswith(extensions):
                files.append(os.path.join(root, f))
    return files
import sys, os
sys.path.append(os.path.dirname(__file__))
from naive_rag import answer_question

if __name__ == "__main__":
    query = input("Ask a question about httpx: ")
    answer, retrieved = answer_question(query)

    print("\n" + "="*60)
    print("ANSWER:")
    print("="*60)
    print(answer)

    print("\n" + "="*60)
    print("RETRIEVED SOURCES:")
    print("="*60)
    for i, r in enumerate(retrieved):
        chunk = r["chunk"]
        print(f"{i+1}. {chunk.get('file')} - {chunk.get('name', chunk.get('type'))}")
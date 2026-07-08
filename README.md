# CodeSage

**Self-correcting agentic RAG system for codebase Q&A** — built over the `httpx` Python library, using AST-aware chunking, hybrid retrieval, and an LLM-graded correction loop.

**[Live demo →](https://aicodesage.streamlit.app/)**

---

## What it does

CodeSage answers natural-language questions about the `httpx` codebase and its documentation. Unlike naive RAG — which retrieves once and answers regardless of quality — CodeSage evaluates its own retrieval before committing to an answer. If the retrieved context is insufficient, it automatically rewrites the query and retries (up to 2 times) before either generating an answer or honestly stating it couldn't find enough information, rather than hallucinating.

---

## Architecture
retrieve → grade → [relevant] → answer
│
└── [partial / irrelevant] → correct (rewrite query) → retrieve (loop, max 2 retries)
│
(after max retries) [partial] → answer
[irrelevant] → fallback (honest refusal)

**Ingestion**
- AST-based Python chunking using the `ast` module — splits by function/class boundaries, keeping each chunk (code + docstring) semantically complete rather than cut at arbitrary character limits
- Markdown documentation chunked by header sections
- 789 total chunks from 23 source files + 25 documentation files (changelog and test files excluded as noise)

**Retrieval**
- Hybrid search: BM25 (keyword/exact-match) + dense vector search (`BAAI/bge-small-en-v1.5`, run locally — zero embedding API cost)
- Results merged via Reciprocal Rank Fusion (RRF), which combines rankings from both methods without needing to normalize incompatible score scales
- Verified empirically: exact-match queries (e.g., a specific function name) correctly rank the true match first; conceptual queries retrieve semantically relevant results even without exact keyword overlap

**Agentic loop**
- Built with LangGraph as an explicit state machine, not a single linear chain
- A dedicated grading node (separate LLM call) judges whether retrieved context is sufficient — it does not answer the question, only evaluates retrieval quality
- A query-rewriting node activates when grading returns "partial" or "irrelevant," rewriting the query and looping back to retrieval
- Bounded by a configurable retry limit (default: 2) to prevent infinite loops
- LLM: Llama 3.3 70B via Groq

---

## Evaluation

Compared against a naive (single-pass, no correction) RAG baseline across a hand-written 10-question benchmark spanning exact-match, conceptual, multi-part, and deliberately out-of-scope queries.

**Note on methodology:** RAGAS was the original planned evaluation tool, but its dependency chain (specifically `langchain-community`'s eager import of Google Cloud/VertexAI packages) proved unresolvable in this Windows + Groq environment even after multiple targeted fixes. Rather than force a fragile dependency chain, a custom LLM-as-judge evaluator was built from scratch (faithfulness and answer-relevancy scoring via structured prompting), which also allowed full control over the scoring logic — including catching and fixing a real bug where the initial scorer penalized honest refusals as "unfaithful."

| Metric | Naive RAG | Agentic RAG |
|---|---|---|
| Faithfulness | 0.90 | 0.80 |
| Answer Relevancy | 0.88 | 0.87 |

**Honest analysis (not a clean win, and that's the interesting part):**
- The faithfulness gap traces almost entirely to a single question (WebSocket support) where the judge scored materially identical answer content inconsistently between the two pipelines. Confirmed via two independent full evaluation runs landing within 0.01 of each other — ruling out random noise and pointing instead to LLM-as-judge sensitivity on borderline single-source answers, a known limitation of automated evaluation, not a flaw in the retrieval/correction system itself.
- A genuine, explainable win: on "How do I retry a request automatically on failure?" — the hardest, most ambiguous question in the benchmark — relevancy improved from **0.40 (naive) to 0.90 (agentic)**. The correction loop's query rewriting surfaced a third-party library (`httpx-retries`) and specific parameters that single-shot retrieval missed entirely.
- Out-of-scope questions (e.g., gRPC support, which httpx does not have) are correctly and honestly refused by both pipelines rather than hallucinated — refusals structurally score lower on "relevancy" (which measures whether the question was directly answered) even when they are the objectively correct response.

---

## Tech stack

Python · LangGraph · LangChain · Qdrant (vector database) · Groq (LLM inference, Llama 3.3 70B) · sentence-transformers (local embeddings) · rank-bm25 · Streamlit

---

## Live demo UI

The deployed app includes:
- A live execution trace panel showing each graph step (retrieve → grade → correct → answer) as it happens
- Retrieved chunk previews with relative relevance ranking
- The evaluation comparison above, displayed transparently within the app itself

---

## Running locally

```bash
git clone https://github.com/ishitaxgupta07/codesage.git
cd codesage
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`) with:
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key

Then run:
```bash
python ingestion/run_embedding.py   # first time only, populates the vector DB
streamlit run app/streamlit_app.py
```

---

## Known limitations

- LLM-as-judge evaluation has documented consistency limitations on borderline, single-source-answer questions (see Evaluation section above)
- Self-correction trades latency for reliability — queries requiring 2 retry cycles can take ~10-12 seconds
- Currently scoped to a single indexed repository (`httpx`); not yet generalized to arbitrary codebases
- Free-tier LLM inference (Groq) is subject to daily rate limits, which affected evaluation run timing during development but does not affect normal interactive use

---

## What I'd build next

- Multi-repository support (index and query across multiple codebases)
- Tool-use extension: execute retrieved code snippets to verify correctness before presenting an answer
- Human-in-the-loop evaluation to validate the LLM-as-judge findings against real user judgments

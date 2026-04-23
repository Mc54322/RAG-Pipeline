# RAG Pipeline — From Scratch

A production-style Retrieval-Augmented Generation (RAG) pipeline built without frameworks like LangChain. Every component is implemented directly to demonstrate a clear understanding of the underlying mechanics.

> Built as a portfolio project targeting ML/AI engineering roles.

---

## Why From Scratch?

Most RAG tutorials hide complexity behind abstractions. This project builds each layer explicitly so the trade-offs at every step are visible and deliberate.

---

## Stack

| Layer | Choice | Reason |
|---|---|---|
| PDF parsing | PyMuPDF | Fast, no Java dependency |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | Free, runs locally, strong baseline |
| Vector store | FAISS | Battle-tested, no infra needed |
| Keyword search | BM25 (`rank-bm25`) | Catches exact terms embeddings miss |
| Hybrid fusion | Reciprocal Rank Fusion | Scale-free, no hyperparameters to tune |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Joint query-passage scoring, higher precision |
| Evaluation | SQuAD token F1 + exact match | No API key needed, standard QA metrics |
| Streaming | Anthropic SDK streaming + FastAPI SSE | Incremental token delivery, lower perceived latency |
| Memory | Fixed-capacity FIFO conversation store | Multi-turn dialogue without blowing context window |
| LLM | Anthropic Claude | Best-in-class retrieval-augmented generation |
| API | FastAPI | Async, typed, auto-docs |

---

## Getting Started

**Requirements:** Python 3.11+

```bash
git clone https://github.com/Mc54322/RAG-Pipeline.git
cd rag-pipeline

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

---

## Project Structure

```
rag-pipeline/
├── api/
│   ├── main.py             # FastAPI app — /health, /ingest, /query, /query/stream, /chat
│   └── models.py           # Pydantic request/response schemas
├── frontend/
│   ├── index.html          # Single-page UI
│   ├── style.css           # Styles
│   └── app.js              # Upload, query, and render logic
├── src/
│   ├── ingestion.py        # PDF loading and text cleaning
│   ├── chunker.py          # Fixed-size and sentence-boundary chunking
│   ├── embedder.py         # sentence-transformers wrapper
│   ├── vector_store.py     # FAISS index with save/load
│   ├── retriever.py        # Query → ranked chunks
│   ├── prompt.py           # Prompt assembly from chunks + query
│   ├── generator.py        # Anthropic API wrapper (generate, stream, generate_with_history)
│   ├── memory.py           # Fixed-capacity conversation history
│   ├── document_store.py   # Multi-document registry with add/remove/rebuild
│   └── benchmark.py        # Retrieval metrics and timing utilities
├── scripts/
│   └── run_benchmark.py    # CLI benchmark runner
├── tests/
│   ├── conftest.py             # Session-scoped embedder + reranker fixtures
│   ├── test_ingestion.py       # PDF loading and text cleaning
│   ├── test_chunker.py         # Fixed-size and sentence-boundary chunking
│   ├── test_embedder.py        # Embedding shape, dtype, and semantic tests
│   ├── test_vector_store.py    # FAISS search, persistence, integration
│   ├── test_retriever.py       # Retriever unit + integration + full pipeline
│   ├── test_bm25.py            # BM25 tokenisation and search
│   ├── test_hybrid.py          # RRF fusion unit + hybrid integration
│   ├── test_reranker.py        # Cross-encoder unit + integration
│   ├── test_prompt.py          # Prompt assembly and system prompt
│   ├── test_generator.py       # generate, stream, generate_with_history
│   ├── test_memory.py          # ConversationMemory FIFO eviction
│   ├── test_document_store.py  # DocumentStore add/remove/rebuild
│   ├── test_evaluator.py       # SQuAD metrics + PipelineEvaluator
│   ├── test_benchmark.py       # Retrieval metrics and timing
│   └── test_api.py             # All FastAPI endpoints
├── Procfile                # Railway / Heroku start command
├── railway.json            # Railway deployment config
├── runtime.txt             # Python version pin
├── .env.example
└── requirements.txt
```

---

## Ingestion & Chunking

### PDF Ingestion (`src/ingestion.py`)

Extracts text from PDFs using PyMuPDF and normalises the output:

```python
from src.ingestion import load_pdf, clean_text

raw = load_pdf("paper.pdf")
text = clean_text(raw)
```

### Chunking (`src/chunker.py`)

Two strategies are provided:

**Fixed-size with overlap** — fast, predictable, good baseline:
```python
from src.chunker import chunk_text

chunks = chunk_text(text, chunk_size=512, overlap=64, source="paper.pdf")
```

**Sentence-boundary** — avoids mid-sentence cuts, better retrieval quality. This is the default used by the API:
```python
from src.chunker import chunk_by_sentence

chunks = chunk_by_sentence(text, max_chunk_size=512, source="paper.pdf")
```

Each `Chunk` carries `text`, `index`, `start_char`, `end_char`, and `metadata`.

---

## Embeddings & Vector Store

### Embedder (`src/embedder.py`)

Converts text into dense vectors using a local model — no API calls, no cost:

```python
from src.embedder import Embedder

embedder = Embedder()  # downloads all-MiniLM-L6-v2 on first run, cached after

embeddings = embedder.embed([c.text for c in chunks])  # shape: (n, 384)
query_vec  = embedder.embed_one("What is the capital of France?")
```

All vectors are L2-normalised, so similarity scores are cosine similarities in `[-1, 1]`.

### Vector Store (`src/vector_store.py`)

A FAISS index that stores embeddings alongside their source chunks and supports similarity search and disk persistence:

```python
from src.vector_store import VectorStore

store = VectorStore(dimension=embedder.dimension)
store.add(chunks, embeddings)

results = store.search(query_vec, k=5)
for chunk, score in results:
    print(f"{score:.3f}  {chunk.text[:80]}")

store.save("my_index/")
store = VectorStore.load("my_index/")  # fully restored
```

---

## Retrieval, Prompting & Generation

### Retriever (`src/retriever.py`)

Bridges the embedder and vector store — takes a plain text query and returns ranked chunks:

```python
from src.retriever import Retriever

retriever = Retriever(embedder, store)
results = retriever.retrieve("What causes climate change?", k=5, min_score=0.3)
```

`min_score` is a cosine similarity threshold. Setting it to `0.3–0.4` filters out weakly related chunks before they reach the prompt.

### Prompt Builder (`src/prompt.py`)

Formats retrieved chunks and the user's question into a structured prompt:

```python
from src.prompt import build_prompt, get_system_prompt

prompt = build_prompt(query, results, max_context_chars=6000)
system = get_system_prompt()
```

The prompt tells the model to answer only from the provided context and admit when it can't — reducing hallucination.

### Generator (`src/generator.py`)

Sends the constructed prompt to Claude and returns the answer:

```python
from src.generator import Generator

generator = Generator()  # reads ANTHROPIC_API_KEY from .env
answer = generator.generate(prompt, system=system)
print(answer)
```

**Streaming** — receive tokens as they arrive instead of waiting for the full response:

```python
for chunk in generator.stream(prompt, system=system):
    print(chunk, end="", flush=True)
```

**Multi-turn generation** — pass full conversation history for follow-up questions:

```python
messages = memory.get_messages() + [{"role": "user", "content": rag_prompt}]
answer = generator.generate_with_history(messages, system=system)
```

### Conversation Memory (`src/memory.py`)

Fixed-capacity FIFO store of (user, assistant) turn pairs. Stores bare questions and answers — not full RAG prompts — to keep the context window manageable:

```python
from src.memory import ConversationMemory

memory = ConversationMemory(max_turns=10)
memory.add_exchange("What is Python?", "Python is a high-level programming language.")
memory.add_exchange("Who created it?", "Guido van Rossum created Python.")

messages = memory.get_messages()
# [{"role": "user", "content": "What is Python?"}, {"role": "assistant", ...}, ...]

memory.clear()  # start a fresh session
```

When `max_turns` is exceeded, the oldest exchange is evicted automatically.

### Document Store (`src/document_store.py`)

Tracks all ingested documents so the index can be rebuilt after any add or remove:

```python
from src.document_store import DocumentStore

store = DocumentStore(dimension=embedder.dimension)

# Add multiple documents
store.add("paper.pdf", chunks_a, embeddings_a)
store.add("notes.pdf", chunks_b, embeddings_b)

# Build the combined FAISS index
vector_store = store.build_vector_store()
retriever    = Retriever(embedder, vector_store)

# Inspect what's indexed
for doc in store.list_documents():
    print(doc.source, doc.chunk_count, doc.ingested_at)

# Remove a document and rebuild
store.remove("paper.pdf")
vector_store = store.build_vector_store()
```

Re-ingesting the same filename replaces the old record — safe for updating documents.

### Full pipeline

```python
results = retriever.retrieve(query, k=5)
prompt  = build_prompt(query, results)
answer  = generator.generate(prompt, system=get_system_prompt())
```

---

## Frontend

A lightweight single-page interface served directly by FastAPI at `/`.

- Drag-and-drop (or browse) PDF upload — multiple files supported, uploaded sequentially
- Status message reflects the backend's running totals (total documents and chunks in index)
- "Clear selection" button wipes the backend index via `POST /reset` and resets the UI
- Configurable `k` (passages) and `min_score` threshold
- Cmd/Ctrl+Enter to submit
- Collapsible source passages colour-coded by similarity score (green ≥ 0.5, yellow ≥ 0.25, red < 0.25)
- XSS-safe rendering of chunk text

---

## REST API

### Running locally

```bash
source .venv/bin/activate
uvicorn api.main:app --reload
# UI at http://localhost:8000
# Interactive API docs at http://localhost:8000/docs
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service status and index size |
| `POST` | `/ingest` | Upload a PDF and add it to the index |
| `GET` | `/documents` | List all ingested documents |
| `DELETE` | `/documents/{source}` | Remove a document and rebuild the index |
| `POST` | `/reset` | Clear all documents and reset the pipeline |
| `POST` | `/query` | Ask a question, get an answer |
| `POST` | `/query/stream` | Ask a question, receive answer as SSE stream |
| `POST` | `/chat` | Ask a question with conversation memory |
| `POST` | `/chat/clear` | Clear the conversation history |

### Example

```bash
# Ingest a PDF
curl -X POST http://localhost:8000/ingest \
     -F "file=@paper.pdf"

# Ask a question
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the main argument?", "k": 5}'
```

Response from `/query`:
```json
{
  "answer": "The main argument is...",
  "sources": [
    {"text": "...", "score": 0.8241, "source": "paper.pdf"},
    {"text": "...", "score": 0.7103, "source": "paper.pdf"}
  ]
}
```

---

## Deployment (Railway)

1. Push the repo to GitHub
2. Create a new Railway project → **Deploy from GitHub repo**
3. Add `ANTHROPIC_API_KEY` in Railway → Variables
4. Railway auto-detects `Procfile` and starts the server

The embedding model downloads on first boot (~80 MB, cached after).

---

## Benchmarking

The benchmark evaluates retrieval quality and ingestion speed across both
chunking strategies. Corpus: five topic-specific PDFs in `data/benchmark/`
(machine learning, climate change, French Revolution, DNA, Python) — each
a short document of ~4 focused paragraphs. Queries are deliberate paraphrases
so the model must rely on semantic understanding, not keyword overlap.

No API key required — benchmarks only test the embedder + FAISS stack.

```bash
# Generate the benchmark PDFs (one-time)
python scripts/create_benchmark_data.py

# Run the evaluation
python scripts/run_benchmark.py
```

**Corpus:** 5 PDFs · 1,064 words · 5 evaluation queries  
**Hardware:** Apple M-series CPU (no GPU)

### Dense retrieval — chunking strategy comparison

| Strategy | P@1 | R@1 | P@3 | R@3 | P@5 | R@5 | MRR | Hit Rate@5 |
|---|---|---|---|---|---|---|---|---|
| Fixed-size (512 chars, 64 overlap) | 1.000 | 0.290 | 0.933 | 0.803 | 0.680 | 0.960 | 1.000 | 1.000 |
| Sentence boundary (max 512 chars) | 0.800 | 0.152 | 0.933 | 0.556 | 0.880 | 0.876 | 0.900 | 1.000 |

### Dense vs hybrid vs reranked — fixed-size chunking

BM25 keyword search fused with dense embeddings via Reciprocal Rank Fusion (RRF, k=60).
Re-ranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`, fetches 10 candidates then re-ranks to k.

| Strategy | P@1 | R@1 | P@3 | R@3 | P@5 | R@5 | MRR | Hit Rate@5 |
|---|---|---|---|---|---|---|---|---|
| Dense only | 1.000 | 0.290 | 0.933 | 0.803 | 0.680 | 0.960 | 1.000 | 1.000 |
| Hybrid (BM25 + dense) | 1.000 | 0.290 | 0.867 | 0.763 | 0.640 | 0.893 | 1.000 | 1.000 |
| Dense + reranker | 1.000 | 0.290 | 0.867 | 0.763 | 0.680 | 0.960 | 1.000 | 1.000 |

All three strategies achieve MRR=1.0 and Hit Rate=1.0 — the first-stage retrieval is already
saturated on this focused five-topic corpus. The value of re-ranking surfaces on larger,
noisier corpora where the first-stage returns mixed-relevance candidates.

### Per-query breakdown — fixed-size, k=1

| Query | Dense | Hybrid (RRF) | Reranked (logit) | Relevant? |
|---|---|---|---|---|
| How do neural networks learn from data? | 0.606 | 0.033 | 7.73 | ✓ |
| What is responsible for global warming? | 0.669 | 0.033 | 3.51 | ✓ |
| What triggered the French Revolution? | 0.703 | 0.033 | 3.95 | ✓ |
| What is the structure of DNA? | 0.638 | 0.033 | 6.78 | ✓ |
| Why is Python popular for machine learning? | 0.642 | 0.032 | 4.34 | ✓ |

Dense scores are cosine similarities (0–1). Hybrid scores are RRF values (~0.033 for
two retrievers with k=60). Reranked scores are raw cross-encoder logits (unbounded) —
each column is on a different scale and they are not directly comparable to each other.

### Ingestion timing

| Strategy | Chunks | Embedding | Total |
|---|---|---|---|
| Fixed-size (512 chars, 64 overlap) | 17 | 35 ms | 35 ms |
| Sentence boundary (max 512 chars) | 27 | 52 ms | 52 ms |

The embedding step dominates. Chunking and FAISS indexing are negligible at this scale.

---

## Tests

```bash
pytest tests/ -v
# 385 passed
```

---

## End-to-End Evaluation

`src/evaluator.py` measures the quality of the full pipeline — retrieval AND generation — using structured QA pairs with reference answers.

```python
from src.evaluator import PipelineEvaluator, QAPair

dataset = [
    QAPair(
        question="How do neural networks learn?",
        reference_answer="Neural networks learn using gradient descent to minimise a loss function.",
        relevant_keywords=["gradient descent", "loss function"],
    ),
    ...
]

evaluator = PipelineEvaluator(retriever, generator, k=5)
report = evaluator.evaluate(dataset)

print(f"Context Hit Rate : {report.context_hit_rate:.2%}")
print(f"Mean Token F1    : {report.mean_answer_f1:.2%}")
print(f"Exact Match      : {report.exact_match_rate:.2%}")
```

### Metrics

| Metric | What it measures |
|---|---|
| **Context Hit Rate** | Fraction of questions where retrieval returned at least one relevant chunk |
| **Token F1** | Token-level F1 of the generated answer vs reference (SQuAD metric — partial credit) |
| **Exact Match** | Fraction of questions where the answer is exactly right after normalisation |

Token F1 and Exact Match are computed without API calls using the SQuAD evaluation method: lowercase, strip punctuation and articles, then compare token bags. This lets you evaluate answer quality locally with no cost.

---

## Design Decisions

**No LangChain.** The goal is to understand every component, not to configure abstractions. LangChain is a valid production choice — this project is a learning and demonstration exercise.

**Two chunking strategies.** Fixed-size chunking is easy to reason about and tune. Sentence-boundary chunking produces cleaner retrieval units at the cost of variable chunk sizes. Having both makes downstream benchmarking meaningful.

**Overlap on fixed chunks.** Without overlap, a sentence split across a boundary would be absent from both chunks at retrieval time. 64 characters (≈ half a sentence) is a reasonable default.

**Local embeddings.** `all-MiniLM-L6-v2` is 22M parameters, produces 384-dimensional vectors, and runs comfortably on CPU. It scores competitively on retrieval benchmarks for its size and costs nothing to run.

**Cosine similarity via dot product.** Embeddings are L2-normalised before storage, so a FAISS `IndexFlatIP` (dot product) is mathematically equivalent to cosine similarity — one less operation per query with no accuracy cost.

**Exact search (IndexFlatIP).** Approximate nearest-neighbour indexes (HNSW, IVF) trade a small accuracy loss for much faster search at large scale. For a portfolio-scale corpus, exact search is simpler to reason about and leaves the trade-off visible for later benchmarking.

**Grounded system prompt.** The model is instructed to answer only from the provided context and say "I don't know" otherwise. This is the single most important hallucination-reduction technique in RAG — without it, a capable model will confidently fill gaps from its training data rather than the documents you gave it.

**`max_context_chars` budget.** The prompt builder stops adding passages once the context section hits a character limit. This prevents accidentally exceeding the model's token limit on large retrievals, at the cost of dropping the lowest-ranked passages — the ones least likely to help anyway.

**Pipeline as a singleton.** The embedding model takes ~2 seconds to load. Loading it per-request would make the API unusable. A single `Pipeline` object is created at startup and injected into route functions via FastAPI's dependency injection system — which also makes it trivially replaceable with a mock in tests.

**Frontend served by FastAPI, not a separate process.** Serving `index.html` via `FileResponse` and mounting `frontend/` as a `StaticFiles` sub-app keeps the deployment a single process — one `uvicorn` command, one Railway service, no CORS configuration needed.

**No JS framework.** The frontend is vanilla HTML/CSS/JS. For a tool that lives inside a portfolio API, adding React or Vue would be engineering for its own sake. Vanilla JS is easier to read, has no build step, and works fine for a two-section UI.

**XSS protection on chunk text.** Chunk content comes from user-uploaded PDFs, which can contain arbitrary characters. Before inserting chunk text into `innerHTML`, `escapeHtml()` replaces `&`, `<`, `>`, and `"` with their HTML entities. This prevents a malicious PDF from injecting `<script>` tags into the page.

**HTTP 503 before ingest, not 500.** A 503 (Service Unavailable) clearly communicates "the service is up but not ready" — distinct from a 500 (Internal Server Error) which implies something broke. A client can handle 503 by showing a "please upload a document first" message.

---

## License

MIT

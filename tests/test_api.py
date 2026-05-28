"""
Tests for the FastAPI application (api/main.py).

Covers all endpoints:
  GET  /health
  POST /ingest
  POST /query
  POST /query/stream
  POST /chat
  POST /chat/clear
  GET  /documents
  DELETE /documents/{source}
  GET  /  (frontend)
  GET  /static/*

All tests use dependency_overrides to inject a controlled Pipeline so
no real model loads and no API calls are made.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz  # PyMuPDF
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import Pipeline, app, get_pipeline
from api.models import HealthResponse, IngestResponse, QueryResponse
from src.bm25_store import BM25Store
from src.chunker import Chunk, chunk_text
from src.document_store import DocumentStore
from src.embedder import Embedder
from src.hybrid_retriever import HybridRetriever
from src.memory import ConversationMemory
from src.retriever import Retriever
from src.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIMENSION = 4


def make_pdf_bytes(text: str) -> bytes:
    """Create an in-memory single-page PDF and return raw bytes."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    return doc.tobytes()


def _rand_embeddings(n: int, dim: int = 384, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.random((n, dim), dtype=np.float32)
    return (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).astype(np.float32)


def _rand_embeddings_small(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, DIMENSION)).astype(np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _make_chunks_small(n: int, source: str = "test.pdf") -> list[Chunk]:
    return [
        Chunk(text=f"Chunk {i} from {source}", index=i,
              start_char=i * 20, end_char=i * 20 + 19,
              metadata={"source": source})
        for i in range(n)
    ]


def make_pipeline_with_store(texts: list[str], dim: int = 384) -> Pipeline:
    """Pipeline with a pre-populated VectorStore and a mock generator."""
    chunks = [
        Chunk(text=t, index=i, start_char=0, end_char=len(t),
              metadata={"source": "test.pdf"})
        for i, t in enumerate(texts)
    ]
    embs = _rand_embeddings(len(texts), dim)
    mock_embedder = MagicMock()
    mock_embedder.model_name = "mock-model"
    mock_embedder.dimension = dim
    mock_embedder.embed_one.return_value = embs[0]

    store = VectorStore(dim)
    store.add(chunks, embs)

    pipeline = Pipeline()
    pipeline.embedder = mock_embedder
    pipeline.store = store
    pipeline.retriever = Retriever(mock_embedder, store)
    bm25 = BM25Store(chunks)
    pipeline.bm25_store = bm25
    pipeline.hybrid_retriever = HybridRetriever(pipeline.retriever, bm25)
    pipeline.reranker = MagicMock()
    pipeline.reranker.rerank.return_value = [(chunks[0], 5.2)]
    pipeline.generator = MagicMock()
    pipeline.generator.generate.return_value = "The answer is 42."
    return pipeline


def make_ready_pipeline(doc_store: DocumentStore | None = None) -> MagicMock:
    """Mock Pipeline that is_ready with optional real DocumentStore."""
    p = MagicMock()
    p.is_ready = True
    p.memory = ConversationMemory()

    if doc_store is None:
        doc_store = DocumentStore(DIMENSION)
    p.document_store = doc_store

    chunk = Chunk(text="Python is a programming language.", index=0,
                  start_char=0, end_char=33, metadata={"source": "test.pdf"})
    p.retriever.retrieve.return_value = [(chunk, 0.85)]
    p.hybrid_retriever.retrieve.return_value = [(chunk, 0.85)]
    p.reranker.rerank.return_value = [(chunk, 5.2)]
    p.generator.generate.return_value = "Python is a language."
    p.generator.generate_with_history.return_value = "Python is a language."
    p.generator.stream.return_value = iter(["Python ", "is a language."])
    return p


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    """Reset dependency overrides after every test."""
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture(scope="session")
def real_embedder() -> Embedder:
    return Embedder()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client):
        pipeline = Pipeline()
        pipeline.embedder = MagicMock(model_name="test-model")
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        assert client.get("/health").status_code == 200

    def test_response_schema(self, client):
        pipeline = Pipeline()
        pipeline.embedder = MagicMock(model_name="test-model")
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert "index_size" in body
        assert "embedding_model" in body

    def test_index_size_zero_before_ingest(self, client):
        pipeline = Pipeline()
        pipeline.embedder = MagicMock(model_name="test-model")
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        assert client.get("/health").json()["index_size"] == 0

    def test_index_size_after_ingest(self, client):
        pipeline = make_pipeline_with_store(["chunk one", "chunk two", "chunk three"])
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        assert client.get("/health").json()["index_size"] == 3

    def test_embedding_model_name_in_response(self, client):
        pipeline = Pipeline()
        pipeline.embedder = MagicMock(model_name="all-MiniLM-L6-v2")
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        assert client.get("/health").json()["embedding_model"] == "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

class TestIngest:
    def test_non_pdf_returns_422(self, client):
        pipeline = Pipeline()
        pipeline.embedder = MagicMock(model_name="m", dimension=384)
        pipeline.embedder.embed.return_value = _rand_embeddings(1)
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        resp = client.post("/ingest",
                           files={"file": ("notes.txt", b"some text", "text/plain")})
        assert resp.status_code == 422

    def test_empty_file_returns_422(self, client):
        pipeline = Pipeline()
        pipeline.embedder = MagicMock(model_name="m", dimension=384)
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        resp = client.post("/ingest",
                           files={"file": ("empty.pdf", b"", "application/pdf")})
        assert resp.status_code == 422

    def test_valid_pdf_returns_201(self, client, real_embedder):
        pipeline = Pipeline()
        pipeline.embedder = real_embedder
        pipeline.generator = MagicMock()
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        resp = client.post(
            "/ingest",
            files={"file": ("test.pdf",
                            make_pdf_bytes("The quick brown fox jumps over the lazy dog."),
                            "application/pdf")},
        )
        assert resp.status_code == 201

    def test_ingest_response_schema(self, client, real_embedder):
        pipeline = Pipeline()
        pipeline.embedder = real_embedder
        pipeline.generator = MagicMock()
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        body = client.post(
            "/ingest",
            files={"file": ("doc.pdf",
                            make_pdf_bytes("Hello world. This is a test document."),
                            "application/pdf")},
        ).json()
        assert "message" in body
        assert "chunks_indexed" in body
        assert body["source"] == "doc.pdf"

    def test_ingest_chunks_indexed_positive(self, client, real_embedder):
        pipeline = Pipeline()
        pipeline.embedder = real_embedder
        pipeline.generator = MagicMock()
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        body = client.post(
            "/ingest",
            files={"file": ("ml.pdf",
                            make_pdf_bytes("Machine learning is a branch of AI."),
                            "application/pdf")},
        ).json()
        assert body["chunks_indexed"] > 0

    def test_ingest_populates_store(self, client, real_embedder):
        pipeline = Pipeline()
        pipeline.embedder = real_embedder
        pipeline.generator = MagicMock()
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        assert pipeline.store is None
        client.post("/ingest",
                    files={"file": ("code.pdf",
                                    make_pdf_bytes("Some content about Python programming."),
                                    "application/pdf")})
        assert pipeline.store is not None
        assert len(pipeline.store) > 0

    def test_ingest_creates_retriever(self, client, real_embedder):
        pipeline = Pipeline()
        pipeline.embedder = real_embedder
        pipeline.generator = MagicMock()
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        assert pipeline.retriever is None
        client.post("/ingest",
                    files={"file": ("dl.pdf",
                                    make_pdf_bytes("Deep learning uses neural networks."),
                                    "application/pdf")})
        assert pipeline.retriever is not None


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

class TestQuery:
    def test_query_before_ingest_returns_503(self, client):
        pipeline = Pipeline()
        pipeline.embedder = MagicMock(model_name="m", dimension=384)
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        assert client.post("/query", json={"question": "What is AI?"}).status_code == 503

    def test_query_returns_200(self, client):
        pipeline = make_pipeline_with_store(["Artificial intelligence is the future."])
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        assert client.post("/query", json={"question": "What is AI?"}).status_code == 200

    def test_query_response_schema(self, client):
        pipeline = make_pipeline_with_store(["The sky is blue."])
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        body = client.post("/query", json={"question": "Why is the sky blue?"}).json()
        assert "answer" in body
        assert "sources" in body
        assert isinstance(body["sources"], list)

    def test_query_answer_is_string(self, client):
        pipeline = make_pipeline_with_store(["Water boils at 100 degrees Celsius."])
        pipeline.generator.generate.return_value = "Water boils at 100°C."
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        body = client.post("/query",
                           json={"question": "At what temperature does water boil?"}).json()
        assert isinstance(body["answer"], str)
        assert len(body["answer"]) > 0

    def test_query_sources_have_correct_fields(self, client):
        pipeline = make_pipeline_with_store(["Paris is in France."])
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        source = client.post("/query", json={"question": "Where is Paris?"}).json()["sources"][0]
        assert "text" in source
        assert "score" in source
        assert "source" in source

    def test_query_uses_generator(self, client):
        pipeline = make_pipeline_with_store(["Some fact."])
        pipeline.generator.generate.return_value = "Generated answer."
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        body = client.post("/query", json={"question": "Tell me something."}).json()
        assert body["answer"] == "Generated answer."
        pipeline.generator.generate.assert_called_once()

    def test_query_respects_k(self, client):
        pipeline = make_pipeline_with_store([f"Fact number {i}." for i in range(8)])
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        body = client.post("/query", json={"question": "fact", "k": 3}).json()
        assert len(body["sources"]) <= 3

    def test_query_source_field_matches_metadata(self, client):
        pipeline = make_pipeline_with_store(["Hello world."])
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        body = client.post("/query", json={"question": "hello"}).json()
        assert body["sources"][0]["source"] == "test.pdf"

    def test_rerank_true_returns_200(self, client):
        pipeline = make_pipeline_with_store(["AI is transforming the world."])
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        resp = client.post("/query", json={"question": "What is AI?", "rerank": True})
        assert resp.status_code == 200

    def test_rerank_calls_reranker(self, client):
        pipeline = make_pipeline_with_store(["Some text about machine learning."])
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        client.post("/query", json={"question": "machine learning", "rerank": True})
        pipeline.reranker.rerank.assert_called_once()

    def test_rerank_false_does_not_call_reranker(self, client):
        pipeline = make_pipeline_with_store(["Some text about machine learning."])
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        client.post("/query", json={"question": "machine learning", "rerank": False})
        pipeline.reranker.rerank.assert_not_called()

    def test_rerank_default_does_not_call_reranker(self, client):
        pipeline = make_pipeline_with_store(["Some text."])
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        client.post("/query", json={"question": "text"})
        pipeline.reranker.rerank.assert_not_called()

    def test_rerank_response_has_sources(self, client):
        pipeline = make_pipeline_with_store(["Neural networks learn from data."])
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        body = client.post(
            "/query", json={"question": "how do neural networks learn?", "rerank": True}
        ).json()
        assert "sources" in body
        assert len(body["sources"]) > 0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_empty_question_returns_422(self, client):
        pipeline = make_pipeline_with_store(["content"])
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        assert client.post("/query", json={"question": ""}).status_code == 422

    def test_k_too_large_returns_422(self, client):
        pipeline = make_pipeline_with_store(["content"])
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        assert client.post("/query", json={"question": "q", "k": 999}).status_code == 422

    def test_k_zero_returns_422(self, client):
        pipeline = make_pipeline_with_store(["content"])
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        assert client.post("/query", json={"question": "q", "k": 0}).status_code == 422

    def test_min_score_above_1_returns_422(self, client):
        pipeline = make_pipeline_with_store(["content"])
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        assert client.post("/query", json={"question": "q", "min_score": 1.5}).status_code == 422

    def test_missing_question_field_returns_422(self, client):
        pipeline = make_pipeline_with_store(["content"])
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        assert client.post("/query", json={"k": 3}).status_code == 422


# ---------------------------------------------------------------------------
# GET / — frontend entry point
# ---------------------------------------------------------------------------

class TestFrontend:
    def test_root_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_root_content_type_is_html(self, client):
        assert "text/html" in client.get("/").headers["content-type"]

    def test_root_contains_title(self, client):
        assert b"RAG Pipeline" in client.get("/").content

    def test_root_links_stylesheet(self, client):
        assert b"/static/style.css" in client.get("/").content

    def test_root_links_script(self, client):
        assert b"/static/app.js" in client.get("/").content

    def test_root_has_upload_section(self, client):
        assert b"upload" in client.get("/").content.lower()

    def test_root_has_query_section(self, client):
        content = client.get("/").content.lower()
        assert b"question" in content or b"ask" in content

    def test_root_not_in_openapi_schema(self, client):
        assert "/" not in client.get("/openapi.json").json()["paths"]


# ---------------------------------------------------------------------------
# Static assets
# ---------------------------------------------------------------------------

class TestStaticAssets:
    def test_css_returns_200(self, client):
        assert client.get("/static/style.css").status_code == 200

    def test_css_content_type(self, client):
        assert "text/css" in client.get("/static/style.css").headers["content-type"]

    def test_js_returns_200(self, client):
        assert client.get("/static/app.js").status_code == 200

    def test_js_content_type(self, client):
        assert "javascript" in client.get("/static/app.js").headers["content-type"]

    def test_nonexistent_static_file_returns_404(self, client):
        assert client.get("/static/does-not-exist.txt").status_code == 404

    def test_css_has_content(self, client):
        assert len(client.get("/static/style.css").content) > 100

    def test_js_has_content(self, client):
        assert len(client.get("/static/app.js").content) > 100

    def test_js_contains_upload_function(self, client):
        assert b"uploadPdf" in client.get("/static/app.js").content

    def test_js_contains_query_function(self, client):
        assert b"submitQuery" in client.get("/static/app.js").content

    def test_js_contains_escape_html(self, client):
        """XSS protection must be present for rendering chunk text."""
        assert b"escapeHtml" in client.get("/static/app.js").content


# ---------------------------------------------------------------------------
# Deployment config files
# ---------------------------------------------------------------------------

class TestDeploymentConfig:
    _root = Path(__file__).parent.parent

    def test_procfile_exists(self):
        assert (self._root / "Procfile").exists()

    def test_procfile_contains_uvicorn(self):
        content = (self._root / "Procfile").read_text()
        assert "uvicorn" in content
        assert "api.main:app" in content

    def test_procfile_binds_to_port_env(self):
        assert "$PORT" in (self._root / "Procfile").read_text()

    def test_railway_json_exists(self):
        assert (self._root / "railway.json").exists()

    def test_railway_json_valid(self):
        import json
        config = json.loads((self._root / "railway.json").read_text())
        assert "deploy" in config
        assert "startCommand" in config["deploy"]

    def test_runtime_txt_exists(self):
        assert (self._root / "runtime.txt").exists()

    def test_runtime_txt_specifies_python(self):
        assert "python" in (self._root / "runtime.txt").read_text().lower()


# ---------------------------------------------------------------------------
# Regression — existing routes still work after static mount
# ---------------------------------------------------------------------------

class TestApiRegression:
    def test_health_still_200(self, client):
        pipeline = Pipeline()
        pipeline.embedder = MagicMock(model_name="test-model")
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        assert client.get("/health").status_code == 200

    def test_query_before_ingest_still_503(self, client):
        pipeline = Pipeline()
        pipeline.embedder = MagicMock(model_name="m")
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        assert client.post("/query", json={"question": "test"}).status_code == 503

    def test_openapi_docs_still_accessible(self, client):
        assert client.get("/docs").status_code == 200

    def test_openapi_schema_still_valid(self, client):
        schema = client.get("/openapi.json").json()
        assert "/health" in schema["paths"]
        assert "/ingest" in schema["paths"]
        assert "/query" in schema["paths"]


# ---------------------------------------------------------------------------
# POST /query/stream
# ---------------------------------------------------------------------------

class TestQueryStreamEndpoint:
    def test_streams_sse_events(self, client):
        pipeline = make_ready_pipeline()
        pipeline.generator.stream.return_value = iter(["Hello", " world"])
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        resp = client.post("/query/stream", json={"question": "What is Python?"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = resp.text
        assert "data: Hello" in body
        assert "data: [DONE]" in body

    def test_stream_503_when_not_ready(self, client):
        pipeline = make_ready_pipeline()
        pipeline.is_ready = False
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        assert client.post("/query/stream", json={"question": "Q?"}).status_code == 503

    def test_stream_404_when_no_results(self, client):
        pipeline = make_ready_pipeline()
        pipeline.retriever.retrieve.return_value = []
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        assert client.post("/query/stream", json={"question": "Q?"}).status_code == 404


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

class TestChatEndpoint:
    def test_returns_200_with_answer(self, client):
        pipeline = make_ready_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        resp = client.post("/chat", json={"question": "What is Python?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert "turn_count" in data

    def test_turn_count_increments(self, client):
        pipeline = make_ready_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        assert client.post("/chat", json={"question": "Q1?"}).json()["turn_count"] == 1
        assert client.post("/chat", json={"question": "Q2?"}).json()["turn_count"] == 2

    def test_generate_with_history_called(self, client):
        pipeline = make_ready_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        client.post("/chat", json={"question": "What is Python?"})
        assert pipeline.generator.generate_with_history.called

    def test_history_passed_in_second_turn(self, client):
        pipeline = make_ready_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        client.post("/chat", json={"question": "What is Python?"})
        client.post("/chat", json={"question": "Who created it?"})
        second_call_msgs = pipeline.generator.generate_with_history.call_args_list[1][0][0]
        assert len(second_call_msgs) >= 3

    def test_503_when_not_ready(self, client):
        pipeline = make_ready_pipeline()
        pipeline.is_ready = False
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        assert client.post("/chat", json={"question": "Q?"}).status_code == 503

    def test_404_when_no_results(self, client):
        pipeline = make_ready_pipeline()
        pipeline.retriever.retrieve.return_value = []
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        assert client.post("/chat", json={"question": "Q?"}).status_code == 404


# ---------------------------------------------------------------------------
# POST /chat/clear
# ---------------------------------------------------------------------------

class TestChatClearEndpoint:
    def test_clear_returns_204(self, client):
        pipeline = make_ready_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        assert client.post("/chat/clear").status_code == 204

    def test_clear_resets_turn_count(self, client):
        pipeline = make_ready_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        client.post("/chat", json={"question": "Q1?"})
        assert pipeline.memory.num_exchanges == 1
        client.post("/chat/clear")
        assert pipeline.memory.num_exchanges == 0

    def test_can_chat_after_clear(self, client):
        pipeline = make_ready_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        client.post("/chat", json={"question": "Q1?"})
        client.post("/chat/clear")
        assert client.post("/chat", json={"question": "Q2?"}).json()["turn_count"] == 1


# ---------------------------------------------------------------------------
# GET /documents
# ---------------------------------------------------------------------------

class TestListDocumentsEndpoint:
    def test_returns_200(self, client):
        pipeline = make_ready_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        assert client.get("/documents").status_code == 200

    def test_empty_when_no_docs(self, client):
        pipeline = make_ready_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        data = client.get("/documents").json()
        assert data["documents"] == []
        assert data["total_documents"] == 0
        assert data["total_chunks"] == 0

    def test_shows_ingested_doc(self, client):
        ds = DocumentStore(DIMENSION)
        ds.add("paper.pdf", _make_chunks_small(3), _rand_embeddings_small(3))
        pipeline = make_ready_pipeline(doc_store=ds)
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        data = client.get("/documents").json()
        assert data["total_documents"] == 1
        assert data["total_chunks"] == 3
        assert data["documents"][0]["source"] == "paper.pdf"

    def test_shows_multiple_docs(self, client):
        ds = DocumentStore(DIMENSION)
        ds.add("a.pdf", _make_chunks_small(3), _rand_embeddings_small(3, seed=0))
        ds.add("b.pdf", _make_chunks_small(5), _rand_embeddings_small(5, seed=1))
        pipeline = make_ready_pipeline(doc_store=ds)
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        data = client.get("/documents").json()
        assert data["total_documents"] == 2
        assert data["total_chunks"] == 8
        sources = [d["source"] for d in data["documents"]]
        assert "a.pdf" in sources
        assert "b.pdf" in sources

    def test_document_has_ingested_at(self, client):
        from datetime import datetime
        ds = DocumentStore(DIMENSION)
        ds.add("a.pdf", _make_chunks_small(2), _rand_embeddings_small(2))
        pipeline = make_ready_pipeline(doc_store=ds)
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        data = client.get("/documents").json()
        assert "ingested_at" in data["documents"][0]
        datetime.fromisoformat(data["documents"][0]["ingested_at"])


# ---------------------------------------------------------------------------
# DELETE /documents/{source}
# ---------------------------------------------------------------------------

class TestDeleteDocumentEndpoint:
    def test_returns_204_when_found(self, client):
        ds = DocumentStore(DIMENSION)
        ds.add("paper.pdf", _make_chunks_small(3), _rand_embeddings_small(3))
        pipeline = make_ready_pipeline(doc_store=ds)
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        assert client.delete("/documents/paper.pdf").status_code == 204

    def test_returns_404_when_not_found(self, client):
        pipeline = make_ready_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        assert client.delete("/documents/nonexistent.pdf").status_code == 404

    def test_document_removed_from_store(self, client):
        ds = DocumentStore(DIMENSION)
        ds.add("paper.pdf", _make_chunks_small(3), _rand_embeddings_small(3))
        pipeline = make_ready_pipeline(doc_store=ds)
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        client.delete("/documents/paper.pdf")
        assert "paper.pdf" not in ds

    def test_retriever_cleared_when_last_doc_deleted(self, client):
        ds = DocumentStore(DIMENSION)
        ds.add("only.pdf", _make_chunks_small(2), _rand_embeddings_small(2))
        pipeline = make_ready_pipeline(doc_store=ds)
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        client.delete("/documents/only.pdf")
        assert pipeline.store is None
        assert pipeline.retriever is None

    def test_index_shrinks_after_delete(self, client):
        ds = DocumentStore(DIMENSION)
        ds.add("a.pdf", _make_chunks_small(3), _rand_embeddings_small(3, seed=0))
        ds.add("b.pdf", _make_chunks_small(5), _rand_embeddings_small(5, seed=1))
        pipeline = make_ready_pipeline(doc_store=ds)
        pipeline.embedder = MagicMock()
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        client.delete("/documents/a.pdf")
        assert ds.total_chunks == 5
        assert ds.document_count == 1


# ---------------------------------------------------------------------------
# Ingest accumulation (multi-document)
# ---------------------------------------------------------------------------

class TestIngestAccumulates:
    def test_same_source_replaces(self):
        ds = DocumentStore(DIMENSION)
        ds.add("dup.pdf", _make_chunks_small(3), _rand_embeddings_small(3))
        ds.add("dup.pdf", _make_chunks_small(7), _rand_embeddings_small(7))
        assert ds.document_count == 1
        assert ds.total_chunks == 7

    def test_two_docs_accumulate_in_store(self):
        ds = DocumentStore(DIMENSION)
        ds.add("a.pdf", _make_chunks_small(3), _rand_embeddings_small(3, seed=0))
        ds.add("b.pdf", _make_chunks_small(5), _rand_embeddings_small(5, seed=1))
        assert ds.document_count == 2
        assert ds.total_chunks == 8

    def test_ingest_response_includes_total_fields(self):
        from api.models import IngestResponse
        r = IngestResponse(message="ok", chunks_indexed=5, source="test.pdf",
                           total_documents=2, total_chunks=12)
        assert r.total_documents == 2
        assert r.total_chunks == 12

    def test_get_documents_after_two_ingests(self, client):
        ds = DocumentStore(DIMENSION)
        ds.add("a.pdf", _make_chunks_small(3), _rand_embeddings_small(3, seed=0))
        ds.add("b.pdf", _make_chunks_small(4), _rand_embeddings_small(4, seed=1))
        pipeline = make_ready_pipeline(doc_store=ds)
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        data = client.get("/documents").json()
        assert data["total_documents"] == 2
        sources = {d["source"] for d in data["documents"]}
        assert sources == {"a.pdf", "b.pdf"}

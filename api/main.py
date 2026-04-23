"""
FastAPI application — exposes the RAG pipeline over HTTP.

Endpoints:
  GET  /health          — liveness check and index stats
  POST /ingest          — upload a PDF and add it to the index
  GET  /documents       — list all ingested documents
  DELETE /documents/{source} — remove a document and rebuild the index
  POST /query           — ask a question, get an answer with source passages
  POST /query/stream    — same as /query but streams the answer as SSE
  POST /chat            — ask a question with full conversation memory
  POST /chat/clear      — reset conversation history
"""

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api.models import (
    ChatRequest,
    ChatResponse,
    DocumentInfo,
    DocumentListResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from src.chunker import chunk_by_sentence, chunk_text  # both available for easy switching
from src.document_store import DocumentStore
from src.embedder import Embedder
from src.generator import Generator
from src.ingestion import clean_text, load_pdf
from src.memory import ConversationMemory
from src.prompt import build_prompt, get_system_prompt
from src.retriever import Retriever
from src.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

class Pipeline:
    """
    Holds all stateful components shared across requests.

    Initialised once at startup (embedder + generator + document_store),
    then updated each time a document is ingested or removed.

    The document_store is the source of truth for which documents are
    indexed. After every add or remove, store and retriever are rebuilt
    from it so they always reflect the current document set.
    """

    def __init__(self) -> None:
        self.embedder: Embedder | None = None
        self.store: VectorStore | None = None
        self.retriever: Retriever | None = None
        self.generator: Generator | None = None
        self.memory: ConversationMemory = ConversationMemory()
        self.document_store: DocumentStore | None = None

    def startup(self) -> None:
        """Load the embedding model and LLM client. Called once at app startup."""
        self.embedder = Embedder()
        self.generator = Generator()
        self.document_store = DocumentStore(self.embedder.dimension)

    def _rebuild_retriever(self) -> None:
        """Rebuild the dense retriever from the current document store."""
        self.store = self.document_store.build_vector_store()
        self.retriever = Retriever(self.embedder, self.store)

    @property
    def index_size(self) -> int:
        if self.document_store is not None:
            return self.document_store.total_chunks
        return len(self.store) if self.store else 0

    @property
    def is_ready(self) -> bool:
        """True once at least one document has been ingested."""
        if self.document_store is not None:
            return self.document_store.document_count > 0
        # Fallback for test helpers that set store/retriever directly.
        return self.store is not None and self.retriever is not None


_pipeline = Pipeline()


def get_pipeline() -> Pipeline:
    """FastAPI dependency — returns the shared pipeline instance."""
    return _pipeline


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavyweight components once at startup, release at shutdown."""
    _pipeline.startup()
    yield


FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = FastAPI(
    title="RAG Pipeline",
    description="Retrieval-Augmented Generation API — built from scratch, no LangChain.",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    """Serve the single-page frontend with no-cache headers."""
    return FileResponse(
        FRONTEND_DIR / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/health", response_model=HealthResponse)
def health(pipeline: Pipeline = Depends(get_pipeline)) -> HealthResponse:
    """Return service status, current index size, and embedding model name."""
    return HealthResponse(
        status="ok",
        index_size=pipeline.index_size,
        embedding_model=pipeline.embedder.model_name if pipeline.embedder else "not loaded",
    )


@app.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest(
    file: UploadFile = File(...),
    pipeline: Pipeline = Depends(get_pipeline),
) -> IngestResponse:
    """
    Upload a PDF and add it to the index.

    The uploaded file is chunked, embedded, and added to the shared FAISS
    index alongside any previously ingested documents. If a document with
    the same filename already exists it is replaced (re-ingestion is safe).

    Returns the number of chunks indexed for this file and the total
    document and chunk counts across the whole index.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Only PDF files are accepted.",
        )

    # PyMuPDF needs a real file path, so write the upload to a temp file.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Uploaded file is empty.",
            )
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        raw_text = load_pdf(tmp_path)
        clean = clean_text(raw_text)
        # Sentence-boundary chunking — cleaner passages, better embedding quality.
        # To switch back to fixed character windows, comment the line below and
        # uncomment the one after it.
        chunks = chunk_by_sentence(clean, max_chunk_size=512, overlap_sentences=1, source=file.filename)
        # chunks = chunk_text(clean, chunk_size=512, overlap=64, source=file.filename)
        embeddings = pipeline.embedder.embed([c.text for c in chunks])

        if pipeline.document_store is None:
            pipeline.document_store = DocumentStore(pipeline.embedder.dimension)
        pipeline.document_store.add(file.filename, chunks, embeddings)
        pipeline._rebuild_retriever()
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(
        message=f"Successfully ingested '{file.filename}'.",
        chunks_indexed=len(chunks),
        source=file.filename,
        total_documents=pipeline.document_store.document_count,
        total_chunks=pipeline.document_store.total_chunks,
    )


@app.get("/documents", response_model=DocumentListResponse)
def list_documents(pipeline: Pipeline = Depends(get_pipeline)) -> DocumentListResponse:
    """
    List all documents currently in the index.

    Returns metadata (source name, chunk count, ingestion timestamp) for
    every document that has been ingested via POST /ingest. Does not
    require any documents to be present — returns an empty list if the
    index is empty.
    """
    if pipeline.document_store is None:
        return DocumentListResponse(documents=[], total_documents=0, total_chunks=0)

    records = pipeline.document_store.list_documents()
    return DocumentListResponse(
        documents=[
            DocumentInfo(
                source=r.source,
                chunk_count=r.chunk_count,
                ingested_at=r.ingested_at,
            )
            for r in records
        ],
        total_documents=pipeline.document_store.document_count,
        total_chunks=pipeline.document_store.total_chunks,
    )


@app.delete("/documents/{source}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    source: str,
    pipeline: Pipeline = Depends(get_pipeline),
) -> None:
    """
    Remove a document from the index by source name.

    The source name is the filename used when the document was ingested
    (e.g. "paper.pdf"). After removal the index is rebuilt from the
    remaining documents. If the last document is removed, the pipeline
    transitions back to the not-ready state.

    Returns 204 No Content on success, 404 if the source is not found.
    """
    if pipeline.document_store is None or not pipeline.document_store.remove(source):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{source}' not found.",
        )

    if pipeline.document_store.document_count > 0:
        pipeline._rebuild_retriever()
    else:
        pipeline.store = None
        pipeline.retriever = None


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    pipeline: Pipeline = Depends(get_pipeline),
) -> QueryResponse:
    """
    Ask a question against the ingested documents.

    Retrieves the most relevant passages, constructs a grounded prompt,
    and returns Claude's answer alongside the source passages used.
    """
    if not pipeline.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No documents ingested yet. POST a PDF to /ingest first.",
        )

    results = pipeline.retriever.retrieve(
        request.question,
        k=request.k,
        min_score=request.min_score,
    )

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant passages found. Try lowering min_score.",
        )

    prompt = build_prompt(request.question, results)
    answer = pipeline.generator.generate(prompt, system=get_system_prompt())

    sources = [
        SourceChunk(
            text=chunk.text,
            score=round(score, 4),
            source=chunk.metadata.get("source", ""),
        )
        for chunk, score in results
    ]

    return QueryResponse(answer=answer, sources=sources)


@app.post("/query/stream")
def query_stream(
    request: QueryRequest,
    pipeline: Pipeline = Depends(get_pipeline),
) -> StreamingResponse:
    """
    Ask a question and receive the answer as a Server-Sent Events stream.

    Tokens arrive incrementally so the client can begin rendering before
    the full response is complete. Each event is formatted as:

        data: <text chunk>\\n\\n

    A final sentinel event signals completion:

        data: [DONE]\\n\\n

    Single-turn only — does not use conversation memory. For multi-turn
    dialogue with streaming, combine /chat memory management with this
    endpoint's streaming approach in a future iteration.
    """
    if not pipeline.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No documents ingested yet. POST a PDF to /ingest first.",
        )

    results = pipeline.retriever.retrieve(
        request.question,
        k=request.k,
        min_score=request.min_score,
    )

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant passages found. Try lowering min_score.",
        )

    prompt = build_prompt(request.question, results)
    system = get_system_prompt()

    def event_stream():
        for chunk in pipeline.generator.stream(prompt, system=system):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    pipeline: Pipeline = Depends(get_pipeline),
) -> ChatResponse:
    """
    Ask a question with full conversation memory.

    Retrieves fresh context for the current question, then sends the full
    conversation history alongside the new RAG prompt to the model. The
    model can refer to previous answers when answering follow-up questions.

    The conversation history persists across calls until POST /chat/clear
    resets it. History stores bare questions and answers (not raw RAG
    prompts) to keep the context window manageable.
    """
    if not pipeline.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No documents ingested yet. POST a PDF to /ingest first.",
        )

    results = pipeline.retriever.retrieve(
        request.question,
        k=request.k,
        min_score=request.min_score,
    )

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant passages found. Try lowering min_score.",
        )

    # Build the RAG prompt for this turn and append it to the history
    rag_prompt = build_prompt(request.question, results)
    history = pipeline.memory.get_messages()
    messages = history + [{"role": "user", "content": rag_prompt}]

    answer = pipeline.generator.generate_with_history(
        messages, system=get_system_prompt()
    )

    # Store the bare question (not the full RAG prompt) to keep history compact
    pipeline.memory.add_exchange(request.question, answer)

    sources = [
        SourceChunk(
            text=chunk.text,
            score=round(score, 4),
            source=chunk.metadata.get("source", ""),
        )
        for chunk, score in results
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        turn_count=pipeline.memory.num_exchanges,
    )


@app.post("/chat/clear", status_code=status.HTTP_204_NO_CONTENT)
def chat_clear(pipeline: Pipeline = Depends(get_pipeline)) -> None:
    """Reset the conversation memory, starting a fresh session."""
    pipeline.memory.clear()


@app.post("/reset", status_code=status.HTTP_204_NO_CONTENT)
def reset(pipeline: Pipeline = Depends(get_pipeline)) -> None:
    """
    Clear all ingested documents and reset the pipeline to its initial state.

    Wipes the document store, FAISS index, and conversation memory so the
    user can start fresh without restarting the server.
    """
    pipeline.document_store = DocumentStore(pipeline.embedder.dimension)
    pipeline.store = None
    pipeline.retriever = None
    pipeline.memory.clear()

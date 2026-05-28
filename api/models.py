"""
Pydantic request and response schemas for the RAG API.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Body for POST /query."""

    question: str = Field(..., min_length=1, description="The question to answer.")
    k: int = Field(default=5, ge=1, le=20, description="Number of passages to retrieve.")
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold — passages scoring below this are excluded. The API uses hybrid retrieval (RRF fusion), so live scores are small values (~0.005–0.03), not cosine similarities in [0, 1]. Leave at 0.0 unless you understand RRF score ranges.",
    )


class SourceChunk(BaseModel):
    """A single retrieved passage included in a query response."""

    text: str
    score: float
    source: str


class QueryResponse(BaseModel):
    """Body for a successful POST /query response."""

    answer: str
    sources: list[SourceChunk]


class IngestResponse(BaseModel):
    """Body for a successful POST /ingest response."""

    message: str
    chunks_indexed: int
    source: str
    total_documents: int = Field(
        description="Total number of documents currently in the index."
    )
    total_chunks: int = Field(
        description="Total number of chunks across all indexed documents."
    )


class HealthResponse(BaseModel):
    """Body for GET /health."""

    status: str
    index_size: int
    embedding_model: str


class ChatRequest(BaseModel):
    """Body for POST /chat — same parameters as /query, adds conversation memory."""

    question: str = Field(..., min_length=1, description="The question to answer.")
    k: int = Field(default=5, ge=1, le=20, description="Number of passages to retrieve.")
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold — passages scoring below this are excluded. The API uses hybrid retrieval (RRF fusion), so live scores are small values (~0.005–0.03), not cosine similarities in [0, 1]. Leave at 0.0 unless you understand RRF score ranges.",
    )


class ChatResponse(BaseModel):
    """Body for a successful POST /chat response."""

    answer: str
    sources: list[SourceChunk]
    turn_count: int  # number of complete exchanges in memory after this turn


class DocumentInfo(BaseModel):
    """Metadata for a single ingested document, as returned by GET /documents."""

    source: str
    chunk_count: int
    ingested_at: datetime


class DocumentListResponse(BaseModel):
    """Body for GET /documents."""

    documents: list[DocumentInfo]
    total_documents: int
    total_chunks: int

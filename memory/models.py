from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class SemanticEntry(BaseModel):
    """Structured knowledge — replaces raw file dumps in ChromaDB"""
    type: str = Field(
        default="file_knowledge",
        description="Type of knowledge: file_knowledge | concept | fact"
    )
    path: str = Field(
        default="",
        description="File path if applicable"
    )
    summary: str = Field(
        default="",
        description="AI-generated 2-line summary"
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Extracted keywords for filtering"
    )
    content: str = Field(
        default="",
        description="Actual content"
    )
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Importance score, reinforced on repeated access"
    )


class EpisodicEntry(BaseModel):
    """Event log — what happened and when (stored in SQLite)"""
    event: str = Field(
        ...,
        description="Event type: created_file | modified_file | deleted_file | renamed_file | searched | answered | reflected"
    )
    path: str = Field(
        default="",
        description="Affected file path if any"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO timestamp"
    )
    summary: str = Field(
        default="",
        description="Human-readable summary of what happened"
    )
    user_message: str = Field(
        default="",
        description="The original user request that triggered this event"
    )


class CachedSummary(BaseModel):
    """Cached file summary to avoid recomputation"""
    path: str
    content_hash: str  # hash of the content that was summarized
    summary: str
    keywords: list[str]
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )

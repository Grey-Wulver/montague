# app/models/discovery_models.py
"""
Discovery Data Models
Consistent data structures for command discovery results across all tiers
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class DiscoveryResult:
    """
    Structured result from command discovery operations.

    Used by HybridCommandDiscovery to return consistent data
    regardless of discovery tier (Redis/PostgreSQL/LLM).
    """

    command: str
    confidence: float
    source: str  # "redis_cache", "postgres_db", "llm_discovery", "fallback"
    reasoning: Optional[str] = None
    discovery_time: float = 0.0
    cached_at: Optional[datetime] = None
    usage_count: Optional[int] = None
    discovery_metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate discovery result data"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")

        if self.source not in [
            "redis_cache",
            "postgres_db",
            "llm_discovery",
            "fallback",
        ]:
            raise ValueError(f"Invalid source: {self.source}")

    @property
    def is_cached(self) -> bool:
        """Check if result came from cache (Redis or promoted PostgreSQL)"""
        return self.source in ["redis_cache", "postgres_db"]

    @property
    def is_fresh_discovery(self) -> bool:
        """Check if result is from fresh LLM discovery"""
        return self.source == "llm_discovery"

    @property
    def is_fallback(self) -> bool:
        """Check if result is from intelligent fallback"""
        return self.source == "fallback"


@dataclass
class NormalizationResult:
    """
    Result from LLM-powered intent normalization.

    Used to track normalization performance and quality.
    """

    original_intent: str
    normalized_intent: str
    normalization_time: float
    confidence: float = 1.0
    normalization_metadata: Optional[Dict[str, Any]] = None

    @property
    def was_changed(self) -> bool:
        """Check if normalization changed the original intent"""
        return self.original_intent.strip() != self.normalized_intent.strip()


@dataclass
class PlatformDiscoveryResult:
    """
    Discovery result for a specific platform.

    Used when discovering commands for multiple platforms simultaneously.
    """

    platform: str
    discovery_result: DiscoveryResult
    device_count: int = 1

    @property
    def total_discovery_time(self) -> float:
        """Total time including per-device overhead"""
        return self.discovery_result.discovery_time * self.device_count

# ~/net-chatbot/app/models/command_mappings.py
"""
Database Models for Three-Tier Discovery System
PostgreSQL + Redis optimized schemas for progressive performance
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.core.database import Base


class CommandMapping(Base):
    """
    Core table for command discovery mappings
    Optimized for three-tier lookup performance
    """

    __tablename__ = "command_mappings"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Core discovery data
    intent = Column(String(100), nullable=False, index=True)
    platform = Column(String(50), nullable=False, index=True)
    command = Column(Text, nullable=False)

    # Discovery metadata
    confidence = Column(Float, default=1.0, index=True)
    source = Column(
        String(20), default="community", index=True
    )  # community, llm_discovery, manual

    # Usage tracking for intelligent caching
    usage_count = Column(Integer, default=0, index=True)
    last_used = Column(DateTime, default=None, index=True)

    # Audit fields
    created_at = Column(DateTime, default=func.now(), index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Metadata for LLM discoveries
    discovery_metadata = Column(
        JSONB, nullable=True
    )  # Store LLM reasoning, alternatives, etc.

    # Performance optimization constraints
    __table_args__ = (
        UniqueConstraint("intent", "platform", name="uq_intent_platform"),
        Index("idx_intent_platform", "intent", "platform"),  # Primary lookup index
        Index("idx_usage_count_desc", "usage_count", postgresql_using="btree"),
        Index("idx_confidence_desc", "confidence", postgresql_using="btree"),
        Index("idx_last_used_desc", "last_used", postgresql_using="btree"),
        Index("idx_source", "source"),
    )

    def __repr__(self):
        return f"<CommandMapping(intent='{self.intent}', platform='{self.platform}', command='{self.command[:50]}...')>"

    def to_cache_dict(self) -> dict:
        """Convert to Redis cache format"""
        return {
            "command": self.command,
            "confidence": float(self.confidence),
            "source": self.source,
            "usage_count": self.usage_count,
            "cached_at": datetime.utcnow().isoformat(),
            "metadata": self.discovery_metadata or {},
        }

    @classmethod
    def from_llm_discovery(
        cls,
        intent: str,
        platform: str,
        command: str,
        confidence: float,
        reasoning: Optional[str] = None,
        alternatives: Optional[list] = None,
    ):
        """Create CommandMapping from LLM discovery result"""
        discovery_metadata = {
            "discovery_method": "llm",
            "reasoning": reasoning,
            "alternatives": alternatives or [],
            "discovered_at": datetime.utcnow().isoformat(),
        }

        return cls(
            intent=intent,
            platform=platform,
            command=command,
            confidence=confidence,
            source="llm_discovery",
            usage_count=1,
            last_used=datetime.utcnow(),
            discovery_metadata=discovery_metadata,
        )


class CommunitySync(Base):
    """
    Track community repository synchronization
    Enables automated updates of command mappings
    """

    __tablename__ = "community_sync_logs"

    id = Column(Integer, primary_key=True, index=True)

    # Sync metadata
    repository_url = Column(String(255), nullable=False)
    sync_type = Column(
        String(50), nullable=False
    )  # initial_seed, update_check, forced_sync

    # Sync results
    commands_discovered = Column(Integer, default=0)
    commands_updated = Column(Integer, default=0)
    commands_added = Column(Integer, default=0)
    sync_duration_ms = Column(Integer, nullable=True)

    # Status tracking
    status = Column(String(20), nullable=False)  # success, failure, partial
    error_message = Column(Text, nullable=True)

    # Timestamps
    synced_at = Column(DateTime, default=func.now(), index=True)

    # Sync metadata
    sync_metadata = Column(JSONB, nullable=True)  # Store additional sync info

    __table_args__ = (
        Index("idx_sync_status", "status"),
        Index("idx_sync_type", "sync_type"),
        Index("idx_synced_at_desc", "synced_at", postgresql_using="btree"),
    )

    def __repr__(self):
        return f"<CommunitySync(url='{self.repository_url}', status='{self.status}', discovered={self.commands_discovered})>"


class DiscoveryAnalytics(Base):
    """
    Daily analytics for three-tier discovery performance
    Enables data-driven optimization
    """

    __tablename__ = "discovery_analytics"

    id = Column(Integer, primary_key=True, index=True)

    # Date partition
    date = Column(DateTime, nullable=False, index=True)

    # Request volume metrics
    total_requests = Column(Integer, default=0)
    unique_intents = Column(Integer, default=0)
    unique_platforms = Column(Integer, default=0)

    # Three-tier performance metrics
    redis_hits = Column(Integer, default=0)
    postgres_hits = Column(Integer, default=0)
    llm_discoveries = Column(Integer, default=0)

    # Response time metrics (in milliseconds)
    avg_redis_response_ms = Column(Float, nullable=True)
    avg_postgres_response_ms = Column(Float, nullable=True)
    avg_llm_response_ms = Column(Float, nullable=True)

    # Cache performance
    cache_hit_rate = Column(Float, nullable=True)  # Combined Redis + PostgreSQL
    redis_hit_rate = Column(Float, nullable=True)
    postgres_hit_rate = Column(Float, nullable=True)

    # Learning metrics
    new_mappings_learned = Column(Integer, default=0)
    confidence_scores_avg = Column(Float, nullable=True)

    # System health
    error_count = Column(Integer, default=0)
    timeout_count = Column(Integer, default=0)

    # Daily metadata
    peak_concurrent_requests = Column(Integer, default=0)
    analytics_metadata = Column(JSONB, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("date", name="uq_analytics_date"),
        Index("idx_date_desc", "date", postgresql_using="btree"),
        Index("idx_cache_hit_rate", "cache_hit_rate"),
        Index("idx_total_requests", "total_requests"),
    )

    def __repr__(self):
        return f"<DiscoveryAnalytics(date='{self.date.date()}', total_requests={self.total_requests}, cache_hit_rate={self.cache_hit_rate})>"

    def calculate_performance_grade(self) -> str:
        """Calculate performance grade based on metrics"""
        if not self.cache_hit_rate:
            return "F"

        if self.cache_hit_rate >= 90:
            return "A+"
        elif self.cache_hit_rate >= 80:
            return "A"
        elif self.cache_hit_rate >= 70:
            return "B"
        elif self.cache_hit_rate >= 60:
            return "C"
        elif self.cache_hit_rate >= 50:
            return "D"
        else:
            return "F"

    def get_optimization_recommendations(self) -> list:
        """Get optimization recommendations based on current metrics"""
        recommendations = []

        if self.redis_hit_rate and self.redis_hit_rate < 50:
            recommendations.append(
                "Increase Redis cache TTL for frequently used commands"
            )

        if self.postgres_hit_rate and self.postgres_hit_rate < 30:
            recommendations.append("Expand community command seeding")

        if self.llm_discoveries and self.llm_discoveries > self.total_requests * 0.3:
            recommendations.append("Implement predictive caching for common patterns")

        if self.avg_llm_response_ms and self.avg_llm_response_ms > 3000:
            recommendations.append("Optimize LLM prompt templates for faster responses")

        if self.error_count > self.total_requests * 0.05:
            recommendations.append(
                "Investigate error patterns and improve error handling"
            )

        return recommendations


# Migration helper functions
async def seed_initial_data(session):
    """Seed database with initial community mappings"""
    from app.services.community_mapper import community_mapper

    # Get community mappings
    mappings = community_mapper.get_combined_mappings()

    # Insert initial mappings
    for platform, commands in mappings.items():
        for command in commands:
            # Create a generic intent from the command
            intent = command.replace("show ", "").replace(" ", "_")

            mapping = CommandMapping(
                intent=intent,
                platform=platform,
                command=command,
                confidence=0.8,  # Default community confidence
                source="community",
                usage_count=0,
                discovery_metadata={
                    "seeded_from": "community_mapper",
                    "seed_date": datetime.utcnow().isoformat(),
                },
            )

            session.add(mapping)

    await session.commit()

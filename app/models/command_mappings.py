# File: ~/net-chatbot/app/models/command_mappings.py
"""
Enhanced Database Models for Three-Tier Discovery System with Learning
PostgreSQL + Redis optimized schemas with intelligent confidence management
- NAPALM: 0.95-1.0 confidence (industry standard)
- Community: 0.8-0.9 confidence (community verified)
- LLM: 0.7 initial → 0.9+ with learning (+0.1 per success)
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
    Enhanced command discovery mappings with learning capabilities
    Optimized for three-tier lookup performance + confidence evolution
    """

    __tablename__ = "command_mappings"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Core discovery data
    intent = Column(String(100), nullable=False, index=True)
    platform = Column(String(50), nullable=False, index=True)
    command = Column(Text, nullable=False)

    # Enhanced confidence system with learning
    confidence = Column(Float, default=0.7, index=True)  # Start LLM at 0.7
    source = Column(
        String(20), default="llm_discovery", index=True
    )  # community, llm_discovery, manual

    # Usage tracking for intelligent caching
    usage_count = Column(Integer, default=0, index=True)
    last_used = Column(DateTime, default=None, index=True)

    # NEW: Learning and performance tracking
    execution_count = Column(Integer, default=0, index=True)
    success_count = Column(Integer, default=0, index=True)
    last_success = Column(DateTime, default=None)
    last_failure = Column(DateTime, default=None)
    confidence_history = Column(JSONB, default=list)  # Track confidence evolution

    # Audit fields
    created_at = Column(DateTime, default=func.now(), index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Metadata for discoveries and conflicts
    discovery_metadata = Column(JSONB, nullable=True)

    # Performance optimization constraints
    __table_args__ = (
        UniqueConstraint("intent", "platform", name="uq_intent_platform"),
        Index("idx_intent_platform", "intent", "platform"),  # Primary lookup index
        Index("idx_confidence_desc", "confidence", postgresql_using="btree"),
        Index("idx_usage_performance", "usage_count", "success_count"),
        Index("idx_learning_candidates", "execution_count", "confidence"),
        Index("idx_last_used_desc", "last_used", postgresql_using="btree"),
        Index("idx_source_confidence", "source", "confidence"),
    )

    def __repr__(self):
        success_rate = self.get_success_rate()
        return (
            f"<CommandMapping(intent='{self.intent}', platform='{self.platform}', "
            f"confidence={self.confidence:.2f}, success_rate={success_rate:.1%})>"
        )

    def get_success_rate(self) -> float:
        """Calculate current success rate"""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count

    def should_promote_confidence(self) -> bool:
        """Determine if command confidence should be promoted"""
        # Promote LLM discoveries that prove themselves
        return (
            self.source == "llm_discovery"
            and self.execution_count >= 2  # At least 2 executions
            and self.get_success_rate() >= 0.8  # 80%+ success rate
            and self.confidence < 0.9  # Not already at max
        )

    def should_demote_confidence(self) -> bool:
        """Determine if command confidence should be demoted"""
        # Demote commands with poor performance
        return (
            self.execution_count >= 5  # Meaningful sample size
            and self.get_success_rate() < 0.6  # Less than 60% success
            and self.confidence > 0.4  # Don't demote below minimum threshold
        )

    def calculate_new_confidence(self, success: bool) -> float:
        """Calculate new confidence after execution result"""
        current_confidence = self.confidence

        if success and self.should_promote_confidence():
            # Promote by +0.1 for LLM learning, cap at 0.9
            new_confidence = min(0.9, current_confidence + 0.1)
        elif not success and self.should_demote_confidence():
            # Demote based on failure rate, but don't go below 0.3
            success_rate = self.get_success_rate()
            penalty_factor = max(0.3, success_rate)
            new_confidence = max(0.3, current_confidence * penalty_factor)
        else:
            # No confidence change
            new_confidence = current_confidence

        return new_confidence

    async def record_execution(
        self, session, success: bool, execution_time: float = None
    ):
        """Record execution result and update confidence if needed"""
        # Update execution counters
        self.execution_count += 1
        if success:
            self.success_count += 1
            self.last_success = datetime.utcnow()
        else:
            self.last_failure = datetime.utcnow()

        # Calculate new confidence
        old_confidence = self.confidence
        new_confidence = self.calculate_new_confidence(success)

        # Update confidence if changed
        if new_confidence != old_confidence:
            self.confidence = new_confidence

            # Add to confidence history
            if not self.confidence_history:
                self.confidence_history = []

            self.confidence_history.append(
                {
                    "old_confidence": old_confidence,
                    "new_confidence": new_confidence,
                    "reason": "learning_adjustment"
                    if success
                    else "performance_demotion",
                    "success": success,
                    "execution_count": self.execution_count,
                    "success_rate": self.get_success_rate(),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

        # Update usage tracking
        self.usage_count += 1
        self.last_used = datetime.utcnow()
        self.updated_at = datetime.utcnow()

        # Commit changes
        await session.commit()

        return new_confidence != old_confidence

    def to_cache_dict(self) -> dict:
        """Convert to Redis cache format with performance data"""
        return {
            "command": self.command,
            "confidence": float(self.confidence),
            "source": self.source,
            "usage_count": self.usage_count,
            "success_rate": self.get_success_rate(),
            "execution_count": self.execution_count,
            "cached_at": datetime.utcnow().isoformat(),
            "metadata": self.discovery_metadata or {},
            "learning_status": self.get_learning_status(),
        }

    def get_learning_status(self) -> str:
        """Get current learning status for monitoring"""
        if self.source != "llm_discovery":
            return "static"
        elif self.execution_count == 0:
            return "unproven"
        elif self.should_promote_confidence():
            return "learning_well"
        elif self.should_demote_confidence():
            return "performing_poorly"
        else:
            return "stable"

    @classmethod
    def from_llm_discovery(
        cls,
        intent: str,
        platform: str,
        command: str,
        confidence: float = 0.7,  # LLM starts at 0.7
        reasoning: Optional[str] = None,
        alternatives: Optional[list] = None,
    ):
        """Create CommandMapping from LLM discovery result with learning setup"""
        discovery_metadata = {
            "discovery_method": "llm",
            "reasoning": reasoning,
            "alternatives": alternatives or [],
            "discovered_at": datetime.utcnow().isoformat(),
            "initial_confidence": confidence,
            "learning_enabled": True,
        }

        return cls(
            intent=intent,
            platform=platform,
            command=command,
            confidence=confidence,
            source="llm_discovery",
            usage_count=0,
            execution_count=0,
            success_count=0,
            confidence_history=[
                {
                    "confidence": confidence,
                    "reason": "initial_llm_discovery",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ],
            discovery_metadata=discovery_metadata,
        )


class CommandConflicts(Base):
    """
    Track conflicts between different sources for same intent+platform
    Enables auditing and learning from conflict resolution
    """

    __tablename__ = "command_conflicts"

    id = Column(Integer, primary_key=True, index=True)

    # Conflict identification
    intent = Column(String(100), nullable=False, index=True)
    platform = Column(String(50), nullable=False, index=True)

    # Resolution details
    winning_source = Column(String(20), nullable=False)
    winning_confidence = Column(Float, nullable=False)
    losing_sources = Column(JSONB, nullable=False)  # Array of losing source details
    confidence_diff = Column(Float, nullable=False)

    # Conflict context
    conflict_reason = Column(Text, nullable=True)
    resolution_strategy = Column(String(50), default="confidence_based")

    # Timestamps
    created_at = Column(DateTime, default=func.now(), index=True)

    __table_args__ = (
        Index("idx_conflict_intent_platform", "intent", "platform"),
        Index("idx_conflict_source", "winning_source"),
        Index("idx_conflict_confidence", "confidence_diff"),
    )

    def __repr__(self):
        return (
            f"<CommandConflicts(intent='{self.intent}', platform='{self.platform}', "
            f"winner='{self.winning_source}', diff={self.confidence_diff:.2f})>"
        )


class CommunitySync(Base):
    """
    Enhanced community repository synchronization tracking
    """

    __tablename__ = "community_sync_logs"

    id = Column(Integer, primary_key=True, index=True)

    # Sync metadata
    repository_url = Column(String(255), nullable=False)
    sync_type = Column(String(50), nullable=False)

    # Enhanced sync results
    commands_discovered = Column(Integer, default=0)
    commands_updated = Column(Integer, default=0)
    commands_added = Column(Integer, default=0)
    conflicts_resolved = Column(Integer, default=0)  # NEW
    confidence_adjustments = Column(Integer, default=0)  # NEW
    sync_duration_ms = Column(Integer, nullable=True)

    # Status tracking
    status = Column(String(20), nullable=False)
    error_message = Column(Text, nullable=True)

    # Timestamps
    synced_at = Column(DateTime, default=func.now(), index=True)

    # Enhanced sync metadata
    sync_metadata = Column(JSONB, nullable=True)

    __table_args__ = (
        Index("idx_sync_status", "status"),
        Index("idx_sync_type", "sync_type"),
        Index("idx_synced_at_desc", "synced_at"),
    )


class DiscoveryAnalytics(Base):
    """
    Enhanced analytics with learning metrics
    """

    __tablename__ = "discovery_analytics"

    id = Column(Integer, primary_key=True, index=True)
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
    cache_hit_rate = Column(Float, nullable=True)
    redis_hit_rate = Column(Float, nullable=True)
    postgres_hit_rate = Column(Float, nullable=True)

    # NEW: Learning metrics
    new_mappings_learned = Column(Integer, default=0)
    confidence_promotions = Column(Integer, default=0)  # LLM commands promoted
    confidence_demotions = Column(
        Integer, default=0
    )  # Commands demoted for poor performance
    avg_llm_confidence = Column(Float, nullable=True)  # Average LLM confidence
    learning_velocity = Column(Float, nullable=True)  # Rate of confidence improvement

    # System health
    error_count = Column(Integer, default=0)
    timeout_count = Column(Integer, default=0)

    # Enhanced metadata
    peak_concurrent_requests = Column(Integer, default=0)
    analytics_metadata = Column(JSONB, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("date", name="uq_analytics_date"),
        Index("idx_date_desc", "date"),
        Index("idx_cache_hit_rate", "cache_hit_rate"),
        Index("idx_learning_velocity", "learning_velocity"),
    )

    def calculate_learning_grade(self) -> str:
        """Calculate learning system performance grade"""
        if not self.learning_velocity:
            return "N/A"

        # Grade based on learning velocity and confidence promotions
        promotion_rate = self.confidence_promotions / max(self.new_mappings_learned, 1)

        if promotion_rate >= 0.8 and self.learning_velocity >= 0.1:
            return "A+"
        elif promotion_rate >= 0.6 and self.learning_velocity >= 0.05:
            return "A"
        elif promotion_rate >= 0.4:
            return "B"
        elif promotion_rate >= 0.2:
            return "C"
        else:
            return "D"


# Helper functions for learning system
async def update_command_performance(
    session, intent: str, platform: str, success: bool, execution_time: float = None
):
    """Update command performance and learning metrics"""
    from sqlalchemy import select

    # Get command mapping
    stmt = select(CommandMapping).where(
        CommandMapping.intent == intent, CommandMapping.platform == platform
    )
    result = await session.execute(stmt)
    mapping = result.scalar_one_or_none()

    if mapping:
        # Record execution and update confidence
        confidence_changed = await mapping.record_execution(
            session, success, execution_time
        )
        return mapping, confidence_changed

    return None, False


async def get_learning_candidates(session, min_executions: int = 2):
    """Get LLM discoveries that are candidates for confidence promotion"""
    from sqlalchemy import select

    stmt = (
        select(CommandMapping)
        .where(
            CommandMapping.source == "llm_discovery",
            CommandMapping.execution_count >= min_executions,
            CommandMapping.confidence < 0.9,
        )
        .order_by(CommandMapping.get_success_rate().desc())
    )

    result = await session.execute(stmt)
    return result.scalars().all()


# Migration helper functions
async def migrate_learning_schema(session):
    """Add learning columns to existing command_mappings table"""
    from sqlalchemy import text

    learning_columns = [
        "ALTER TABLE command_mappings ADD COLUMN IF NOT EXISTS execution_count INTEGER DEFAULT 0",
        "ALTER TABLE command_mappings ADD COLUMN IF NOT EXISTS success_count INTEGER DEFAULT 0",
        "ALTER TABLE command_mappings ADD COLUMN IF NOT EXISTS last_success TIMESTAMP",
        "ALTER TABLE command_mappings ADD COLUMN IF NOT EXISTS last_failure TIMESTAMP",
        "ALTER TABLE command_mappings ADD COLUMN IF NOT EXISTS confidence_history JSONB DEFAULT '[]'::jsonb",
    ]

    for sql in learning_columns:
        await session.execute(text(sql))

    await session.commit()

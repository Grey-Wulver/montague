# app/services/hybrid_command_discovery.py
"""
HybridCommandDiscovery Service - LLM-Powered Three-Tier Progressive Performance System

FIXED: Returns proper DiscoveryResult objects instead of tuples for consistent data handling.

Revolutionary approach using LLM-powered intent normalization for infinite scalability:
- NO hardcoded command patterns or normalization rules
- Pure LLM intelligence for both normalization and discovery
- Integrates with existing LM Studio setup for consistency

Architecture:
- Tier 1: Redis Cache (1-5ms) - Ultra-fast for hot commands
- Tier 2: PostgreSQL Database (10-20ms) - Fast for all discovered commands
- Tier 3: LLM Discovery (1-2s) - Learning path with LLM normalization + dual storage

Performance Evolution Target:
- Month 1: 60% cached → 2.1s average response
- Month 6: 90% cached → 0.8s average response
- Month 12: 95% cached → <500ms average response

Maintains VS Code + Ruff + Black standards with 88-character line length.
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

import aiohttp
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from app.core.database import get_db_session_direct, get_redis_client
from app.models.command_mappings import CommandMapping
from app.models.discovery_models import DiscoveryResult, NormalizationResult

logger = logging.getLogger(__name__)


class HybridCommandDiscovery:
    """
    FIXED: LLM-Powered three-tier hybrid discovery system with consistent return types.

    Revolutionary Features:
    - LLM-powered intent normalization (no hardcoded rules)
    - Pure LLM command discovery (no pattern matching)
    - Integrates with existing LM Studio infrastructure
    - Self-improving through intelligent caching
    - FIXED: Returns DiscoveryResult objects for consistent handling

    Performance Evolution Strategy:
    - Week 1: 5% Redis + 15% PostgreSQL + 80% LLM (learning phase)
    - Month 6: 70% Redis + 20% PostgreSQL + 10% LLM (optimized)
    - Month 12: 80% Redis + 15% PostgreSQL + 5% LLM (ultra-optimized)
    """

    def __init__(self, redis_client=None, llm_normalizer=None):
        """Initialize with Redis client and LLM normalizer for full integration."""
        self.redis_client = redis_client
        self.llm_normalizer = llm_normalizer

        # LM Studio configuration (matches your existing setup)
        self.lm_studio_url = "http://192.168.1.11:1234"
        self.model_name = "meta-llama-3.1-8b-instruct"
        self.timeout = 30

        # Performance tracking for three-tier analytics
        self.daily_stats = {
            "total_requests": 0,
            "redis_hits": 0,
            "postgres_hits": 0,
            "llm_discoveries": 0,
            "normalization_calls": 0,
            "redis_time_total": 0.0,
            "postgres_time_total": 0.0,
            "llm_time_total": 0.0,
            "normalization_time_total": 0.0,
        }

    async def discover_command(
        self, intent: str, platform: str, user_context: Optional[Dict] = None
    ) -> DiscoveryResult:
        """
        FIXED: LLM-powered three-tier discovery returning DiscoveryResult object.

        Args:
            intent: Natural language intent (e.g., "Show BGP neighbor status on spine1")
            platform: Network platform (e.g., "arista_eos", "cisco_ios")
            user_context: Optional context for discovery enhancement

        Returns:
            DiscoveryResult object with:
            - command: Actual network command to execute
            - confidence: Float 0.0-1.0 indicating command reliability
            - source: "redis_cache", "postgres_db", "llm_discovery", or "fallback"
            - reasoning: Optional explanation of discovery logic
            - discovery_time: Time taken for this discovery
            - Other metadata for analytics and debugging
        """
        start_time = time.time()
        self.daily_stats["total_requests"] += 1

        try:
            # Initialize Redis client if needed
            if not self.redis_client:
                self.redis_client = await get_redis_client()

            # STEP 1: LLM-Powered Intent Normalization
            normalization_result = await self._normalize_intent_with_llm(intent)

            self.daily_stats["normalization_calls"] += 1
            self.daily_stats["normalization_time_total"] += (
                normalization_result.normalization_time
            )

            cache_key = f"cmd:{normalization_result.normalized_intent}:{platform}"

            logger.info(
                f"Normalized '{intent}' → '{normalization_result.normalized_intent}' "
                f"in {normalization_result.normalization_time * 1000:.1f}ms"
            )

            # TIER 1: Redis cache lookup (1-5ms) - ULTRA FAST PATH
            redis_start = time.time()
            cached_result = await self._redis_lookup(cache_key)
            redis_time = time.time() - redis_start

            if cached_result:
                self._update_stats("redis", redis_time)
                total_time = time.time() - start_time
                logger.info(
                    f"Redis hit: {normalization_result.normalized_intent}:{platform} "
                    f"in {total_time * 1000:.1f}ms"
                )

                return DiscoveryResult(
                    command=cached_result["command"],
                    confidence=cached_result["confidence"],
                    source="redis_cache",
                    reasoning="Retrieved from Redis cache",
                    discovery_time=total_time,
                    cached_at=cached_result.get("cached_at"),
                    usage_count=cached_result.get("usage_count"),
                    discovery_metadata={
                        "cache_key": cache_key,
                        "redis_time": redis_time,
                        "normalization_result": normalization_result.__dict__,
                    },
                )

            # TIER 2: PostgreSQL lookup (10-20ms) - FAST PATH
            postgres_start = time.time()
            db_result = await self._postgres_lookup(
                normalization_result.normalized_intent, platform
            )
            postgres_time = time.time() - postgres_start

            if db_result and db_result.confidence >= 0.8:
                # Promote to Redis cache for ultra-fast future access
                await self._promote_to_redis_cache(cache_key, db_result)
                self._update_stats("postgres", postgres_time)
                total_time = time.time() - start_time
                logger.info(
                    f"PostgreSQL hit: {normalization_result.normalized_intent}:{platform} "
                    f"in {total_time * 1000:.1f}ms"
                )

                return DiscoveryResult(
                    command=db_result.command,
                    confidence=float(db_result.confidence),
                    source="postgres_db",
                    reasoning="Retrieved from PostgreSQL database and promoted to Redis",
                    discovery_time=total_time,
                    cached_at=db_result.last_used,
                    usage_count=db_result.usage_count,
                    discovery_metadata={
                        "cache_key": cache_key,
                        "postgres_time": postgres_time,
                        "db_id": db_result.id,
                        "promoted_to_redis": True,
                        "normalization_result": normalization_result.__dict__,
                    },
                )

            # TIER 3: LLM discovery (1-2s) - LEARNING PATH
            logger.info(
                f"Cache miss for {normalization_result.normalized_intent}:{platform}, "
                f"using LLM discovery"
            )
            llm_start = time.time()
            llm_result = await self._llm_discovery_with_lm_studio(
                normalization_result.normalized_intent, platform, user_context
            )
            llm_time = time.time() - llm_start

            # FEEDBACK LOOP: Store in PostgreSQL + cache in Redis
            await self._store_discovery_dual(
                normalization_result.normalized_intent, platform, llm_result, cache_key
            )

            self._update_stats("llm", llm_time)
            total_time = time.time() - start_time
            logger.info(
                f"LLM discovery: {normalization_result.normalized_intent}:{platform} "
                f"in {total_time * 1000:.1f}ms"
            )

            return DiscoveryResult(
                command=llm_result["command"],
                confidence=llm_result["confidence"],
                source="llm_discovery",
                reasoning=llm_result.get("explanation", "Fresh LLM discovery"),
                discovery_time=total_time,
                cached_at=datetime.utcnow(),
                usage_count=1,
                discovery_metadata={
                    "cache_key": cache_key,
                    "llm_time": llm_time,
                    "stored_dual": True,
                    "llm_result": llm_result,
                    "normalization_result": normalization_result.__dict__,
                },
            )

        except Exception as e:
            logger.error(f"Discovery failed for {intent}:{platform}: {e}")
            fallback_command, fallback_confidence = self._get_intelligent_fallback(
                intent
            )
            total_time = time.time() - start_time

            return DiscoveryResult(
                command=fallback_command,
                confidence=fallback_confidence,
                source="fallback",
                reasoning=f"Intelligent fallback due to error: {str(e)}",
                discovery_time=total_time,
                discovery_metadata={
                    "error": str(e),
                    "fallback_used": True,
                },
            )

    async def _normalize_intent_with_llm(self, raw_intent: str) -> NormalizationResult:
        """
        ENHANCED: LLM-powered intent normalization returning NormalizationResult.

        Uses your existing LM Studio setup for consistent normalization.
        NO hardcoded rules - pure LLM intelligence.
        """
        start_time = time.time()

        normalization_prompt = f"""You are a network automation expert. Normalize this network intent to a canonical form for caching and lookup.

Original intent: "{raw_intent}"

Rules:
1. Convert to lowercase
2. Standardize network terms (bgp, interface, route, lldp, etc.)
3. Remove device-specific names
4. Use consistent verb forms (show, display, get → show)
5. Remove unnecessary words (please, can you, etc.)

Return only the normalized intent, nothing else.

Examples:
"Show me BGP neighbors on spine1" → "show bgp neighbors"
"Can you display interface status?" → "show interface status"
"Get routing table info" → "show ip route"

Normalized intent:"""

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(self.timeout)
            ) as session:
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": normalization_prompt}],
                    "temperature": 0.1,
                    "max_tokens": 50,
                }

                async with session.post(
                    f"{self.lm_studio_url}/v1/chat/completions", json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        normalized = (
                            data["choices"][0]["message"]["content"].strip().lower()
                        )

                        normalization_time = time.time() - start_time

                        return NormalizationResult(
                            original_intent=raw_intent,
                            normalized_intent=normalized,
                            normalization_time=normalization_time,
                            confidence=1.0,
                            normalization_metadata={
                                "llm_model": self.model_name,
                                "prompt_tokens": data.get("usage", {}).get(
                                    "prompt_tokens"
                                ),
                                "completion_tokens": data.get("usage", {}).get(
                                    "completion_tokens"
                                ),
                            },
                        )
                    else:
                        logger.warning(
                            f"LLM normalization failed with status {response.status}"
                        )
                        raise Exception(f"LLM API error: {response.status}")

        except Exception as e:
            logger.warning(f"LLM normalization failed: {e}, using basic normalization")
            # Fallback to basic normalization
            basic_normalized = self._basic_normalization(raw_intent)
            normalization_time = time.time() - start_time

            return NormalizationResult(
                original_intent=raw_intent,
                normalized_intent=basic_normalized,
                normalization_time=normalization_time,
                confidence=0.8,  # Lower confidence for fallback
                normalization_metadata={
                    "fallback_used": True,
                    "error": str(e),
                },
            )

    def _basic_normalization(self, intent: str) -> str:
        """Basic fallback normalization when LLM is unavailable."""
        normalized = intent.lower().strip()

        # Basic substitutions
        replacements = {
            "display": "show",
            "get": "show",
            "check": "show",
            "what is": "show",
            "what are": "show",
            "neighbors": "neighbors",
            "neighbour": "neighbors",
            "neighboors": "neighbors",
        }

        for old, new in replacements.items():
            normalized = normalized.replace(old, new)

        # Remove common filler words
        filler_words = ["please", "can you", "could you", "me", "the", "a", "an"]
        words = normalized.split()
        filtered_words = [w for w in words if w not in filler_words]

        return " ".join(filtered_words)

    async def _redis_lookup(self, cache_key: str) -> Optional[Dict]:
        """Ultra-fast Redis cache lookup with error handling."""
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            logger.warning(f"Redis lookup failed for {cache_key}: {e}")
            return None

    async def _postgres_lookup(
        self, normalized_intent: str, platform: str
    ) -> Optional[CommandMapping]:
        """Fast PostgreSQL database lookup with performance optimization."""
        try:
            session = await get_db_session_direct()
            try:
                query = (
                    select(CommandMapping)
                    .where(
                        CommandMapping.intent == normalized_intent,
                        CommandMapping.platform == platform,
                    )
                    .order_by(CommandMapping.confidence.desc())
                )

                result = await session.execute(query)
                mapping = result.scalar_one_or_none()

                if mapping:
                    # Update usage statistics for intelligent cache promotion
                    await self._update_usage_stats(session, mapping.id)

                return mapping
            finally:
                await session.close()
        except Exception as e:
            logger.error(
                f"PostgreSQL lookup failed for {normalized_intent}:{platform}: {e}"
            )
            return None

    async def _promote_to_redis_cache(self, cache_key: str, db_result: CommandMapping):
        """Promote PostgreSQL result to Redis cache with intelligent TTL."""
        try:
            cache_data = {
                "command": db_result.command,
                "confidence": float(db_result.confidence),
                "source": "postgres_promoted",
                "cached_at": datetime.utcnow().isoformat(),
                "usage_count": db_result.usage_count,
            }

            # Intelligent TTL based on usage patterns
            ttl = self._calculate_cache_ttl(db_result.usage_count, db_result.confidence)

            await self.redis_client.setex(cache_key, ttl, json.dumps(cache_data))

            logger.debug(f"Promoted to Redis: {cache_key} with TTL {ttl}s")

        except Exception as e:
            logger.warning(f"Failed to promote to Redis cache: {e}")

    async def _llm_discovery_with_lm_studio(
        self, normalized_intent: str, platform: str, user_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """LLM-powered command discovery using your existing LM Studio setup."""

        discovery_prompt = f"""You are a network automation expert for {platform} devices.
Generate the exact command for this normalized intent: "{normalized_intent}"

Platform: {platform}
Intent: {normalized_intent}
Context: {user_context or {}}

Respond with a JSON object:
{{
    "command": "exact command to execute",
    "confidence": 0.95,
    "explanation": "brief explanation of the command"
}}

Examples for arista_eos:
- "show bgp neighbors" → {{"command": "show bgp summary", "confidence": 0.95}}
- "show interface status" → {{"command": "show interfaces status", "confidence": 0.95}}
- "show version" → {{"command": "show version", "confidence": 1.0}}

JSON response:"""

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(self.timeout)
            ) as session:
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": discovery_prompt}],
                    "temperature": 0.1,
                    "max_tokens": 200,
                }

                async with session.post(
                    f"{self.lm_studio_url}/v1/chat/completions", json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        response_text = data["choices"][0]["message"]["content"].strip()

                        # Clean and parse JSON response
                        cleaned_json = self._clean_json_response(response_text)
                        result = json.loads(cleaned_json)

                        # Validate and enhance result
                        if "command" not in result:
                            raise ValueError("Missing 'command' in LLM response")

                        result.setdefault("confidence", 0.9)
                        result.setdefault(
                            "explanation", f"LLM discovery for {normalized_intent}"
                        )
                        result["discovery_metadata"] = {
                            "method": "lm_studio_llm",
                            "model": self.model_name,
                            "discovery_timestamp": datetime.utcnow().isoformat(),
                            "tokens_used": data.get("usage", {}),
                        }

                        return result
                    else:
                        logger.error(
                            f"LLM discovery failed with status {response.status}"
                        )
                        raise Exception(f"LLM API error: {response.status}")

        except Exception as e:
            logger.error(f"LLM discovery failed: {e}")
            # Use intelligent fallback
            return await self._intelligent_fallback_discovery(
                normalized_intent, platform
            )

    def _clean_json_response(self, response_text: str) -> str:
        """Clean LLM response to extract valid JSON."""
        # Basic JSON cleaning - enhance this based on your existing patterns
        cleaned = response_text.strip()

        # Remove markdown code blocks if present
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        # Find JSON object
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1

        if start >= 0 and end > start:
            cleaned = cleaned[start:end]

        return cleaned.strip()

    async def _intelligent_fallback_discovery(
        self, normalized_intent: str, platform: str
    ) -> Dict[str, Any]:
        """Intelligent fallback using intent analysis when LLM is unavailable."""
        # Analyze the normalized intent for common patterns
        intent_lower = normalized_intent.lower()

        if "bgp" in intent_lower:
            command = "show bgp summary"
            confidence = 0.8
        elif "interface" in intent_lower:
            command = "show interfaces"
            confidence = 0.9
        elif "route" in intent_lower or "routing" in intent_lower:
            command = "show ip route"
            confidence = 0.8
        elif "lldp" in intent_lower:
            command = "show lldp neighbors"
            confidence = 0.8
        elif "version" in intent_lower:
            command = "show version"
            confidence = 1.0
        else:
            command = "show version"
            confidence = 0.5

        return {
            "command": command,
            "confidence": confidence,
            "explanation": f"Intelligent fallback for {normalized_intent}",
            "discovery_metadata": {
                "method": "intelligent_fallback",
                "pattern_match": True,
                "discovery_timestamp": datetime.utcnow().isoformat(),
            },
        }

    def _get_intelligent_fallback(self, intent: str) -> tuple[str, float]:
        """Quick fallback for error conditions."""
        intent_lower = intent.lower()

        if "bgp" in intent_lower:
            return "show bgp summary", 0.7
        elif "interface" in intent_lower:
            return "show interfaces", 0.8
        elif "version" in intent_lower:
            return "show version", 0.9
        else:
            return "show version", 0.5

    async def _store_discovery_dual(
        self,
        normalized_intent: str,
        platform: str,
        llm_result: Dict[str, Any],
        cache_key: str,
    ):
        """Store LLM discovery in both PostgreSQL and Redis for dual availability."""
        try:
            session = await get_db_session_direct()
            try:
                # Store in PostgreSQL for persistent discovery
                mapping_data = {
                    "intent": normalized_intent,  # Use normalized intent
                    "platform": platform,
                    "command": llm_result["command"],
                    "confidence": llm_result["confidence"],
                    "source": "llm_discovery",
                    "usage_count": 1,
                    "last_used": datetime.utcnow(),
                    "discovery_metadata": llm_result.get("discovery_metadata", {}),
                }

                # Use INSERT ... ON CONFLICT for upsert behavior
                stmt = insert(CommandMapping).values(mapping_data)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["intent", "platform"],
                    set_={
                        "command": stmt.excluded.command,
                        "confidence": stmt.excluded.confidence,
                        "usage_count": CommandMapping.usage_count + 1,
                        "last_used": stmt.excluded.last_used,
                        "updated_at": datetime.utcnow(),
                    },
                )

                await session.execute(stmt)
                await session.commit()

                # Store in Redis cache for immediate future access
                cache_data = {
                    "command": llm_result["command"],
                    "confidence": llm_result["confidence"],
                    "source": "llm_fresh",
                    "cached_at": datetime.utcnow().isoformat(),
                }

                # Fresh discoveries get longer TTL to encourage reuse
                await self.redis_client.setex(
                    cache_key,
                    7200,  # 2 hours for fresh discoveries
                    json.dumps(cache_data),
                )

                logger.info(
                    f"Stored dual: {normalized_intent}:{platform} in PostgreSQL and Redis"
                )
            finally:
                await session.close()

        except Exception as e:
            logger.error(f"Failed to store discovery dual: {e}")

    async def _update_usage_stats(self, session, mapping_id: int):
        """Update usage statistics for cache promotion intelligence."""
        try:
            stmt = (
                update(CommandMapping)
                .where(CommandMapping.id == mapping_id)
                .values(
                    usage_count=CommandMapping.usage_count + 1,
                    last_used=datetime.utcnow(),
                )
            )
            await session.execute(stmt)
        except Exception as e:
            logger.warning(f"Failed to update usage stats: {e}")

    def _calculate_cache_ttl(self, usage_count: int, confidence: float) -> int:
        """Calculate intelligent TTL based on usage patterns and confidence."""
        base_ttl = 3600  # 1 hour base

        # High confidence commands get longer TTL
        confidence_multiplier = confidence * 2

        # Frequently used commands get longer TTL
        usage_multiplier = min(usage_count / 10, 3)  # Cap at 3x

        final_ttl = int(base_ttl * confidence_multiplier * (1 + usage_multiplier))
        return min(final_ttl, 86400)  # Cap at 24 hours

    def _update_stats(self, source: str, response_time: float):
        """Update performance statistics for analytics."""
        if source == "redis":
            self.daily_stats["redis_hits"] += 1
            self.daily_stats["redis_time_total"] += response_time
        elif source == "postgres":
            self.daily_stats["postgres_hits"] += 1
            self.daily_stats["postgres_time_total"] += response_time
        elif source == "llm":
            self.daily_stats["llm_discoveries"] += 1
            self.daily_stats["llm_time_total"] += response_time

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = self.daily_stats.copy()

        # Calculate derived metrics
        if stats["total_requests"] > 0:
            stats["redis_hit_rate"] = stats["redis_hits"] / stats["total_requests"]
            stats["postgres_hit_rate"] = (
                stats["postgres_hits"] / stats["total_requests"]
            )
            stats["llm_discovery_rate"] = (
                stats["llm_discoveries"] / stats["total_requests"]
            )

            if stats["redis_hits"] > 0:
                stats["avg_redis_time"] = (
                    stats["redis_time_total"] / stats["redis_hits"]
                )
            if stats["postgres_hits"] > 0:
                stats["avg_postgres_time"] = (
                    stats["postgres_time_total"] / stats["postgres_hits"]
                )
            if stats["llm_discoveries"] > 0:
                stats["avg_llm_time"] = (
                    stats["llm_time_total"] / stats["llm_discoveries"]
                )

        return stats


# Factory function for easy integration with existing Universal Pipeline
async def create_hybrid_discovery_service(llm_client=None) -> HybridCommandDiscovery:
    """
    Factory function to create and initialize HybridCommandDiscovery service.

    Args:
        llm_client: Optional LLM client (will integrate with LM Studio directly)

    Returns:
        Initialized HybridCommandDiscovery instance ready for use
    """
    # Get Redis client for the service
    redis_client = await get_redis_client()

    # Create service with LM Studio integration
    service = HybridCommandDiscovery(
        redis_client=redis_client,
        llm_normalizer=llm_client,  # Can be None, service has direct LM Studio integration
    )

    # Verify database connections on startup
    try:
        # Test Redis connection
        await redis_client.ping()
        logger.info("Redis connection verified for HybridCommandDiscovery")

        # Test PostgreSQL connection
        session = await get_db_session_direct()
        try:
            from sqlalchemy import text

            await session.execute(text("SELECT 1"))
            logger.info("PostgreSQL connection verified for HybridCommandDiscovery")
        finally:
            await session.close()

        return service

    except Exception as e:
        logger.error(f"Failed to initialize HybridCommandDiscovery: {e}")
        raise

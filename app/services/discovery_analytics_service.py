# File: ~/net-chatbot/app/services/discovery_analytics_service.py
# Discovery Analytics Service - Three-Tier Performance Optimization and Intelligence

"""
Discovery Analytics Service - Three-Tier Performance Optimization

Tracks, analyzes, and optimizes the progressive performance evolution:
Week 1: 80% LLM usage → Month 6: 70% Redis + 20% PostgreSQL + 10% LLM

Provides intelligent cache promotion, TTL optimization, and performance insights.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

from sqlalchemy import and_, desc, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session_direct, get_redis_client
from app.models.command_mappings import CommandMapping, DiscoveryAnalytics

logger = logging.getLogger(__name__)


class DiscoveryAnalyticsService:
    """
    Analytics service for three-tier discovery performance optimization.

    Key Features:
    - Real-time performance tracking
    - Intelligent cache promotion recommendations
    - Progressive performance optimization
    - Performance evolution predictions
    - Cache efficiency optimization
    """

    def __init__(self):
        """Initialize analytics service."""
        self.daily_cache = {}  # In-memory cache for today's stats

    async def record_discovery_event(
        self,
        intent: str,
        platform: str,
        source: str,  # "redis_cache", "postgres_db", "llm_discovery"
        response_time_ms: float,
        confidence: float,
        success: bool = True,
    ):
        """
        Record individual discovery event for analytics.

        Called by HybridCommandDiscovery for every discovery request.
        """
        try:
            # Update in-memory daily stats
            today = datetime.now().date()
            if today not in self.daily_cache:
                self.daily_cache[today] = {
                    "total_requests": 0,
                    "redis_hits": 0,
                    "postgres_hits": 0,
                    "llm_discoveries": 0,
                    "redis_time_total": 0.0,
                    "postgres_time_total": 0.0,
                    "llm_time_total": 0.0,
                    "failed_requests": 0,
                }

            stats = self.daily_cache[today]
            stats["total_requests"] += 1

            if not success:
                stats["failed_requests"] += 1
                return

            # Update source-specific stats
            if source == "redis_cache":
                stats["redis_hits"] += 1
                stats["redis_time_total"] += response_time_ms
            elif source == "postgres_db":
                stats["postgres_hits"] += 1
                stats["postgres_time_total"] += response_time_ms
            elif source == "llm_discovery":
                stats["llm_discoveries"] += 1
                stats["llm_time_total"] += response_time_ms

            # Periodic flush to PostgreSQL (every 100 requests)
            if stats["total_requests"] % 100 == 0:
                await self._flush_daily_stats_to_db(today)

        except Exception as e:
            logger.error(f"Failed to record discovery event: {e}")

    async def _flush_daily_stats_to_db(self, date: datetime.date):
        """Flush in-memory daily stats to PostgreSQL for persistence."""
        session = None
        try:
            if date not in self.daily_cache:
                return

            stats = self.daily_cache[date]

            # Calculate performance metrics
            cache_hits = stats["redis_hits"] + stats["postgres_hits"]
            total_requests = stats["total_requests"]

            if total_requests == 0:
                return

            cache_hit_rate = (cache_hits / total_requests) * 100
            redis_hit_rate = (stats["redis_hits"] / total_requests) * 100
            postgres_hit_rate = (stats["postgres_hits"] / total_requests) * 100

            # Calculate average response times
            avg_redis_ms = (
                stats["redis_time_total"] / stats["redis_hits"]
                if stats["redis_hits"] > 0
                else 0
            )
            avg_postgres_ms = (
                stats["postgres_time_total"] / stats["postgres_hits"]
                if stats["postgres_hits"] > 0
                else 0
            )
            avg_llm_ms = (
                stats["llm_time_total"] / stats["llm_discoveries"]
                if stats["llm_discoveries"] > 0
                else 0
            )

            session = await get_db_session_direct()
            # Upsert daily analytics
            analytics_data = {
                "date": date,
                "total_requests": total_requests,
                "redis_hits": stats["redis_hits"],
                "postgres_hits": stats["postgres_hits"],
                "llm_discoveries": stats["llm_discoveries"],
                "avg_redis_response_ms": avg_redis_ms,
                "avg_postgres_response_ms": avg_postgres_ms,
                "avg_llm_response_ms": avg_llm_ms,
                "cache_hit_rate": cache_hit_rate,
                "redis_hit_rate": redis_hit_rate,
                "postgres_hit_rate": postgres_hit_rate,
            }

            stmt = insert(DiscoveryAnalytics).values(analytics_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=["date"],
                set_={
                    "total_requests": stmt.excluded.total_requests,
                    "redis_hits": stmt.excluded.redis_hits,
                    "postgres_hits": stmt.excluded.postgres_hits,
                    "llm_discoveries": stmt.excluded.llm_discoveries,
                    "avg_redis_response_ms": stmt.excluded.avg_redis_response_ms,
                    "avg_postgres_response_ms": stmt.excluded.avg_postgres_response_ms,
                    "avg_llm_response_ms": stmt.excluded.avg_llm_response_ms,
                    "cache_hit_rate": stmt.excluded.cache_hit_rate,
                    "redis_hit_rate": stmt.excluded.redis_hit_rate,
                    "postgres_hit_rate": stmt.excluded.postgres_hit_rate,
                },
            )

            await session.execute(stmt)
            await session.commit()

            logger.debug(
                f"Flushed daily stats for {date}: {cache_hit_rate:.1f}% cache hit rate"
            )

        except Exception as e:
            logger.error(f"Failed to flush daily stats to DB: {e}")
        finally:
            if session:
                await session.close()

    async def get_performance_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive performance dashboard for monitoring and optimization.

        Returns real-time and historical performance metrics.
        """
        session = None
        try:
            session = await get_db_session_direct()
            # Get last 30 days of analytics
            thirty_days_ago = datetime.now().date() - timedelta(days=30)

            query = (
                select(DiscoveryAnalytics)
                .where(DiscoveryAnalytics.date >= thirty_days_ago)
                .order_by(desc(DiscoveryAnalytics.date))
            )

            result = await session.execute(query)
            analytics = result.scalars().all()

            if not analytics:
                return {
                    "status": "no_data",
                    "message": "No analytics data available",
                }

            # Current performance (latest day)
            latest = analytics[0] if analytics else None

            # Performance trends (last 7 days vs previous 7 days)
            week_trends = await self._calculate_week_trends(session)

            # Progressive performance evolution
            evolution = await self._calculate_performance_evolution(analytics)

            # Cache optimization recommendations
            recommendations = await self._generate_optimization_recommendations(session)

            # Performance predictions
            predictions = await self._predict_future_performance(analytics)

            return {
                "status": "healthy",
                "current_performance": {
                    "cache_hit_rate": latest.cache_hit_rate if latest else 0,
                    "redis_hit_rate": latest.redis_hit_rate if latest else 0,
                    "postgres_hit_rate": latest.postgres_hit_rate if latest else 0,
                    "avg_response_time_ms": self._calculate_weighted_avg_response(
                        latest
                    ),
                    "total_requests_today": latest.total_requests if latest else 0,
                    "performance_tier": self._get_performance_tier(
                        latest.cache_hit_rate if latest else 0
                    ),
                },
                "weekly_trends": week_trends,
                "performance_evolution": evolution,
                "optimization_recommendations": recommendations,
                "predictions": predictions,
                "historical_data": [
                    {
                        "date": a.date.isoformat(),
                        "cache_hit_rate": a.cache_hit_rate,
                        "total_requests": a.total_requests,
                        "avg_response_time_ms": self._calculate_weighted_avg_response(
                            a
                        ),
                    }
                    for a in analytics[:14]  # Last 2 weeks
                ],
            }

        except Exception as e:
            logger.error(f"Failed to generate performance dashboard: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            if session:
                await session.close()

    async def _calculate_week_trends(self, session: AsyncSession) -> Dict[str, Any]:
        """Calculate week-over-week performance trends."""
        try:
            # Last 7 days
            week_ago = datetime.now().date() - timedelta(days=7)
            two_weeks_ago = datetime.now().date() - timedelta(days=14)

            # Current week stats
            current_week_query = select(
                func.avg(DiscoveryAnalytics.cache_hit_rate),
                func.avg(DiscoveryAnalytics.redis_hit_rate),
                func.sum(DiscoveryAnalytics.total_requests),
            ).where(DiscoveryAnalytics.date >= week_ago)

            current_result = await session.execute(current_week_query)
            current_stats = current_result.first()

            # Previous week stats
            previous_week_query = select(
                func.avg(DiscoveryAnalytics.cache_hit_rate),
                func.avg(DiscoveryAnalytics.redis_hit_rate),
                func.sum(DiscoveryAnalytics.total_requests),
            ).where(
                and_(
                    DiscoveryAnalytics.date >= two_weeks_ago,
                    DiscoveryAnalytics.date < week_ago,
                )
            )

            previous_result = await session.execute(previous_week_query)
            previous_stats = previous_result.first()

            # Calculate trends
            cache_trend = self._calculate_trend(
                current_stats[0] or 0, previous_stats[0] or 0
            )
            redis_trend = self._calculate_trend(
                current_stats[1] or 0, previous_stats[1] or 0
            )

            return {
                "cache_hit_rate_trend": cache_trend,
                "redis_hit_rate_trend": redis_trend,
                "total_requests_current_week": current_stats[2] or 0,
                "total_requests_previous_week": previous_stats[2] or 0,
                "request_volume_trend": self._calculate_trend(
                    current_stats[2] or 0, previous_stats[2] or 0
                ),
            }

        except Exception as e:
            logger.error(f"Failed to calculate week trends: {e}")
            return {}

    def _calculate_trend(self, current: float, previous: float) -> Dict[str, Any]:
        """Calculate trend percentage and direction."""
        if previous == 0:
            return {
                "change_percent": 0,
                "direction": "stable",
                "message": "No previous data",
            }

        change_percent = ((current - previous) / previous) * 100

        if change_percent > 5:
            direction = "improving"
        elif change_percent < -5:
            direction = "declining"
        else:
            direction = "stable"

        return {
            "change_percent": round(change_percent, 1),
            "direction": direction,
            "current_value": round(current, 1),
            "previous_value": round(previous, 1),
        }

    async def _calculate_performance_evolution(
        self, analytics: List[DiscoveryAnalytics]
    ) -> Dict[str, Any]:
        """Calculate performance evolution over time."""
        if len(analytics) < 7:
            return {"status": "insufficient_data"}

        # Group by weeks for evolution tracking
        weeks = {}
        for a in analytics:
            week_start = a.date - timedelta(days=a.date.weekday())
            week_key = week_start.isoformat()

            if week_key not in weeks:
                weeks[week_key] = []
            weeks[week_key].append(a)

        # Calculate weekly averages
        weekly_performance = []
        for week_key in sorted(weeks.keys(), reverse=True)[:12]:  # Last 12 weeks
            week_data = weeks[week_key]
            avg_cache_rate = sum(a.cache_hit_rate for a in week_data) / len(week_data)
            avg_requests = sum(a.total_requests for a in week_data) / len(week_data)

            weekly_performance.append(
                {
                    "week": week_key,
                    "cache_hit_rate": round(avg_cache_rate, 1),
                    "avg_requests_per_day": round(avg_requests, 0),
                    "performance_tier": self._get_performance_tier(avg_cache_rate),
                }
            )

        # Calculate evolution trajectory
        if len(weekly_performance) >= 4:
            first_month = weekly_performance[-4:]  # Oldest 4 weeks
            last_month = weekly_performance[:4]  # Newest 4 weeks

            first_avg = sum(w["cache_hit_rate"] for w in first_month) / len(first_month)
            last_avg = sum(w["cache_hit_rate"] for w in last_month) / len(last_month)

            evolution_rate = last_avg - first_avg

            return {
                "status": "tracking",
                "weekly_performance": weekly_performance,
                "evolution_summary": {
                    "cache_hit_rate_improvement": round(evolution_rate, 1),
                    "trajectory": "improving" if evolution_rate > 0 else "stable",
                    "weeks_tracked": len(weekly_performance),
                },
            }

        return {
            "status": "early_tracking",
            "weekly_performance": weekly_performance,
            "message": "More data needed for evolution analysis",
        }

    async def _generate_optimization_recommendations(
        self, session: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Generate intelligent cache optimization recommendations."""
        recommendations = []

        try:
            # Find high-usage, low-confidence commands for promotion
            high_usage_query = (
                select(CommandMapping)
                .where(
                    and_(
                        CommandMapping.usage_count > 5, CommandMapping.confidence < 0.9
                    )
                )
                .order_by(desc(CommandMapping.usage_count))
                .limit(10)
            )

            result = await session.execute(high_usage_query)
            high_usage_commands = result.scalars().all()

            if high_usage_commands:
                recommendations.append(
                    {
                        "type": "confidence_boost",
                        "priority": "high",
                        "title": "Boost High-Usage Command Confidence",
                        "description": f"Found {len(high_usage_commands)} frequently used commands with low confidence",
                        "action": "Review and manually boost confidence for better cache promotion",
                        "commands": [
                            f"{cmd.intent} ({cmd.usage_count} uses, {cmd.confidence} confidence)"
                            for cmd in high_usage_commands[:3]
                        ],
                    }
                )

            # Find commands that should be cached but aren't
            recent_llm_query = (
                select(CommandMapping)
                .where(
                    and_(
                        CommandMapping.source == "llm_discovery",
                        CommandMapping.usage_count > 3,
                        CommandMapping.last_used
                        > datetime.utcnow() - timedelta(days=7),
                    )
                )
                .limit(5)
            )

            result = await session.execute(recent_llm_query)
            recent_llm_commands = result.scalars().all()

            if recent_llm_commands:
                recommendations.append(
                    {
                        "type": "cache_promotion",
                        "priority": "medium",
                        "title": "Promote Recent LLM Discoveries",
                        "description": f"Found {len(recent_llm_commands)} recent LLM discoveries with repeat usage",
                        "action": "These should be automatically promoted to cache",
                        "commands": [
                            f"{cmd.intent} (used {cmd.usage_count} times this week)"
                            for cmd in recent_llm_commands[:3]
                        ],
                    }
                )

            # Check for cache efficiency opportunities
            redis_client = await get_redis_client()
            cache_info = await redis_client.info("memory")
            used_memory_mb = cache_info.get("used_memory", 0) / 1024 / 1024

            if used_memory_mb < 256:  # Less than 50% of 512MB
                recommendations.append(
                    {
                        "type": "cache_expansion",
                        "priority": "low",
                        "title": "Cache Underutilized",
                        "description": f"Only using {used_memory_mb:.1f}MB of 512MB Redis cache",
                        "action": "Consider caching more commands or extending TTL for high-usage items",
                    }
                )

            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return [
                {
                    "type": "error",
                    "title": "Recommendation Generation Failed",
                    "description": str(e),
                }
            ]

    async def _predict_future_performance(
        self, analytics: List[DiscoveryAnalytics]
    ) -> Dict[str, Any]:
        """Predict future performance based on current trends."""
        if len(analytics) < 14:
            return {"status": "insufficient_data"}

        try:
            # Simple linear trend analysis
            recent_data = analytics[:14]  # Last 2 weeks

            # Calculate trend slope for cache hit rate
            x_values = list(range(len(recent_data)))
            cache_rates = [a.cache_hit_rate for a in reversed(recent_data)]

            # Simple linear regression
            n = len(cache_rates)
            sum_x = sum(x_values)
            sum_y = sum(cache_rates)
            sum_xy = sum(x * y for x, y in zip(x_values, cache_rates))
            sum_x2 = sum(x * x for x in x_values)

            # Calculate slope (trend)
            if n * sum_x2 - sum_x * sum_x != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                intercept = (sum_y - slope * sum_x) / n

                # Predict 30 days out
                future_x = len(recent_data) + 30
                predicted_cache_rate = slope * future_x + intercept

                # Bound predictions to realistic range
                predicted_cache_rate = max(0, min(100, predicted_cache_rate))

                # Predict response time improvement
                current_avg_cache_rate = sum(cache_rates[-7:]) / 7  # Last week average
                response_time_improvement = (
                    predicted_cache_rate - current_avg_cache_rate
                ) * 0.1

                return {
                    "status": "predicted",
                    "next_month": {
                        "predicted_cache_hit_rate": round(predicted_cache_rate, 1),
                        "predicted_response_time_improvement_percent": round(
                            response_time_improvement, 1
                        ),
                        "confidence": "medium" if abs(slope) > 0.1 else "low",
                    },
                    "trend_analysis": {
                        "daily_improvement_rate": round(slope, 2),
                        "trend_direction": "improving"
                        if slope > 0
                        else "declining"
                        if slope < 0
                        else "stable",
                    },
                }

            return {"status": "stable_trend"}

        except Exception as e:
            logger.error(f"Failed to predict future performance: {e}")
            return {"status": "prediction_error", "error": str(e)}

    def _calculate_weighted_avg_response(self, analytics: DiscoveryAnalytics) -> float:
        """Calculate weighted average response time across all tiers."""
        if not analytics or analytics.total_requests == 0:
            return 0.0

        total_time = (
            analytics.redis_hits * analytics.avg_redis_response_ms
            + analytics.postgres_hits * analytics.avg_postgres_response_ms
            + analytics.llm_discoveries * analytics.avg_llm_response_ms
        )

        total_requests = (
            analytics.redis_hits + analytics.postgres_hits + analytics.llm_discoveries
        )

        return round(total_time / total_requests, 1) if total_requests > 0 else 0.0

    def _get_performance_tier(self, cache_hit_rate: float) -> str:
        """Classify performance tier based on cache hit rate."""
        if cache_hit_rate >= 90:
            return "Ultra-Optimized"
        elif cache_hit_rate >= 70:
            return "Optimized"
        elif cache_hit_rate >= 50:
            return "Improving"
        elif cache_hit_rate >= 30:
            return "Learning"
        else:
            return "Initial"

    async def get_top_commands_analysis(self, limit: int = 20) -> Dict[str, Any]:
        """Analyze top commands for optimization opportunities."""
        session = None
        try:
            session = await get_db_session_direct()
            # Top commands by usage
            top_usage_query = (
                select(CommandMapping)
                .order_by(desc(CommandMapping.usage_count))
                .limit(limit)
            )

            result = await session.execute(top_usage_query)
            top_commands = result.scalars().all()

            # Analyze cache efficiency
            cache_analysis = []
            for cmd in top_commands:
                efficiency_score = self._calculate_cache_efficiency(cmd)
                cache_analysis.append(
                    {
                        "intent": cmd.intent,
                        "platform": cmd.platform,
                        "usage_count": cmd.usage_count,
                        "confidence": float(cmd.confidence),
                        "source": cmd.source,
                        "cache_efficiency_score": efficiency_score,
                        "optimization_potential": self._get_optimization_potential(cmd),
                    }
                )

            return {
                "status": "success",
                "top_commands": cache_analysis,
                "summary": {
                    "total_analyzed": len(cache_analysis),
                    "high_efficiency": len(
                        [c for c in cache_analysis if c["cache_efficiency_score"] > 80]
                    ),
                    "optimization_candidates": len(
                        [
                            c
                            for c in cache_analysis
                            if c["optimization_potential"] != "optimal"
                        ]
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Failed to analyze top commands: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            if session:
                await session.close()

    def _calculate_cache_efficiency(self, cmd: CommandMapping) -> int:
        """Calculate cache efficiency score for a command (0-100)."""
        score = 0

        # Usage frequency (40 points max)
        if cmd.usage_count > 50:
            score += 40
        elif cmd.usage_count > 20:
            score += 30
        elif cmd.usage_count > 10:
            score += 20
        elif cmd.usage_count > 5:
            score += 10

        # Confidence level (30 points max)
        confidence_points = int(float(cmd.confidence) * 30)
        score += confidence_points

        # Recency (20 points max)
        if cmd.last_used:
            days_since_use = (datetime.utcnow() - cmd.last_used).days
            if days_since_use <= 1:
                score += 20
            elif days_since_use <= 7:
                score += 15
            elif days_since_use <= 30:
                score += 10
            elif days_since_use <= 90:
                score += 5

        # Source bonus (10 points max)
        if cmd.source == "llm_discovery":
            score += 10  # LLM discoveries are valuable
        elif cmd.source == "community":
            score += 8  # Community mappings are reliable
        elif cmd.source == "manual":
            score += 10  # Manual entries are high quality

        return min(score, 100)

    def _get_optimization_potential(self, cmd: CommandMapping) -> str:
        """Determine optimization potential for a command."""
        efficiency = self._calculate_cache_efficiency(cmd)

        if efficiency >= 80:
            return "optimal"
        elif efficiency >= 60:
            return "good"
        elif efficiency >= 40:
            return "moderate"
        elif efficiency >= 20:
            return "low"
        else:
            return "poor"


# Factory function for easy service initialization
async def create_discovery_analytics_service() -> DiscoveryAnalyticsService:
    """
    Create and initialize Discovery Analytics Service.

    Returns:
        Initialized analytics service ready for use
    """
    service = DiscoveryAnalyticsService()

    # Verify database connectivity
    session = None
    try:
        session = await get_db_session_direct()
        await session.execute(select(1))
        logger.info("Discovery Analytics Service initialized successfully")
        return service

    except Exception as e:
        logger.error(f"Failed to initialize Discovery Analytics Service: {e}")
        raise
    finally:
        if session:
            await session.close()


# Background task for periodic optimization
async def run_periodic_optimization():
    """
    Background task to run periodic cache optimization.

    This should be scheduled to run every few hours in production.
    """
    try:
        service = await create_discovery_analytics_service()

        # Flush any pending daily stats
        today = datetime.now().date()
        await service._flush_daily_stats_to_db(today)

        logger.info("Periodic optimization completed successfully")

    except Exception as e:
        logger.error(f"Periodic optimization task failed: {e}")


# FastAPI endpoint integration helpers
async def get_analytics_dashboard_data() -> Dict[str, Any]:
    """Get analytics dashboard data for FastAPI endpoint."""
    try:
        service = await create_discovery_analytics_service()
        return await service.get_performance_dashboard()
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def get_performance_report_data() -> Dict[str, Any]:
    """Get performance report data for FastAPI endpoint."""
    try:
        service = await create_discovery_analytics_service()

        # Generate executive report based on dashboard data
        dashboard = await service.get_performance_dashboard()

        if dashboard.get("status") == "error":
            return dashboard

        current_perf = dashboard.get("current_performance", {})
        cache_hit_rate = current_perf.get("cache_hit_rate", 0)
        avg_response_time = current_perf.get("avg_response_time_ms", 0)

        # Calculate performance improvement
        baseline_response_time = 3000  # 3s baseline with pure LLM
        improvement_percent = (
            (baseline_response_time - avg_response_time) / baseline_response_time
        ) * 100

        return {
            "executive_summary": {
                "current_cache_hit_rate": f"{cache_hit_rate:.1f}%",
                "performance_improvement": f"{improvement_percent:.1f}%",
                "avg_response_time": f"{avg_response_time:.0f}ms",
                "performance_tier": current_perf.get("performance_tier", "Unknown"),
                "total_requests_analyzed": current_perf.get("total_requests_today", 0),
            },
            "key_metrics": {
                "three_tier_efficiency": {
                    "redis_hit_rate": f"{current_perf.get('redis_hit_rate', 0):.1f}%",
                    "postgres_hit_rate": f"{current_perf.get('postgres_hit_rate', 0):.1f}%",
                    "llm_discovery_rate": f"{100 - cache_hit_rate:.1f}%",
                },
                "performance_evolution": dashboard.get("performance_evolution", {}),
                "optimization_impact": dashboard.get(
                    "optimization_recommendations", []
                ),
            },
            "predictions": dashboard.get("predictions", {}),
            "generated_at": datetime.utcnow().isoformat(),
            "report_period": "Last 30 days",
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}

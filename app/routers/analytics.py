# File: ~/net-chatbot/app/routers/analytics.py
# Analytics FastAPI Router - Three-Tier Discovery Performance Endpoints

"""
Analytics FastAPI Router - Three-Tier Discovery Performance Endpoints

Provides real-time monitoring, optimization insights, and performance analytics
for the three-tier discovery system.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

from app.services.discovery_analytics_service import (
    create_discovery_analytics_service,
    get_analytics_dashboard_data,
    get_performance_report_data,
    run_periodic_optimization,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/analytics", tags=["Three-Tier Analytics"])


# Response models for API documentation
class PerformanceStats(BaseModel):
    """Current performance statistics."""

    cache_hit_rate: float
    redis_hit_rate: float
    postgres_hit_rate: float
    avg_response_time_ms: float
    total_requests_today: int
    performance_tier: str


class OptimizationRecommendation(BaseModel):
    """Cache optimization recommendation."""

    type: str
    priority: str
    title: str
    description: str
    action: Optional[str] = None


@router.get("/dashboard", response_model=Dict[str, Any])
async def get_performance_dashboard():
    """
    Get comprehensive three-tier discovery performance dashboard.

    Returns real-time performance metrics, trends, and optimization insights.
    """
    try:
        dashboard_data = await get_analytics_dashboard_data()

        if dashboard_data.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Analytics service error: {dashboard_data.get('error')}",
            )

        return {
            "success": True,
            "data": dashboard_data,
            "generated_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Dashboard endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/performance/current")
async def get_current_performance():
    """
    Get current three-tier discovery performance metrics.

    Returns:
        - Cache hit rates by tier (Redis, PostgreSQL, LLM)
        - Average response times
        - Performance classification
        - Request volume statistics
    """
    try:
        dashboard_data = await get_analytics_dashboard_data()

        if dashboard_data.get("status") == "error":
            raise HTTPException(status_code=500, detail=dashboard_data.get("error"))

        current_perf = dashboard_data.get("current_performance", {})

        return {
            "three_tier_performance": {
                "overall_cache_hit_rate": current_perf.get("cache_hit_rate", 0),
                "tier_breakdown": {
                    "redis_cache_hits": current_perf.get("redis_hit_rate", 0),
                    "postgres_db_hits": current_perf.get("postgres_hit_rate", 0),
                    "llm_discoveries": 100 - current_perf.get("cache_hit_rate", 0),
                },
                "performance_metrics": {
                    "avg_response_time_ms": current_perf.get("avg_response_time_ms", 0),
                    "performance_tier": current_perf.get("performance_tier", "Unknown"),
                    "total_requests_today": current_perf.get("total_requests_today", 0),
                },
            },
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Current performance endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/trends/weekly")
async def get_weekly_trends():
    """
    Get week-over-week performance trends.

    Shows improvement trajectory and performance evolution.
    """
    try:
        dashboard_data = await get_analytics_dashboard_data()

        if dashboard_data.get("status") == "error":
            raise HTTPException(status_code=500, detail=dashboard_data.get("error"))

        trends = dashboard_data.get("weekly_trends", {})
        evolution = dashboard_data.get("performance_evolution", {})

        return {
            "weekly_comparison": trends,
            "performance_evolution": evolution,
            "insights": {
                "cache_trend_direction": trends.get("cache_hit_rate_trend", {}).get(
                    "direction", "unknown"
                ),
                "request_volume_change": trends.get("request_volume_trend", {}).get(
                    "change_percent", 0
                ),
                "evolution_status": evolution.get("status", "tracking"),
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Weekly trends endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/optimization/recommendations")
async def get_optimization_recommendations():
    """
    Get intelligent cache optimization recommendations.

    Provides actionable insights for improving three-tier performance.
    """
    try:
        dashboard_data = await get_analytics_dashboard_data()

        if dashboard_data.get("status") == "error":
            raise HTTPException(status_code=500, detail=dashboard_data.get("error"))

        recommendations = dashboard_data.get("optimization_recommendations", [])

        # Categorize recommendations by priority
        categorized = {
            "high_priority": [
                r for r in recommendations if r.get("priority") == "high"
            ],
            "medium_priority": [
                r for r in recommendations if r.get("priority") == "medium"
            ],
            "low_priority": [r for r in recommendations if r.get("priority") == "low"],
        }

        return {
            "recommendations": categorized,
            "summary": {
                "total_recommendations": len(recommendations),
                "high_priority_count": len(categorized["high_priority"]),
                "actionable_optimizations": len(
                    [r for r in recommendations if "action" in r]
                ),
            },
            "next_review_suggested": (
                datetime.utcnow() + timedelta(days=7)
            ).isoformat(),
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Optimization recommendations endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/commands/top-performers")
async def get_top_performing_commands(limit: int = Query(20, ge=1, le=100)):
    """
    Get analysis of top-performing commands for optimization insights.

    Args:
        limit: Number of top commands to analyze (1-100)
    """
    try:
        service = await create_discovery_analytics_service()
        analysis = await service.get_top_commands_analysis(limit=limit)

        if analysis.get("status") == "error":
            raise HTTPException(status_code=500, detail=analysis.get("error"))

        # Enhance response with optimization insights
        top_commands = analysis.get("top_commands", [])

        # Group by optimization potential
        by_potential = {}
        for cmd in top_commands:
            potential = cmd.get("optimization_potential", "unknown")
            if potential not in by_potential:
                by_potential[potential] = []
            by_potential[potential].append(cmd)

        return {
            "top_commands": top_commands,
            "optimization_analysis": {
                "by_potential": by_potential,
                "summary": analysis.get("summary", {}),
                "recommendations": [
                    "Focus on 'moderate' and 'low' potential commands for biggest impact",
                    "Consider manual confidence boosting for high-usage, low-confidence commands",
                    "Monitor 'poor' efficiency commands for possible removal or improvement",
                ],
            },
            "analyzed_count": len(top_commands),
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Top performers endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/optimization/auto-optimize")
async def trigger_automatic_optimization(background_tasks: BackgroundTasks):
    """
    Trigger automatic cache optimization.

    Analyzes usage patterns and automatically promotes high-value commands
    to appropriate cache tiers with intelligent TTL settings.
    """
    try:
        # Run optimization in background to avoid timeout
        background_tasks.add_task(run_periodic_optimization)

        return {
            "status": "optimization_started",
            "message": "Automatic cache optimization has been triggered",
            "expected_completion": "Background optimization will complete within 5 minutes",
            "check_status_endpoint": "/api/v1/analytics/health",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Auto-optimization trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/predictions/performance")
async def get_performance_predictions():
    """
    Get performance predictions based on current trends.

    Provides forecasts for cache hit rates and response time improvements.
    """
    try:
        dashboard_data = await get_analytics_dashboard_data()

        if dashboard_data.get("status") == "error":
            raise HTTPException(status_code=500, detail=dashboard_data.get("error"))

        predictions = dashboard_data.get("predictions", {})

        if predictions.get("status") == "insufficient_data":
            return {
                "status": "insufficient_data",
                "message": "More historical data needed for accurate predictions",
                "minimum_days_required": 14,
                "timestamp": datetime.utcnow().isoformat(),
            }

        return {
            "predictions": predictions,
            "confidence_level": predictions.get("next_month", {}).get(
                "confidence", "unknown"
            ),
            "methodology": "Linear trend analysis based on last 14 days of performance data",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Performance predictions endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/report/executive")
async def get_executive_performance_report():
    """
    Get comprehensive executive performance report.

    Provides high-level summary suitable for management review.
    """
    try:
        report_data = await get_performance_report_data()

        if report_data.get("status") == "error":
            raise HTTPException(status_code=500, detail=report_data.get("error"))

        return {
            "executive_report": report_data,
            "report_type": "Three-Tier Discovery Performance Analysis",
            "generated_for": "Management Review",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Executive report endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/health")
async def get_analytics_health():
    """
    Get health status of analytics and three-tier discovery system.

    Checks database connections, cache status, and service availability.
    """
    try:
        health_status = {
            "analytics_service": "healthy",
            "database_connections": {},
            "cache_status": {},
            "last_health_check": datetime.utcnow().isoformat(),
        }

        # Test database connectivity - FIXED
        try:
            from sqlalchemy import text

            from app.core.database import get_db_session_direct, get_redis_client

            # Test PostgreSQL - CORRECTED
            session = await get_db_session_direct()
            try:
                await session.execute(text("SELECT 1"))
                health_status["database_connections"]["postgresql"] = "healthy"
            finally:
                await session.close()

            # Test Redis
            redis_client = await get_redis_client()
            await redis_client.ping()
            health_status["database_connections"]["redis"] = "healthy"

            # Get Redis memory info
            redis_info = await redis_client.info("memory")
            used_memory_mb = redis_info.get("used_memory", 0) / 1024 / 1024
            max_memory_mb = redis_info.get("maxmemory", 512 * 1024 * 1024) / 1024 / 1024

            health_status["cache_status"] = {
                "redis_memory_used_mb": round(used_memory_mb, 1),
                "redis_memory_max_mb": round(max_memory_mb, 1),
                "redis_memory_utilization_percent": round(
                    (used_memory_mb / max_memory_mb) * 100, 1
                ),
            }

        except Exception as db_error:
            health_status["database_connections"]["error"] = str(db_error)
            health_status["analytics_service"] = "degraded"

        # Overall health determination
        if health_status["analytics_service"] == "healthy":
            return health_status
        else:
            raise HTTPException(status_code=503, detail=health_status)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check endpoint failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}") from e


@router.post("/maintenance/cleanup")
async def trigger_cache_cleanup(background_tasks: BackgroundTasks):
    """
    Trigger cache cleanup and maintenance tasks.

    Cleans expired entries, optimizes memory usage, and updates statistics.
    """
    try:
        # Add cleanup task to background
        background_tasks.add_task(run_periodic_optimization)

        return {
            "status": "cleanup_started",
            "message": "Cache cleanup and maintenance tasks have been triggered",
            "tasks": [
                "Expired cache entry removal",
                "Memory optimization",
                "Statistics update",
                "Performance optimization",
            ],
            "estimated_duration": "2-5 minutes",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Cache cleanup trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Historical data endpoints
@router.get("/history/daily")
async def get_daily_performance_history(
    days: int = Query(30, ge=1, le=90), include_raw_data: bool = Query(False)
):
    """
    Get daily performance history for trend analysis.

    Args:
        days: Number of days to retrieve (1-90)
        include_raw_data: Include raw analytics data in response
    """
    try:
        dashboard_data = await get_analytics_dashboard_data()

        if dashboard_data.get("status") == "error":
            raise HTTPException(status_code=500, detail=dashboard_data.get("error"))

        historical_data = dashboard_data.get("historical_data", [])

        # Limit to requested number of days
        limited_data = historical_data[:days]

        # Calculate summary statistics
        if limited_data:
            avg_cache_rate = sum(d["cache_hit_rate"] for d in limited_data) / len(
                limited_data
            )
            avg_response_time = sum(
                d["avg_response_time_ms"] for d in limited_data
            ) / len(limited_data)
            total_requests = sum(d["total_requests"] for d in limited_data)
        else:
            avg_cache_rate = avg_response_time = total_requests = 0

        response = {
            "daily_performance": limited_data,
            "period_summary": {
                "days_analyzed": len(limited_data),
                "avg_cache_hit_rate": round(avg_cache_rate, 1),
                "avg_response_time_ms": round(avg_response_time, 1),
                "total_requests_period": total_requests,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        if include_raw_data:
            response["raw_analytics_data"] = dashboard_data

        return response

    except Exception as e:
        logger.error(f"Daily history endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

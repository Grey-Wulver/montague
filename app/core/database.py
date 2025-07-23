# ~/net-chatbot/app/core/database.py
"""
Database Core Module - PostgreSQL + Redis Support
Replaces the non-existent database.py with production-ready async database support
"""

import asyncio
import logging
from collections.abc import AsyncGenerator

import redis.asyncio as redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base

from app.core.config import settings

logger = logging.getLogger(__name__)

# SQLAlchemy Base for ORM models
Base = declarative_base()

# Global connection objects
async_engine = None
async_session_factory = None
redis_client = None


async def init_database():
    """Initialize PostgreSQL + Redis connections"""
    global async_engine, async_session_factory, redis_client

    logger.info("🔄 Initializing PostgreSQL + Redis connections...")

    try:
        # PostgreSQL Setup
        if "sqlite" in settings.database_url.lower():
            # Convert SQLite URL to async PostgreSQL for production
            postgres_url = (
                "postgresql+asyncpg://netops:netops_password@localhost:5432/netops_db"
            )
            logger.warning(f"Converting SQLite to PostgreSQL: {postgres_url}")
        else:
            postgres_url = settings.database_url

        # Create async engine
        async_engine = create_async_engine(
            postgres_url,
            echo=settings.debug,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600,  # 1 hour
        )

        # Create session factory
        async_session_factory = async_sessionmaker(
            async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        logger.info("✅ PostgreSQL async engine initialized")

        # Redis Setup
        redis_client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
        )

        # Test Redis connection
        await redis_client.ping()
        logger.info("✅ Redis connection established")

        # Create tables if they don't exist
        await create_tables()

        logger.info("🚀 Database initialization complete!")
        return True

    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False


async def create_tables():
    """Create database tables if they don't exist"""
    try:
        async with async_engine.begin() as conn:
            # Import models to ensure they're registered

            # Create all tables
            await conn.run_sync(Base.metadata.create_all)

        logger.info("✅ Database tables created/verified")

    except Exception as e:
        logger.error(f"❌ Table creation failed: {e}")
        raise


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session - for dependency injection"""
    if not async_session_factory:
        await init_database()

    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db_session_direct() -> AsyncSession:
    """Get async database session - for direct use in services"""
    if not async_session_factory:
        await init_database()

    return async_session_factory()


async def get_redis_client() -> redis.Redis:
    """Get Redis client"""
    if not redis_client:
        await init_database()
    return redis_client


async def health_check() -> dict:
    """Check health of database connections"""
    health = {
        "postgresql": "unknown",
        "redis": "unknown",
    }

    try:
        # Test PostgreSQL
        if async_engine:
            async with async_engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                health["postgresql"] = (
                    "healthy" if result.scalar() == 1 else "unhealthy"
                )
        else:
            health["postgresql"] = "not_initialized"
    except Exception as e:
        health["postgresql"] = f"error: {str(e)}"

    try:
        # Test Redis
        if redis_client:
            await redis_client.ping()
            health["redis"] = "healthy"
        else:
            health["redis"] = "not_initialized"
    except Exception as e:
        health["redis"] = f"error: {str(e)}"

    return health


async def close_connections():
    """Close all database connections"""
    global async_engine, redis_client

    try:
        if async_engine:
            await async_engine.dispose()
            logger.info("✅ PostgreSQL connections closed")

        if redis_client:
            await redis_client.close()
            logger.info("✅ Redis connections closed")

    except Exception as e:
        logger.error(f"Error closing connections: {e}")


# Utility functions for three-tier system
async def execute_with_retry(func, max_retries: int = 3):
    """Execute database function with retry logic"""
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Database operation failed (attempt {attempt + 1}): {e}")
            await asyncio.sleep(0.5 * (attempt + 1))


class DatabaseManager:
    """High-level database manager for the three-tier system"""

    def __init__(self):
        self.postgres_engine = None
        self.redis_client = None

    async def initialize(self):
        """Initialize database manager"""
        return await init_database()

    async def get_postgres_session(self) -> AsyncSession:
        """Get PostgreSQL session"""
        return await get_db_session_direct()

    async def get_redis(self) -> redis.Redis:
        """Get Redis client"""
        return await get_redis_client()

    async def health_status(self) -> dict:
        """Get comprehensive health status"""
        db_health = await health_check()

        # Add performance metrics
        try:
            redis_info = await redis_client.info() if redis_client else {}
            db_health["redis_memory"] = redis_info.get("used_memory_human", "unknown")
            db_health["redis_connected_clients"] = redis_info.get(
                "connected_clients", 0
            )
        except Exception:
            pass

        return db_health


# Global database manager
db_manager = DatabaseManager()

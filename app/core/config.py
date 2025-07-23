# ~/net-chatbot/app/core/config.py
"""
Enhanced Configuration for Three-Tier Discovery System
Supports PostgreSQL + Redis + comprehensive three-tier settings
"""

from typing import Optional

from pydantic import BaseSettings


class Settings(BaseSettings):
    """Enhanced settings for three-tier discovery system"""

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    environment: str = "development"
    log_level: str = "INFO"

    # Database Configuration - THREE-TIER SYSTEM
    database_url: str = (
        "postgresql+asyncpg://netops:netops_password@localhost:5432/netops_db"
    )

    # Database Connection Pooling
    db_pool_size: int = 20
    db_max_overflow: int = 30
    db_pool_recycle: int = 3600

    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl: int = 3600
    redis_max_connections: int = 20

    # Three-Tier Discovery Configuration
    discovery_redis_ttl: int = 3600
    discovery_postgres_timeout: int = 20
    discovery_llm_timeout: int = 30
    discovery_confidence_threshold: float = 0.8

    # Community Sync Configuration
    community_sync_enabled: bool = True
    community_sync_interval_hours: int = 24
    community_repositories: str = "ntc-templates,genie-parsers,napalm-getters"

    # Performance Monitoring
    analytics_enabled: bool = True
    analytics_retention_days: int = 90
    cache_analytics_enabled: bool = True

    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # Network Automation
    default_timeout: int = 30
    max_concurrent_devices: int = 10

    # LLM Configuration - PRESERVING YOUR WORKING SETUP
    ollama_url: str = "http://192.168.1.11:1234"
    ollama_model: str = "meta-llama-3.1-8b-instruct"
    llm_enabled: bool = True
    llm_timeout: int = 30
    llm_max_retries: int = 2
    normalization_cache_size: int = 1000

    # OpenAI (optional fallback)
    openai_api_key: Optional[str] = None

    # Chatbot Integration - PRESERVING YOUR WORKING SETUP
    mattermost_url: str = "http://192.168.1.100:8065"
    mattermost_bot_token: str = "your-bot-token-here"

    # Performance Tuning
    max_redis_memory_mb: int = 512
    postgres_query_timeout: int = 30
    llm_discovery_batch_size: int = 5

    class Config:
        env_file = ".env"
        case_sensitive = False

    # Helper properties
    @property
    def is_development(self) -> bool:
        return self.environment.lower() in ["development", "dev"]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in ["production", "prod"]

    @property
    def postgres_url_sync(self) -> str:
        """Convert async URL to sync for migrations"""
        return self.database_url.replace("+asyncpg", "")

    @property
    def community_repositories_list(self) -> list:
        """Parse community repositories into list"""
        return [repo.strip() for repo in self.community_repositories.split(",")]


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get settings instance"""
    return settings


def get_database_config() -> dict:
    """Get database configuration for three-tier system"""
    return {
        "postgres": {
            "url": settings.database_url,
            "pool_size": settings.db_pool_size,
            "max_overflow": settings.db_max_overflow,
            "pool_recycle": settings.db_pool_recycle,
        },
        "redis": {
            "url": settings.redis_url,
            "default_ttl": settings.redis_cache_ttl,
            "max_connections": settings.redis_max_connections,
        },
        "discovery": {
            "redis_ttl": settings.discovery_redis_ttl,
            "postgres_timeout": settings.discovery_postgres_timeout,
            "llm_timeout": settings.discovery_llm_timeout,
            "confidence_threshold": settings.discovery_confidence_threshold,
        },
    }


def get_three_tier_config() -> dict:
    """Get comprehensive three-tier configuration"""
    return {
        "performance_targets": {
            "redis_hit_rate_target": 70,  # 70% by month 6
            "postgres_hit_rate_target": 20,  # 20% by month 6
            "llm_usage_target": 10,  # 10% by month 6
            "response_time_target_ms": 800,  # 0.8s average by month 6
        },
        "cache_policies": {
            "redis_ttl_hot_commands": 7200,  # 2 hours for frequent commands
            "redis_ttl_normal_commands": 3600,  # 1 hour for normal commands
            "redis_ttl_cold_commands": 1800,  # 30 min for rare commands
            "postgres_confidence_threshold": settings.discovery_confidence_threshold,
        },
        "analytics": {
            "enabled": settings.analytics_enabled,
            "retention_days": settings.analytics_retention_days,
            "batch_size": 100,
            "real_time_updates": True,
        },
        "community_sync": {
            "enabled": settings.community_sync_enabled,
            "interval_hours": settings.community_sync_interval_hours,
            "repositories": settings.community_repositories_list,
            "auto_update_confidence": 0.9,
        },
    }

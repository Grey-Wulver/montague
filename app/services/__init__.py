# app/services/__init__.py
"""
Services module for NetOps ChatBot - Universal Pipeline Edition
Permanent migration with proper fallback handling
"""

# Core working services (keeping these - they work)
# Keep community_mapper (it's working and useful)
from .community_mapper import CommunityCommandMapper, community_mapper
from .executor import NetworkExecutor, network_executor
from .inventory import InventoryService, inventory_service
from .local_llm_normalizer import LocalLLMNormalizer, local_llm_normalizer
from .mattermost_client import MattermostClient

# enhanced_output_normalizer - handled in main.py with fallback
# Don't import here to avoid import errors

# Dynamic discovery (working)
try:
    from .dynamic_command_discovery import DynamicCommandDiscovery
    # Note: dynamic_discovery instance is created in main.py
except ImportError:
    pass

# Universal Pipeline services (new architecture)
try:
    from .intent_parser import IntentParser, intent_parser
    from .universal_formatter import UniversalFormatter, universal_formatter
    from .universal_request_processor import (
        UniversalRequestProcessor,
        universal_processor,
    )
except ImportError:
    # These will be available once all files are created
    pass

__all__ = [
    # Core services (keeping these)
    "InventoryService",
    "inventory_service",
    "NetworkExecutor",
    "network_executor",
    "MattermostClient",
    "LocalLLMNormalizer",
    "local_llm_normalizer",
    "CommunityCommandMapper",
    "community_mapper",
    # Dynamic discovery
    "DynamicCommandDiscovery",
    # Universal Pipeline (new)
    "UniversalRequestProcessor",
    "universal_processor",
    "UniversalFormatter",
    "universal_formatter",
    "IntentParser",
    "intent_parser",
]

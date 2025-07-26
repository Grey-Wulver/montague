# app/services/universal_request_processor.py
"""
Universal Request Processor - Enhanced with RAG Command Validation (Phase 6)

FIXED ARCHITECTURE FLOW:
User Input → LLM Intent Parser → Three-Tier Discovery →
                                    ↓
[NEW] RAG Command Validation → Command Execution → Universal Formatter → Response

Key Enhancements:
- RAG-powered command validation before execution
- Smart command suggestions for invalid commands
- Interactive correction workflow
- Learning integration with existing PostgreSQL
- Graceful fallback when RAG unavailable
- FIXED: Proper imports matching your project structure

Maintains VS Code + Ruff + Black standards with 88-character line length.
"""

import logging
import time
from typing import Any, Dict, List, Tuple

from app.models.universal_requests import (
    CommandSuggestion,
    CommandValidationInfo,
    DeviceExecutionResult,
    DiscoveryInfo,
    UniversalRequest,
    UniversalResponse,
    create_command_suggestion,
    filter_high_confidence_suggestions,
    should_auto_apply_suggestion,
    validate_request_size,
)

# FIXED: Proper imports using your existing project structure
from app.services.hybrid_command_discovery import create_hybrid_discovery_service
from app.services.rag_command_validation import get_rag_validation_service

logger = logging.getLogger(__name__)


class UniversalRequestProcessor:
    """
    ENHANCED Universal Request Processor with RAG Command Validation

    NEW Phase 6 Features:
    - RAG-powered command validation using Barrow service
    - Smart command suggestions for invalid commands
    - Interactive correction workflow
    - Learning integration for improved future suggestions
    - Enterprise-grade fallback when RAG unavailable

    PRESERVED Features:
    - Pure LLM understanding for intent parsing
    - Three-tier discovery system (Redis → PostgreSQL → LLM)
    - Universal interface support (Chat/API/CLI/Web)
    - Enterprise anti-hallucination protection
    - Performance optimization and analytics
    """

    def __init__(self):
        """Initialize the enhanced universal request processor."""
        # Core services (ENHANCED with RAG validation)
        self._hybrid_discovery = None
        self._rag_validation = None
        self._network_executor = None
        self._inventory_service = None
        self._intent_parser = None
        self._universal_formatter = None

        # Performance tracking (ENHANCED with validation metrics)
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_discoveries": 0,
            "redis_hits": 0,
            "postgres_hits": 0,
            "llm_discoveries": 0,
            "rag_validations": 0,
            "suggestions_generated": 0,
            "suggestions_applied": 0,
            "avg_request_time": 0.0,
            "avg_discovery_time": 0.0,
            "avg_execution_time": 0.0,
            "avg_rag_validation_time": 0.0,
        }

    async def initialize(self) -> bool:
        """Initialize all services for the universal processor."""
        try:
            # FIXED: Import using your exact project structure
            from app.services.executor import network_executor
            from app.services.inventory import inventory_service
            from app.services.pure_llm_intent_parser import (
                create_enhanced_intent_parser,
            )
            from app.services.universal_formatter import universal_formatter

            # Initialize existing services
            self._hybrid_discovery = await create_hybrid_discovery_service()
            self._network_executor = network_executor
            self._inventory_service = inventory_service
            self._intent_parser = create_enhanced_intent_parser()
            self._universal_formatter = universal_formatter

            # NEW: Initialize RAG validation service
            try:
                self._rag_validation = await get_rag_validation_service()
                if self._rag_validation:
                    logger.info("✅ RAG validation service initialized")
                else:
                    logger.warning(
                        "⚠️ RAG validation service not available - will use fallback"
                    )
            except Exception as e:
                logger.warning(f"⚠️ RAG validation service failed to initialize: {e}")
                self._rag_validation = None

            logger.info(
                "✅ Universal Request Processor initialized successfully with "
                "LLM intent understanding + Three-Tier Discovery + RAG Validation"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize Universal Request Processor: {e}")
            return False

    async def process_request(self, request: UniversalRequest) -> UniversalResponse:
        """
        ENHANCED: Process universal request with RAG validation support.

        Enhanced Process Flow:
        1. Parse Intent and Extract Devices
        2. Discover Commands (Three-tier system)
        3. Validate Commands (NEW - RAG validation)
        4. Handle Suggestions (NEW - interactive corrections)
        5. Execute Commands (ENHANCED - with validation info)
        6. Format Response (ENHANCED - with validation metadata)
        """
        start_time = time.time()
        self.stats["total_requests"] += 1

        logger.info(
            f"🔄 Processing universal request with RAG validation: "
            f"'{request.user_input}' via {request.interface_type.value}"
        )

        try:
            # Validate request
            validate_request_size(request)

            # 1. Parse Intent and Extract Devices
            intent, resolved_devices = await self._parse_and_resolve_request(request)

            # 2. Discover Commands using three-tier system
            discovery_results = await self._discover_commands_three_tier(
                intent, resolved_devices, request
            )

            # 3. NEW: Validate Commands using RAG
            validation_results = {}
            suggestions = []
            validation_time = 0.0

            if request.enable_command_validation and self._rag_validation:
                validation_start = time.time()
                validation_results = await self._validate_discovered_commands(
                    discovery_results, resolved_devices
                )
                validation_time = time.time() - validation_start

                # Generate suggestions for invalid commands
                suggestions = await self._generate_command_suggestions(
                    validation_results, discovery_results, resolved_devices
                )

                self.stats["rag_validations"] += 1

                logger.info(
                    f"🔍 RAG validation completed: {len(validation_results)} commands, "
                    f"{len(suggestions)} suggestions, {validation_time:.2f}s"
                )

            # 4. NEW: Handle command suggestions
            if suggestions:
                # Check if we need user confirmation
                requires_confirmation = await self._handle_command_suggestions(
                    suggestions, request, discovery_results
                )

                if requires_confirmation:
                    # Return response with suggestions for user confirmation
                    return await self._build_suggestion_response(
                        request,
                        intent,
                        discovery_results,
                        validation_results,
                        suggestions,
                        start_time,
                        validation_time,
                    )

            # 5. Execute Commands with validation info
            execution_results = await self._execute_commands_with_validation(
                discovery_results, validation_results, request
            )

            # 6. Format Response with validation metadata
            response = await self._build_final_response(
                request,
                intent,
                discovery_results,
                execution_results,
                validation_results,
                suggestions,
                start_time,
                validation_time,
            )

            # Update statistics
            if response.success:
                self.stats["successful_requests"] += 1
            else:
                self.stats["failed_requests"] += 1

            total_time = time.time() - start_time
            logger.info(
                f"✅ Universal request with RAG validation completed in {total_time:.2f}s - "
                f"Success: {response.success}"
            )

            return response

        except Exception as e:
            self.stats["failed_requests"] += 1
            logger.error(f"❌ Universal request processing failed: {e}")

            return UniversalResponse(
                request_id=request.request_id,
                success=False,
                execution_time=time.time() - start_time,
                formatted_response=f"Request processing failed: {str(e)}",
                error_message=str(e),
                total_devices=len(request.devices) if request.devices else 0,
                successful_devices=0,
                discovery_used=False,
                suggestions_generated=[],
            )

    async def _parse_and_resolve_request(
        self, request: UniversalRequest
    ) -> Tuple[str, List[Any]]:
        """Parse intent and resolve devices from request."""
        # Use enhanced LLM intent parser for pure understanding
        logger.info(f"🧠 Using LLM to understand user intent: '{request.user_input}'")

        intent = await self._intent_parser.extract_intent(request.user_input)
        device_names = await self._intent_parser.extract_devices(request.user_input)

        # Override with explicit devices if provided
        if request.devices:
            device_names = request.devices

        logger.info(f"🎯 LLM Intent: '{intent}' | LLM Devices: {device_names}")

        # Resolve device names to device objects using helper function
        resolved_devices = self._validate_and_resolve_devices(
            device_names, request.max_devices
        )

        logger.info(
            f"🎯 Final Intent: '{intent}' | Final Devices: {[d.name for d in resolved_devices]}"
        )

        return intent, resolved_devices

    def _validate_and_resolve_devices(
        self, device_names: List[str], max_devices: int
    ) -> List[Any]:
        """Helper function to validate and resolve device names to device objects"""
        if not device_names:
            return []

        resolved_devices = []

        for device_name in device_names:
            if device_name.lower() == "all":
                # Get all devices from inventory
                all_devices = self._inventory_service.get_all_devices()
                resolved_devices.extend(all_devices[:max_devices])
                logger.info(
                    f"Resolved 'all' to {len(all_devices[:max_devices])} devices"
                )
                break
            else:
                device = self._inventory_service.get_device(device_name)
                if device:
                    resolved_devices.append(device)
                else:
                    logger.warning(f"Device not found in inventory: {device_name}")

        if not resolved_devices:
            raise ValueError("No valid devices found")

        return resolved_devices

    async def _discover_commands_three_tier(
        self, intent: str, devices: List[Any], request: UniversalRequest
    ) -> Dict[str, DiscoveryInfo]:
        """
        FIXED: Discover commands using three-tier system with proper tuple handling

        Now properly handles tuple results returned by your existing HybridCommandDiscovery.
        """
        discovery_results = {}
        discovery_start = time.time()

        # Group devices by platform for efficient discovery
        platform_groups = {}
        for device in devices:
            platform = device.platform
            if platform not in platform_groups:
                platform_groups[platform] = []
            platform_groups[platform].append(device)

        # Discover commands per platform using three-tier system
        for platform, platform_devices in platform_groups.items():
            logger.info(
                f"🔍 Three-tier discovery for intent '{intent}' on platform {platform}"
            )

            try:
                # Use the existing three-tier hybrid discovery system
                # Returns tuple (command, confidence, source)
                discovery_result = await self._hybrid_discovery.discover_command(
                    intent=intent,
                    platform=platform,
                    user_context={"interface_type": request.interface_type.value},
                )

                # Handle tuple result from your existing discovery system
                if isinstance(discovery_result, tuple) and len(discovery_result) >= 3:
                    command, confidence, source = discovery_result[:3]
                elif isinstance(discovery_result, dict):
                    # Handle dictionary result format
                    command = discovery_result.get("command", "show version")
                    confidence = discovery_result.get("confidence", 0.5)
                    source = discovery_result.get("source", "unknown")
                else:
                    # Fallback for unexpected format
                    command = "show version"
                    confidence = 0.5
                    source = "fallback"

                # Apply discovery result to all devices with this platform
                for device in platform_devices:
                    discovery_results[device.name] = DiscoveryInfo(
                        intent_detected=intent,
                        discovered_command=command,
                        confidence=confidence,
                        reasoning=f"Three-tier discovery via {source}",
                        cached_result=source in ["redis_cache", "postgres_db"],
                        discovery_time=time.time() - discovery_start,
                        platform=platform,
                        fallback_used=source == "fallback",
                    )

                self.stats["total_discoveries"] += len(platform_devices)

                # Update cache hit statistics based on source
                if source == "redis_cache":
                    self.stats["redis_hits"] += len(platform_devices)
                elif source == "postgres_db":
                    self.stats["postgres_hits"] += len(platform_devices)
                elif source == "llm_discovery":
                    self.stats["llm_discoveries"] += len(platform_devices)

                logger.debug(
                    f"Discovery for {platform}: {command} "
                    f"(confidence: {confidence:.2f}, source: {source})"
                )

            except Exception as e:
                logger.error(f"Discovery failed for platform {platform}: {e}")
                # Continue with other platforms - don't fail the entire request

                # Add fallback discovery info for failed platforms
                for device in platform_devices:
                    discovery_results[device.name] = DiscoveryInfo(
                        intent_detected=intent,
                        discovered_command="show version",  # Safe fallback
                        confidence=0.3,
                        reasoning=f"Fallback due to discovery error: {str(e)}",
                        cached_result=False,
                        discovery_time=0.1,
                        platform=platform,
                        fallback_used=True,
                    )

        total_discovery_time = time.time() - discovery_start
        self.stats["avg_discovery_time"] = total_discovery_time

        logger.info(
            f"🔍 Three-tier command discovery completed in {total_discovery_time:.2f}s"
        )

        return discovery_results

    # NEW: RAG validation methods
    async def _validate_discovered_commands(
        self,
        discovery_results: Dict[str, DiscoveryInfo],
        devices: List[Any],
    ) -> Dict[str, CommandValidationInfo]:
        """Validate discovered commands using RAG service"""

        logger.info(f"🔍 RAG validating {len(discovery_results)} discovered commands")

        if not self._rag_validation:
            logger.warning("RAG validation service not available - skipping validation")
            return {}

        try:
            # Use RAG validation service to validate all commands
            raw_validation_results = (
                await self._rag_validation.validate_discovered_commands(
                    discovery_results, devices
                )
            )

            # Convert to CommandValidationInfo objects
            validation_results = {}
            for device_name, raw_result in raw_validation_results.items():
                validation_results[device_name] = CommandValidationInfo(
                    command=raw_result.command,
                    is_valid=raw_result.is_valid,
                    confidence=raw_result.confidence,
                    suggested_command=raw_result.suggested_command,
                    explanation=raw_result.explanation,
                    documentation_reference=raw_result.documentation_source,
                    validation_source="rag_barrow",
                )

            return validation_results

        except Exception as e:
            logger.error(f"RAG validation failed: {e}")
            return {}

    async def _generate_command_suggestions(
        self,
        validation_results: Dict[str, CommandValidationInfo],
        discovery_results: Dict[str, DiscoveryInfo],
        devices: List[Any],
    ) -> List[CommandSuggestion]:
        """Generate command suggestions for invalid commands"""

        suggestions = []

        for device_name, validation_info in validation_results.items():
            if not validation_info.is_valid and validation_info.suggested_command:
                # Find the corresponding device
                device = next((d for d in devices if d.name == device_name), None)
                if not device:
                    continue

                # Create suggestion using the helper function
                suggestion = create_command_suggestion(
                    device_name=device_name,
                    original_command=validation_info.command,
                    suggested_command=validation_info.suggested_command,
                    vendor=device.vendor,
                    platform=device.platform,
                    confidence=validation_info.confidence,
                    explanation=validation_info.explanation
                    or "Command syntax correction",
                    documentation_reference=validation_info.documentation_reference
                    or "Vendor documentation",
                )
                suggestions.append(suggestion)

        self.stats["suggestions_generated"] += len(suggestions)

        # Filter by confidence threshold
        high_confidence_suggestions = filter_high_confidence_suggestions(
            suggestions, confidence_threshold=0.7
        )

        logger.info(
            f"🔧 Generated {len(suggestions)} suggestions, "
            f"{len(high_confidence_suggestions)} high-confidence"
        )

        return high_confidence_suggestions

    async def _handle_command_suggestions(
        self,
        suggestions: List[CommandSuggestion],
        request: UniversalRequest,
        discovery_results: Dict[str, DiscoveryInfo],
    ) -> bool:
        """
        Handle command suggestions - determine if user confirmation needed

        Returns True if user confirmation is required
        """

        if not suggestions:
            return False

        auto_applied = 0

        # Check if any suggestions can be auto-applied
        for suggestion in suggestions:
            if should_auto_apply_suggestion(
                suggestion,
                request.auto_apply_suggestions,
                request.suggestion_confidence_threshold,
            ):
                # Auto-apply the suggestion
                if suggestion.device_name in discovery_results:
                    discovery_results[
                        suggestion.device_name
                    ].discovered_command = suggestion.suggested_command
                    auto_applied += 1

        self.stats["suggestions_applied"] += auto_applied

        # Return True if we have suggestions that require user confirmation
        pending_suggestions = [
            s
            for s in suggestions
            if not should_auto_apply_suggestion(
                s,
                request.auto_apply_suggestions,
                request.suggestion_confidence_threshold,
            )
        ]

        logger.info(
            f"🔧 Auto-applied {auto_applied} suggestions, "
            f"{len(pending_suggestions)} require user confirmation"
        )

        return len(pending_suggestions) > 0

    async def _execute_commands_with_validation(
        self,
        discovery_results: Dict[str, DiscoveryInfo],
        validation_results: Dict[str, CommandValidationInfo],
        request: UniversalRequest,
    ) -> Dict[str, DeviceExecutionResult]:
        """Execute commands with validation information attached"""

        logger.info(f"🚀 Executing {len(discovery_results)} validated commands")

        # Build device-command pairs for executor
        device_command_pairs = []

        for device_name, discovery_info in discovery_results.items():
            device = self._inventory_service.get_device(device_name)
            if device:
                device_command_pairs.append((device, discovery_info.discovered_command))

        # Execute commands concurrently using existing executor
        execution_start = time.time()
        raw_results = await self._network_executor.execute_multi_device_commands(
            device_command_pairs
        )
        execution_time = time.time() - execution_start
        self.stats["avg_execution_time"] = execution_time

        # Convert to DeviceExecutionResult objects with validation info
        execution_results = {}

        for device_obj, raw_result in raw_results.items():
            device_name = device_obj.name
            discovery_info = discovery_results[device_name]
            validation_info = validation_results.get(device_name)

            execution_results[device_name] = DeviceExecutionResult(
                device_name=device_name,
                success=raw_result.success,
                execution_time=raw_result.execution_time,
                command_executed=discovery_info.discovered_command,
                discovery_used=True,
                discovery_confidence=discovery_info.confidence,
                cached_discovery=discovery_info.cached_result,
                validation_info=validation_info,  # NEW: Include validation info
                validation_bypassed=not request.enable_command_validation,
                raw_output=raw_result.raw_output
                if request.include_raw_output
                else None,
                error_message=raw_result.error_message
                if not raw_result.success
                else None,
            )

        return execution_results

    async def _build_suggestion_response(
        self,
        request: UniversalRequest,
        intent: str,
        discovery_results: Dict[str, DiscoveryInfo],
        validation_results: Dict[str, CommandValidationInfo],
        suggestions: List[CommandSuggestion],
        start_time: float,
        validation_time: float,
    ) -> UniversalResponse:
        """Build response with command suggestions for user confirmation"""

        total_time = time.time() - start_time

        return UniversalResponse(
            request_id=request.request_id,
            success=False,  # Don't execute, need user confirmation
            execution_time=total_time,
            formatted_response="Command validation found issues that require confirmation",
            total_devices=len(discovery_results),
            successful_devices=0,
            discovery_used=True,
            suggestions_generated=suggestions,
            validation_enabled=True,
            interface_metadata={
                "intent": intent,
                "requires_user_confirmation": True,
                "suggestion_count": len(suggestions),
                "interface_type": request.interface_type.value,
            },
        )

    async def _build_final_response(
        self,
        request: UniversalRequest,
        intent: str,
        discovery_results: Dict[str, DiscoveryInfo],
        execution_results: Dict[str, DeviceExecutionResult],
        validation_results: Dict[str, CommandValidationInfo],
        suggestions: List[CommandSuggestion],
        start_time: float,
        validation_time: float,
    ) -> UniversalResponse:
        """Build final response with all metadata"""

        total_time = time.time() - start_time
        successful_devices = sum(
            1 for result in execution_results.values() if result.success
        )

        # Use universal formatter for consistent response formatting
        formatted_response = await self._universal_formatter.format_response(
            user_request=request.user_input,
            intent=intent,
            execution_results=execution_results,
            discovery_results=discovery_results,
            interface_type=request.interface_type,
            output_format=request.output_format,
            format_options=request.format_options,
        )

        return UniversalResponse(
            request_id=request.request_id,
            success=successful_devices > 0,
            execution_time=total_time,
            formatted_response=formatted_response,
            total_devices=len(discovery_results),
            successful_devices=successful_devices,
            discovery_used=True,
            discovery_info=[discovery_results[device] for device in discovery_results],
            new_discoveries=sum(
                1 for d in discovery_results.values() if not d.cached_result
            ),
            suggestions_generated=suggestions,
            validation_enabled=bool(validation_results),
            validation_time=validation_time,
            interface_metadata={
                "intent": intent,
                "interface_type": request.interface_type.value,
                "validation_enabled": request.enable_command_validation,
                "auto_apply_enabled": request.auto_apply_suggestions,
            },
            device_results=list(execution_results.values()),
        )

    async def health_check(self) -> Dict[str, str]:
        """Check health of all components"""
        health = {
            "universal_processor": "healthy",
            "hybrid_discovery": "unknown",
            "intent_parser": "unknown",
            "network_executor": "unknown",
            "universal_formatter": "unknown",
            "rag_validation": "unknown",
        }

        try:
            # Check hybrid discovery
            if self._hybrid_discovery:
                health["hybrid_discovery"] = "healthy"

            # Check intent parser
            if self._intent_parser:
                health["intent_parser"] = "healthy"

            # Check network executor
            if self._network_executor:
                health["network_executor"] = "healthy"

            # Check universal formatter
            if self._universal_formatter:
                health["universal_formatter"] = "healthy"

            # Check RAG validation
            if self._rag_validation:
                rag_healthy = await self._rag_validation.health_check()
                health["rag_validation"] = "healthy" if rag_healthy else "unavailable"
            else:
                health["rag_validation"] = "disabled"

        except Exception as e:
            health["universal_processor"] = f"error: {str(e)}"

        return health

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.stats.copy()

        # Calculate success rate
        if stats["total_requests"] > 0:
            stats["success_rate"] = (
                stats["successful_requests"] / stats["total_requests"]
            )

            # Calculate hit rates
            stats["redis_hit_rate"] = (
                stats["redis_hits"] / stats["total_discoveries"]
                if stats["total_discoveries"] > 0
                else 0
            )
            stats["postgres_hit_rate"] = (
                stats["postgres_hits"] / stats["total_discoveries"]
                if stats["total_discoveries"] > 0
                else 0
            )
            stats["llm_discovery_rate"] = (
                stats["llm_discoveries"] / stats["total_discoveries"]
                if stats["total_discoveries"] > 0
                else 0
            )

        return stats


# Global instance for use throughout the application
universal_processor = UniversalRequestProcessor()

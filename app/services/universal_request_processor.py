# File: ~/net-chatbot/app/services/universal_request_processor.py
# ENHANCED UNIVERSAL REQUEST PROCESSOR - Integration with RAG Command Validation

"""
Universal Request Processor - Enhanced with RAG Command Validation (Phase 6)

NEW ARCHITECTURE FLOW:
User Input → LLM Intent Parser → Three-Tier Discovery →
                                    ↓
[NEW] RAG Command Validation → Command Execution → Universal Formatter → Response

Key Enhancements:
- RAG-powered command validation before execution
- Smart command suggestions for invalid commands
- Interactive correction workflow
- Learning integration with existing PostgreSQL
- Graceful fallback when RAG unavailable
- Preserves all existing performance optimizations

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
    validate_devices,
    validate_request_size,
)

# Existing imports (UNCHANGED)
from app.services.hybrid_command_discovery import create_hybrid_discovery_service

# NEW: RAG validation service import
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
        # Core services (ENHANCED with RAG validation)
        self._hybrid_discovery = None  # Three-tier discovery system
        self._rag_validation = None  # NEW: RAG command validation service
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
            "cache_hits": 0,
            "redis_hits": 0,
            "postgres_hits": 0,
            "llm_discoveries": 0,
            "average_execution_time": 0.0,
            # NEW: Validation metrics
            "total_validations": 0,
            "validation_successes": 0,
            "validation_failures": 0,
            "suggestions_generated": 0,
            "suggestions_applied": 0,
            "validation_fallbacks": 0,
            "average_validation_time": 0.0,
        }

    async def initialize(self):
        """Initialize all required services - ENHANCED with RAG validation"""
        logger.info(
            "🚀 Initializing Universal Request Processor with "
            "Three-Tier Discovery + RAG Command Validation..."
        )

        try:
            # Import existing services (UNCHANGED)
            from app.services.executor import network_executor
            from app.services.inventory import inventory_service
            from app.services.pure_llm_intent_parser import (
                create_enhanced_intent_parser,
            )
            from app.services.universal_formatter import universal_formatter

            # Initialize existing services (UNCHANGED)
            self._hybrid_discovery = await create_hybrid_discovery_service()
            self._network_executor = network_executor
            self._inventory_service = inventory_service
            self._intent_parser = create_enhanced_intent_parser()
            self._universal_formatter = universal_formatter

            # NEW: Initialize RAG validation service
            self._rag_validation = await get_rag_validation_service()

            logger.info(
                "✅ Universal Request Processor initialized successfully with "
                "LLM intent understanding + Three-Tier Discovery + RAG Validation"
            )
            return True

        except ImportError as e:
            logger.error(f"❌ Failed to initialize Universal Request Processor: {e}")
            return False

    async def process_request(self, request: UniversalRequest) -> UniversalResponse:
        """
        ENHANCED MAIN PIPELINE: Process any request with RAG command validation

        NEW FLOW:
        1. Parse Intent and Extract Devices (UNCHANGED)
        2. Discover Commands (Three-Tier Discovery) (UNCHANGED)
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
            # Validate request (UNCHANGED)
            validate_request_size(request)

            # 1. Parse Intent and Extract Devices (UNCHANGED)
            intent, resolved_devices = await self._parse_and_resolve_request(request)

            # 2. Discover Commands (Three-Tier Discovery) (UNCHANGED)
            discovery_results = await self._discover_commands_three_tier(
                intent, resolved_devices, request
            )

            # 3. NEW: Validate Commands using RAG
            validation_results = {}
            suggestions = []
            validation_time = 0.0

            if request.enable_command_validation:
                validation_start = time.time()
                validation_results = await self._validate_discovered_commands(
                    discovery_results, resolved_devices
                )
                validation_time = time.time() - validation_start

                # Generate suggestions for invalid commands
                suggestions = await self._generate_command_suggestions(
                    validation_results, discovery_results, resolved_devices
                )

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

            # 5. Execute Commands (ENHANCED with validation info)
            execution_results = await self._execute_commands_with_validation(
                discovery_results, validation_results, request
            )

            # 6. Format Response (ENHANCED with validation metadata)
            formatted_response = await self._format_response(
                request, intent, execution_results, discovery_results
            )

            # 7. Build Universal Response (ENHANCED with validation data)
            response = await self._build_enhanced_response(
                request,
                intent,
                execution_results,
                discovery_results,
                validation_results,
                suggestions,
                formatted_response,
                start_time,
                validation_time,
            )

            self.stats["successful_requests"] += 1
            self._update_performance_stats(response.execution_time, validation_time)

            logger.info(
                f"✅ Universal request with RAG validation completed in "
                f"{response.execution_time:.2f}s - Success: {response.success}"
            )
            return response

        except Exception as e:
            self.stats["failed_requests"] += 1
            execution_time = time.time() - start_time

            logger.error(f"❌ Universal request with RAG validation failed: {e}")

            # Return error response (ENHANCED with validation context)
            return UniversalResponse(
                request_id=request.request_id,
                success=False,
                execution_time=execution_time,
                formatted_response=f"Request failed: {str(e)}",
                error_message=str(e),
                validation_enabled=request.enable_command_validation,
            )

    # Existing methods (UNCHANGED) - keeping original implementations
    async def _parse_and_resolve_request(
        self, request: UniversalRequest
    ) -> Tuple[str, List[Any]]:
        """Parse user input and resolve devices - UNCHANGED"""
        logger.info(f"🧠 Using LLM to understand user intent: '{request.user_input}'")

        # Use LLM to parse intent from natural language
        intent = await self._intent_parser.extract_intent(request.user_input)
        llm_devices = await self._intent_parser.extract_devices(request.user_input)

        # Combine with any explicitly provided devices
        if request.devices:
            device_names = validate_devices(request.devices)
        else:
            device_names = llm_devices

        logger.info(f"🎯 LLM Intent: '{intent}' | LLM Devices: {device_names}")

        # Handle "all" devices
        if "all" in device_names:
            all_devices = self._inventory_service.get_all_devices()
            resolved_devices = all_devices[: request.max_devices]
            logger.info(f"Resolved 'all' to {len(resolved_devices)} devices")
        else:
            resolved_devices = []
            for device_name in device_names:
                device = self._inventory_service.get_device(device_name)
                if device:
                    resolved_devices.append(device)
                else:
                    logger.warning(f"Device not found in inventory: {device_name}")

        if not resolved_devices:
            raise ValueError("No valid devices found")

        logger.info(
            f"🎯 Final Intent: '{intent}' | Final Devices: {[d.name for d in resolved_devices]}"
        )

        return intent, resolved_devices

    async def _discover_commands_three_tier(
        self, intent: str, devices: List[Any], request: UniversalRequest
    ) -> Dict[str, DiscoveryInfo]:
        """Discover commands using three-tier system - UNCHANGED"""
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
                # Use the three-tier hybrid discovery system
                discovery_result = await self._hybrid_discovery.discover_command(
                    intent=intent,
                    platform=platform,
                    user_context={"interface_type": request.interface_type.value},
                )

                # Apply discovery result to all devices with this platform
                for device in platform_devices:
                    discovery_results[device.name] = DiscoveryInfo(
                        intent_detected=intent,
                        discovered_command=discovery_result.command,
                        confidence=discovery_result.confidence,
                        reasoning=discovery_result.reasoning,
                        cached_result=discovery_result.source != "llm_discovery",
                        discovery_time=discovery_result.discovery_time,
                        platform=platform,
                        fallback_used=discovery_result.source == "fallback",
                    )

                self.stats["total_discoveries"] += len(platform_devices)

                # Update cache hit statistics
                if discovery_result.source == "redis_cache":
                    self.stats["redis_hits"] += len(platform_devices)
                elif discovery_result.source == "postgres_db":
                    self.stats["postgres_hits"] += len(platform_devices)
                elif discovery_result.source == "llm_discovery":
                    self.stats["llm_discoveries"] += len(platform_devices)

            except Exception as e:
                logger.error(f"Discovery failed for platform {platform}: {e}")
                # Continue with other platforms

        total_discovery_time = time.time() - discovery_start
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
                    validation_source="rag"
                    if not raw_result.fallback_used
                    else "fallback",
                    validation_time=raw_result.validation_time,
                    suggested_command=raw_result.suggested_command,
                    explanation=raw_result.explanation,
                    documentation_reference=raw_result.documentation_source,
                    validation_error=raw_result.error_message,
                    fallback_used=raw_result.fallback_used,
                )

            # Update statistics
            self.stats["total_validations"] += len(validation_results)
            self.stats["validation_successes"] += sum(
                1 for v in validation_results.values() if v.is_valid
            )
            self.stats["validation_failures"] += sum(
                1 for v in validation_results.values() if not v.is_valid
            )
            self.stats["validation_fallbacks"] += sum(
                1 for v in validation_results.values() if v.fallback_used
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
        """Generate user-friendly command suggestions for invalid commands"""

        suggestions = []
        device_lookup = {device.name: device for device in devices}

        for device_name, validation_info in validation_results.items():
            if not validation_info.is_valid and validation_info.suggested_command:
                device = device_lookup.get(device_name)
                if device:
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
        device_lookup = {}

        for device_name, discovery_info in discovery_results.items():
            device = self._inventory_service.get_device(device_name)
            if device:
                device_command_pairs.append((device, discovery_info.discovered_command))
                device_lookup[device_name] = device

        # Execute commands concurrently (UNCHANGED)
        execution_start = time.time()
        raw_results = await self._network_executor.execute_multi_device_commands(
            device_command_pairs
        )
        execution_time = time.time() - execution_start

        # Convert to DeviceExecutionResult objects (ENHANCED with validation info)
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
                error_message=getattr(raw_result, "error_message", None),
                platform=device_obj.platform,
                vendor=device_obj.vendor,
            )

        logger.info(
            f"✅ Command execution completed in {execution_time:.2f}s - "
            f"Success: {sum(1 for r in execution_results.values() if r.success)}/{len(execution_results)}"
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

        execution_time = time.time() - start_time

        # Create formatted response for suggestions
        if request.output_format.value == "json":
            formatted_response = {
                "message": "Command validation found issues - suggestions available",
                "suggestions": [
                    {
                        "device": s.device_name,
                        "original_command": s.original_command,
                        "suggested_command": s.suggested_command,
                        "confidence": s.confidence,
                        "explanation": s.explanation,
                        "suggestion_id": s.suggestion_id,
                    }
                    for s in suggestions
                ],
                "requires_confirmation": True,
            }
        else:
            # Conversational format
            suggestion_text = "\n".join(
                [
                    f"Device {s.device_name}:\n"
                    f"  Original: {s.original_command}\n"
                    f"  Suggested: {s.suggested_command}\n"
                    f"  Reason: {s.explanation}\n"
                    f"  Confidence: {s.confidence:.1%}"
                    for s in suggestions
                ]
            )
            formatted_response = (
                f"⚠️ Command validation detected issues with suggested corrections:\n\n"
                f"{suggestion_text}\n\n"
                f"Would you like to apply these suggestions?"
            )

        return UniversalResponse(
            request_id=request.request_id,
            success=False,  # Not executed yet
            execution_time=execution_time,
            formatted_response=formatted_response,
            total_devices=len(discovery_results),
            successful_devices=0,
            failed_devices=0,
            validation_enabled=True,
            validation_time=validation_time,
            total_validations=len(validation_results),
            validation_failures=sum(
                1 for v in validation_results.values() if not v.is_valid
            ),
            suggestions_generated=suggestions,
            requires_user_confirmation=True,
            pending_suggestions=suggestions,
        )

    async def _build_enhanced_response(
        self,
        request: UniversalRequest,
        intent: str,
        execution_results: Dict[str, DeviceExecutionResult],
        discovery_results: Dict[str, DiscoveryInfo],
        validation_results: Dict[str, CommandValidationInfo],
        suggestions: List[CommandSuggestion],
        formatted_response: Any,
        start_time: float,
        validation_time: float,
    ) -> UniversalResponse:
        """Build enhanced universal response with validation metadata"""

        execution_time = time.time() - start_time
        successful_devices = sum(
            1 for result in execution_results.values() if result.success
        )

        return UniversalResponse(
            request_id=request.request_id,
            success=successful_devices > 0,
            execution_time=execution_time,
            formatted_response=formatted_response,
            total_devices=len(execution_results),
            successful_devices=successful_devices,
            failed_devices=len(execution_results) - successful_devices,
            device_results=list(execution_results.values()),
            discovery_used=len(discovery_results) > 0,
            discovery_info=list(discovery_results.values()),
            new_discoveries=sum(
                1 for d in discovery_results.values() if not d.cached_result
            ),
            # NEW: Validation metadata
            validation_enabled=request.enable_command_validation,
            validation_time=validation_time,
            total_validations=len(validation_results),
            validation_failures=sum(
                1 for v in validation_results.values() if not v.is_valid
            ),
            suggestions_generated=suggestions,
            validation_bypassed_count=sum(
                1 for r in execution_results.values() if r.validation_bypassed
            ),
            requires_user_confirmation=False,  # Commands were executed
            # Existing metadata
            discovery_time=sum(d.discovery_time for d in discovery_results.values()),
            execution_time_per_device={
                name: result.execution_time
                for name, result in execution_results.items()
            },
            partial_success=0 < successful_devices < len(execution_results),
        )

    # Existing methods (UNCHANGED) - keeping format_response, etc.
    async def _format_response(
        self,
        request: UniversalRequest,
        intent: str,
        execution_results: Dict[str, DeviceExecutionResult],
        discovery_results: Dict[str, DiscoveryInfo],
    ) -> Any:
        """Format response using Universal Formatter - UNCHANGED"""
        return await self._universal_formatter.format_response(
            request.user_input,
            intent,
            execution_results,
            discovery_results,
            request.interface_type,
            request.output_format,
            request.format_options or {},
        )

    def _update_performance_stats(
        self, execution_time: float, validation_time: float = 0.0
    ):
        """Update performance statistics - ENHANCED with validation metrics"""
        # Update existing stats
        total_requests = self.stats["total_requests"]
        current_avg = self.stats["average_execution_time"]
        self.stats["average_execution_time"] = (
            current_avg * (total_requests - 1) + execution_time
        ) / total_requests

        # Update validation stats
        if validation_time > 0 and self.stats["total_validations"] > 0:
            total_validations = self.stats["total_validations"]
            current_val_avg = self.stats.get("average_validation_time", 0.0)
            self.stats["average_validation_time"] = (
                current_val_avg * (total_validations - 1) + validation_time
            ) / total_validations


# Global processor instance (UNCHANGED)
universal_processor = UniversalRequestProcessor()

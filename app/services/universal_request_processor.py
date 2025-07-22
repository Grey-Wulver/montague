# app/services/universal_request_processor.py
"""
Universal Request Processor - The Core Engine (ENHANCED WITH LLM)
Processes ALL requests regardless of interface (Chat, API, CLI, Web)
Now uses pure LLM understanding for intent parsing
"""

import logging
import time
from typing import Any, Dict, List, Tuple

from app.models.universal_requests import (
    DeviceExecutionResult,
    DiscoveryInfo,
    UniversalRequest,
    UniversalResponse,
    validate_devices,
    validate_request_size,
)

logger = logging.getLogger(__name__)


class UniversalRequestProcessor:
    """
    The ONE engine that processes ALL network automation requests
    Chat, API, CLI, Web - all interfaces use this same pipeline
    NOW WITH PURE LLM UNDERSTANDING!
    """

    def __init__(self):
        # Core services - imported dynamically to avoid circular imports
        self._dynamic_discovery = None
        self._network_executor = None
        self._inventory_service = None
        self._intent_parser = None
        self._universal_formatter = None

        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_discoveries": 0,
            "cache_hits": 0,
            "average_execution_time": 0.0,
        }

    async def initialize(self):
        """Initialize all required services"""
        logger.info("🚀 Initializing Universal Request Processor...")

        try:
            # Import services
            from app.services.dynamic_command_discovery import dynamic_discovery
            from app.services.executor import network_executor
            from app.services.inventory import inventory_service
            from app.services.pure_llm_intent_parser import (
                create_enhanced_intent_parser,  # CHANGED: Use LLM intent parser
            )
            from app.services.universal_formatter import universal_formatter

            self._dynamic_discovery = dynamic_discovery
            self._network_executor = network_executor
            self._inventory_service = inventory_service
            self._intent_parser = (
                create_enhanced_intent_parser()
            )  # CHANGED: Use LLM intent parser
            self._universal_formatter = universal_formatter

            logger.info(
                "✅ Universal Request Processor initialized successfully with LLM intent understanding"
            )
            return True

        except ImportError as e:
            logger.error(f"❌ Failed to initialize Universal Request Processor: {e}")
            return False

    async def process_request(self, request: UniversalRequest) -> UniversalResponse:
        """
        THE MAIN PIPELINE: Process any request from any interface
        This is the heart of the universal system - NOW WITH LLM UNDERSTANDING!
        """
        start_time = time.time()
        self.stats["total_requests"] += 1

        logger.info(
            f"🔄 Processing universal request: '{request.user_input}' via {request.interface_type.value}"
        )

        try:
            # Validate request
            validate_request_size(request)

            # 1. Parse Intent and Extract Devices (NOW USES LLM!)
            intent, resolved_devices = await self._parse_and_resolve_request(request)

            # 2. Discover Commands for Each Device
            discovery_results = await self._discover_commands(
                intent, resolved_devices, request
            )

            # 3. Execute Commands on All Devices
            execution_results = await self._execute_commands(discovery_results, request)

            # 4. Format Response for Requested Interface
            formatted_response = await self._format_response(
                request, intent, execution_results, discovery_results
            )

            # 5. Build Universal Response
            response = await self._build_response(
                request,
                intent,
                execution_results,
                discovery_results,
                formatted_response,
                start_time,
            )

            self.stats["successful_requests"] += 1
            self._update_performance_stats(response.execution_time)

            logger.info(
                f"✅ Universal request completed in {response.execution_time:.2f}s - Success: {response.success}"
            )
            return response

        except Exception as e:
            self.stats["failed_requests"] += 1
            execution_time = time.time() - start_time

            logger.error(f"❌ Universal request failed: {e}")

            # Return error response
            return UniversalResponse(
                request_id=request.request_id,
                success=False,
                execution_time=execution_time,
                formatted_response=f"Request failed: {str(e)}",
                error_message=str(e),
            )

    async def _parse_and_resolve_request(
        self, request: UniversalRequest
    ) -> Tuple[str, List[Any]]:
        """Parse user input and resolve devices - NOW WITH PURE LLM UNDERSTANDING!"""

        logger.info(f"🧠 Using LLM to understand user intent: '{request.user_input}'")

        # Use LLM to parse intent from natural language - NO PATTERN MATCHING!
        intent = await self._intent_parser.extract_intent(request.user_input)

        # Use LLM to extract devices from natural language
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
            resolved_devices = all_devices[: request.max_devices]  # Respect limits
            logger.info(f"Resolved 'all' to {len(resolved_devices)} devices")
        else:
            # Get device objects from inventory
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

    async def _discover_commands(
        self, intent: str, devices: List[Any], request: UniversalRequest
    ) -> Dict[str, DiscoveryInfo]:
        """Discover appropriate commands for each device platform"""

        discovery_results = {}
        discovery_start = time.time()

        # Group devices by platform for efficient discovery
        platform_groups = {}
        for device in devices:
            platform = device.platform
            if platform not in platform_groups:
                platform_groups[platform] = []
            platform_groups[platform].append(device)

        # Discover commands per platform
        for platform, platform_devices in platform_groups.items():
            logger.info(
                f"🧠 Discovering command for intent '{intent}' on platform {platform}"
            )

            # Use first device of platform for discovery
            sample_device = platform_devices[0]

            discovery_result = await self._dynamic_discovery.discover_command(
                user_intent=intent,
                device=sample_device,
                allow_cached=not request.force_fresh_discovery,
            )

            if discovery_result.success:
                # Apply discovery to all devices of this platform
                for device in platform_devices:
                    # FIXED: Check if this was a cached result by examining the reasoning
                    cached_result = (
                        "cached" in discovery_result.reasoning.lower()
                        if discovery_result.reasoning
                        else False
                    )

                    discovery_results[device.name] = DiscoveryInfo(
                        intent_detected=intent,
                        discovered_command=discovery_result.command,  # FIXED: use .command from DiscoveryResult
                        confidence=discovery_result.confidence,
                        reasoning=discovery_result.reasoning,
                        cached_result=cached_result,  # FIXED: derive from reasoning instead of accessing non-existent attribute
                        discovery_time=discovery_result.processing_time,  # FIXED: use .processing_time from DiscoveryResult
                        platform=platform,
                        fallback_used=False,
                    )

                    if not cached_result:
                        self.stats["total_discoveries"] += 1
                    else:
                        self.stats["cache_hits"] += 1

                logger.info(
                    f"✅ Discovered '{discovery_result.command}' for {len(platform_devices)} {platform} devices"
                )
            else:
                logger.error(
                    f"❌ Command discovery failed for platform {platform}: {discovery_result.reasoning if hasattr(discovery_result, 'reasoning') else 'Unknown error'}"
                )
                raise ValueError(f"Could not discover command for platform {platform}")

        total_discovery_time = time.time() - discovery_start
        logger.info(f"🔍 Command discovery completed in {total_discovery_time:.2f}s")

        return discovery_results

    async def _execute_commands(
        self, discovery_results: Dict[str, DiscoveryInfo], request: UniversalRequest
    ) -> Dict[str, DeviceExecutionResult]:
        """Execute discovered commands on all devices"""

        logger.info(f"🚀 Executing commands on {len(discovery_results)} devices")

        # Build device-command pairs for executor
        device_command_pairs = []
        device_lookup = {}

        for device_name, discovery_info in discovery_results.items():
            device = self._inventory_service.get_device(device_name)
            if device:
                device_command_pairs.append((device, discovery_info.discovered_command))
                device_lookup[device_name] = device

        # Execute commands concurrently
        execution_start = time.time()
        raw_results = await self._network_executor.execute_multi_device_commands(
            device_command_pairs
        )
        execution_time = time.time() - execution_start

        # Convert to DeviceExecutionResult objects
        execution_results = {}

        # Handle the executor's return format: {device_obj: response} -> {device_name: response}
        for device_obj, raw_result in raw_results.items():
            device_name = device_obj.name  # Get device name from device object
            discovery_info = discovery_results[device_name]

            execution_results[device_name] = DeviceExecutionResult(
                device_name=device_name,
                success=raw_result.success,  # Access success attribute directly
                execution_time=raw_result.execution_time,  # Access execution_time attribute directly
                command_executed=discovery_info.discovered_command,
                discovery_used=True,
                discovery_confidence=discovery_info.confidence,
                cached_discovery=discovery_info.cached_result,
                raw_output=raw_result.raw_output
                if request.include_raw_output
                else None,
                error_message=raw_result.error_message
                if hasattr(raw_result, "error_message")
                else None,
                platform=device_obj.platform,
                vendor=device_obj.vendor,
                connection_method="ssh",
            )

        successful = sum(1 for r in execution_results.values() if r.success)
        logger.info(
            f"✅ Command execution completed in {execution_time:.2f}s - {successful}/{len(execution_results)} successful"
        )

        return execution_results

    async def _format_response(
        self,
        request: UniversalRequest,
        intent: str,
        execution_results: Dict[str, DeviceExecutionResult],
        discovery_results: Dict[str, DiscoveryInfo],
    ) -> Any:
        """Format response using Universal Formatter"""

        formatting_start = time.time()

        formatted_response = await self._universal_formatter.format_response(
            user_request=request.user_input,
            intent=intent,
            execution_results=execution_results,
            discovery_results=discovery_results,
            interface_type=request.interface_type,
            output_format=request.output_format,
            format_options=request.format_options or {},
        )

        formatting_time = time.time() - formatting_start
        logger.info(
            f"🎨 Response formatted in {formatting_time:.2f}s for {request.interface_type.value}"
        )

        return formatted_response

    async def _build_response(
        self,
        request: UniversalRequest,
        intent: str,
        execution_results: Dict[str, DeviceExecutionResult],
        discovery_results: Dict[str, DiscoveryInfo],
        formatted_response: Any,
        start_time: float,
    ) -> UniversalResponse:
        """Build the final UniversalResponse object"""

        execution_time = time.time() - start_time

        # Calculate statistics
        total_devices = len(execution_results)
        successful_devices = sum(1 for r in execution_results.values() if r.success)
        failed_devices = total_devices - successful_devices

        # Discovery statistics
        discovery_used = any(discovery_results.values())
        new_discoveries = sum(
            1 for d in discovery_results.values() if not d.cached_result
        )
        discovery_time = sum(d.discovery_time for d in discovery_results.values())

        # Build device results list
        device_results = list(execution_results.values())

        # Build discovery info list
        discovery_info_list = list(discovery_results.values())

        # Optional raw outputs
        raw_outputs = None
        if request.include_raw_output:
            raw_outputs = {
                name: result.raw_output
                for name, result in execution_results.items()
                if result.raw_output
            }

        return UniversalResponse(
            request_id=request.request_id,
            success=successful_devices > 0,
            execution_time=execution_time,
            formatted_response=formatted_response,
            total_devices=total_devices,
            successful_devices=successful_devices,
            failed_devices=failed_devices,
            device_results=device_results,
            discovery_used=discovery_used,
            discovery_info=discovery_info_list,
            new_discoveries=new_discoveries,
            raw_device_outputs=raw_outputs,
            partial_success=successful_devices > 0 and failed_devices > 0,
            discovery_time=discovery_time,
            execution_time_per_device={
                name: result.execution_time
                for name, result in execution_results.items()
            },
            interface_metadata={
                "interface_type": request.interface_type.value,
                "output_format": request.output_format.value,
                "intent": intent,
                "llm_enhanced": True,  # NEW: Flag to indicate LLM processing
            },
        )

    def _update_performance_stats(self, execution_time: float):
        """Update running performance statistics"""

        current_avg = self.stats["average_execution_time"]
        total_successful = self.stats["successful_requests"]

        if total_successful == 1:
            self.stats["average_execution_time"] = execution_time
        else:
            # Running average calculation
            self.stats["average_execution_time"] = (
                current_avg * (total_successful - 1) + execution_time
            ) / total_successful

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""

        success_rate = 0.0
        if self.stats["total_requests"] > 0:
            success_rate = (
                self.stats["successful_requests"] / self.stats["total_requests"]
            ) * 100

        cache_hit_rate = 0.0
        total_discoveries = self.stats["total_discoveries"] + self.stats["cache_hits"]
        if total_discoveries > 0:
            cache_hit_rate = (self.stats["cache_hits"] / total_discoveries) * 100

        return {
            **self.stats,
            "success_rate_percent": round(success_rate, 2),
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "total_discoveries_with_cache": total_discoveries,
            "llm_enhanced": True,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all dependent services"""

        health_status = {"universal_processor": "healthy", "services": {}}

        try:
            # Check dynamic discovery
            if self._dynamic_discovery:
                health_status["services"]["dynamic_discovery"] = "healthy"
            else:
                health_status["services"]["dynamic_discovery"] = "not_initialized"

            # Check network executor
            if self._network_executor:
                health_status["services"]["network_executor"] = "healthy"
            else:
                health_status["services"]["network_executor"] = "not_initialized"

            # Check inventory service
            if self._inventory_service:
                device_count = len(self._inventory_service.get_all_devices())
                health_status["services"]["inventory_service"] = (
                    f"healthy ({device_count} devices)"
                )
            else:
                health_status["services"]["inventory_service"] = "not_initialized"

            # Check LLM services
            if self._universal_formatter:
                # Test LLM connectivity
                llm_healthy = await self._universal_formatter.health_check()
                health_status["services"]["llm_formatter"] = (
                    "healthy" if llm_healthy else "llm_unavailable"
                )
            else:
                health_status["services"]["llm_formatter"] = "not_initialized"

            # Check LLM intent parser
            if self._intent_parser:
                health_status["services"]["llm_intent_parser"] = "healthy"
            else:
                health_status["services"]["llm_intent_parser"] = "not_initialized"

        except Exception as e:
            health_status["universal_processor"] = f"error: {str(e)}"

        return health_status


# Global instance
universal_processor = UniversalRequestProcessor()

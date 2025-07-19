# app/services/intent_executor.py
import logging
from typing import Dict, List

from app.models.intents import DeviceIntentResult, IntentRequest, IntentResponse
from app.services.command_translator import command_translator
from app.services.enhanced_output_normalizer import enhanced_normalizer
from app.services.executor import network_executor
from app.services.inventory import inventory_service

logger = logging.getLogger(__name__)


class IntentExecutor:
    """Execute network intents across multiple vendors with enhanced LLM normalization"""

    def __init__(self):
        self.stats = {
            "total_executions": 0,
            "successful_normalizations": 0,
            "failed_normalizations": 0,
            "manual_normalizations": 0,
            "llm_normalizations": 0,
            "cache_hits": 0,
        }

    async def execute_intent(self, request: IntentRequest) -> IntentResponse:
        """Execute a network intent on multiple devices with enhanced normalization"""
        logger.info(f"Executing intent {request.intent} on devices: {request.devices}")
        self.stats["total_executions"] += 1

        # Get device objects and translate commands
        device_commands = {}
        missing_devices = []
        unsupported_devices = []

        for device_name in request.devices:
            device = inventory_service.get_device(device_name)
            if not device:
                logger.warning(f"Device {device_name} not found in inventory")
                missing_devices.append(device_name)
                continue

            command = command_translator.get_command(request.intent, device)
            if not command:
                logger.warning(f"Intent {request.intent} not supported on {device.platform}")
                unsupported_devices.append(device_name)
                continue

            device_commands[device_name] = command
            logger.info(f"Device {device_name} ({device.platform}): {command}")

        results = []

        # Handle missing devices
        for device_name in missing_devices:
            result = DeviceIntentResult(
                device=device_name,
                intent=request.intent,
                status="failed",
                error_message="Device not found in inventory",
                normalization_method="none",
                confidence_score=0.0,
            )
            results.append(result)

        # Handle unsupported devices
        for device_name in unsupported_devices:
            device = inventory_service.get_device(device_name)
            result = DeviceIntentResult(
                device=device_name,
                intent=request.intent,
                status="failed",
                error_message=f"Intent {request.intent} not supported on platform {device.platform}",
                normalization_method="none",
                confidence_score=0.0,
            )
            results.append(result)

        if not device_commands:
            logger.warning("No devices support this intent")
            return IntentResponse(
                intent=request.intent,
                total_devices=len(request.devices),
                successful=0,
                failed=len(request.devices),
                results=results,
            )

        # Group devices by command to optimize execution
        command_groups = self._group_devices_by_command(device_commands)
        logger.info(f"Executing {len(command_groups)} unique commands")

        # Execute each unique command
        for command, devices in command_groups.items():
            logger.info(f"Executing '{command}' on devices: {devices}")

            try:
                # Use existing network executor
                cmd_response = await network_executor.execute_command(
                    device_names=devices, command=command, timeout=request.timeout
                )

                # Convert command results to intent results with enhanced normalization
                for device_result in cmd_response.results:
                    normalized_output = None
                    normalization_method = "none"
                    confidence_score = 0.0

                    if device_result.status == "success" and device_result.raw_output:
                        device = inventory_service.get_device(device_result.device)
                        try:
                            # Enhanced normalization with LLM support
                            normalization_result = await enhanced_normalizer.normalize_output(
                                raw_output=device_result.raw_output,
                                intent=request.intent,
                                vendor=device.platform,
                                command=command,
                                device_name=device_result.device,
                            )

                            if normalization_result.success:
                                normalized_output = normalization_result.normalized_data
                                normalization_method = normalization_result.method_used
                                confidence_score = normalization_result.confidence_score

                                # Update stats
                                if normalization_method == "manual":
                                    self.stats["manual_normalizations"] += 1
                                elif normalization_method == "llm":
                                    self.stats["llm_normalizations"] += 1
                                elif normalization_method == "cache":
                                    self.stats["cache_hits"] += 1

                                self.stats["successful_normalizations"] += 1

                                logger.info(
                                    f"Successfully normalized output for {device_result.device} "
                                    f"using {normalization_method} (confidence: {confidence_score:.2f})"
                                )
                            else:
                                self.stats["failed_normalizations"] += 1
                                normalization_method = "fallback"
                                confidence_score = 0.1
                                logger.warning(
                                    f"Normalization failed for {device_result.device}: "
                                    f"{normalization_result.error_message}"
                                )

                        except Exception as e:
                            logger.error(f"Normalization error for {device_result.device}: {e}")
                            self.stats["failed_normalizations"] += 1
                            normalization_method = "error"
                            confidence_score = 0.0

                    # Determine overall status based on execution and normalization
                    overall_status = device_result.status
                    if device_result.status == "success" and normalized_output is None:
                        overall_status = "partial"  # Command succeeded but normalization failed

                    intent_result = DeviceIntentResult(
                        device=device_result.device,
                        intent=request.intent,
                        status=overall_status,
                        normalized_output=normalized_output,
                        raw_output=device_result.raw_output
                        if normalized_output is None
                        else None,  # Only include raw if normalized failed
                        vendor_command=command,
                        execution_time=device_result.execution_time,
                        error_message=device_result.error_message,
                        normalization_method=normalization_method,
                        confidence_score=confidence_score,
                    )
                    results.append(intent_result)

            except Exception as e:
                logger.error(f"Error executing command '{command}': {e}")
                # Create failed results for devices in this group
                for device_name in devices:
                    result = DeviceIntentResult(
                        device=device_name,
                        intent=request.intent,
                        status="failed",
                        vendor_command=command,
                        error_message=f"Command execution error: {str(e)}",
                        normalization_method="none",
                        confidence_score=0.0,
                    )
                    results.append(result)

        # Calculate summary
        successful = len([r for r in results if r.status == "success"])
        partial = len([r for r in results if r.status == "partial"])
        failed = len(results) - successful - partial

        logger.info(f"Intent execution completed - Success: {successful}, " f"Partial: {partial}, Failed: {failed}")

        # Log normalization statistics
        if self.stats["total_executions"] % 10 == 0:  # Log every 10 executions
            self._log_stats()

        return IntentResponse(
            intent=request.intent,
            total_devices=len(request.devices),
            successful=successful + partial,  # Count partial as successful for API response
            failed=failed,
            results=results,
        )

    def _group_devices_by_command(self, device_commands: Dict[str, str]) -> Dict[str, List[str]]:
        """Group devices that use the same command for efficiency"""
        command_groups = {}
        for device, command in device_commands.items():
            if command not in command_groups:
                command_groups[command] = []
            command_groups[command].append(device)
        return command_groups

    def _log_stats(self):
        """Log execution and normalization statistics"""
        total_norm_attempts = self.stats["successful_normalizations"] + self.stats["failed_normalizations"]

        if total_norm_attempts > 0:
            success_rate = (self.stats["successful_normalizations"] / total_norm_attempts) * 100
            manual_rate = (self.stats["manual_normalizations"] / total_norm_attempts) * 100
            llm_rate = (self.stats["llm_normalizations"] / total_norm_attempts) * 100
            cache_rate = (self.stats["cache_hits"] / total_norm_attempts) * 100

            logger.info(
                f"Intent Executor Stats - Total: {self.stats['total_executions']}, "
                f"Norm Success: {success_rate:.1f}%, "
                f"Manual: {manual_rate:.1f}%, "
                f"LLM: {llm_rate:.1f}%, "
                f"Cache: {cache_rate:.1f}%"
            )

    def get_stats(self) -> Dict:
        """Get execution statistics"""
        total_norm_attempts = self.stats["successful_normalizations"] + self.stats["failed_normalizations"]

        stats = {
            **self.stats,
            "normalization_success_rate": 0.0,
            "manual_normalization_rate": 0.0,
            "llm_normalization_rate": 0.0,
            "cache_hit_rate": 0.0,
        }

        if total_norm_attempts > 0:
            stats["normalization_success_rate"] = round(
                (self.stats["successful_normalizations"] / total_norm_attempts) * 100, 2
            )
            stats["manual_normalization_rate"] = round(
                (self.stats["manual_normalizations"] / total_norm_attempts) * 100, 2
            )
            stats["llm_normalization_rate"] = round((self.stats["llm_normalizations"] / total_norm_attempts) * 100, 2)
            stats["cache_hit_rate"] = round((self.stats["cache_hits"] / total_norm_attempts) * 100, 2)

        return stats

    def reset_stats(self):
        """Reset all statistics"""
        for key in self.stats:
            self.stats[key] = 0
        logger.info("Intent executor statistics reset")


# Global intent executor
intent_executor = IntentExecutor()

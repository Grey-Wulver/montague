# app/services/intent_executor.py
import logging
from typing import Dict, List

from app.models.intents import DeviceIntentResult, IntentRequest, IntentResponse
from app.services.command_translator import command_translator
from app.services.executor import network_executor
from app.services.inventory import inventory_service
from app.services.output_normalizer import output_normalizer

logger = logging.getLogger(__name__)


class IntentExecutor:
    """Execute network intents across multiple vendors"""

    async def execute_intent(self, request: IntentRequest) -> IntentResponse:
        """Execute a network intent on multiple devices"""
        logger.info(f"Executing intent {request.intent} on devices: {request.devices}")

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

                # Convert command results to intent results with normalization
                for device_result in cmd_response.results:
                    normalized_output = None

                    if device_result.status == "success" and device_result.raw_output:
                        device = inventory_service.get_device(device_result.device)
                        try:
                            # Add normalization step
                            normalized_output = output_normalizer.normalize(
                                intent=request.intent,
                                raw_output=device_result.raw_output,
                                device_platform=device.platform,
                                command=command,
                            )
                            if normalized_output:
                                logger.info(f"Successfully normalized output for {device_result.device}")
                            else:
                                logger.info(f"No normalization available for {device_result.device}")
                        except Exception as e:
                            logger.error(f"Normalization failed for {device_result.device}: {e}")
                            normalized_output = None

                    intent_result = DeviceIntentResult(
                        device=device_result.device,
                        intent=request.intent,
                        status=device_result.status,
                        normalized_output=normalized_output.model_dump() if normalized_output else None,
                        raw_output=device_result.raw_output,
                        vendor_command=command,
                        execution_time=device_result.execution_time,
                        error_message=device_result.error_message,
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
                    )
                    results.append(result)

        # Calculate summary
        successful = len([r for r in results if r.status == "success"])
        failed = len(results) - successful

        logger.info(f"Intent execution completed - Success: {successful}, Failed: {failed}")

        return IntentResponse(
            intent=request.intent,
            total_devices=len(request.devices),
            successful=successful,
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


# Global intent executor
intent_executor = IntentExecutor()

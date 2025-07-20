# app/services/executor.py
import logging
import os
import tempfile
import time
from typing import List

import yaml
from nornir import InitNornir
from nornir_netmiko.tasks import netmiko_send_command

from app.models.commands import CommandResponse, CommandStatus, DeviceCommandResult
from app.models.devices import Device
from app.services.inventory import inventory_service

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkExecutor:
    """Network command execution engine using Nornir"""

    def __init__(self):
        self.nr = None

    def _create_nornir_files(self, devices: List[Device]) -> str:
        """Create temporary Nornir inventory files"""

        # Create temporary directory
        temp_dir = tempfile.mkdtemp()

        # Create hosts.yaml
        hosts_data = {}
        for device in devices:
            hosts_data[device.name] = {
                "hostname": device.hostname,
                "username": device.username,
                "password": device.password,
                "platform": "arista_eos",
                "groups": ["arista"],
                "data": {
                    "device_type": device.device_type.value,  # Use .value to get string from enum
                    "site": device.site,
                    "vendor": device.vendor,
                    "role": device.role,
                },
            }

        hosts_file = os.path.join(temp_dir, "hosts.yaml")
        with open(hosts_file, "w") as f:
            yaml.dump(hosts_data, f, default_flow_style=False)

        # Debug: Print the generated YAML to see what's working now
        logger.info(f"Generated hosts.yaml at: {hosts_file}")
        logger.info("Hosts YAML content:")
        with open(hosts_file) as f:
            logger.info(f.read())

        # Create groups.yaml
        groups_data = {
            "arista": {
                "platform": "arista_eos",
                "connection_options": {
                    "netmiko": {
                        "platform": "arista_eos",
                        "extras": {"global_delay_factor": 2, "timeout": 30},
                    }
                },
            }
        }

        groups_file = os.path.join(temp_dir, "groups.yaml")
        with open(groups_file, "w") as f:
            yaml.dump(groups_data, f)

        # Create config.yaml
        config_data = {
            "inventory": {
                "plugin": "SimpleInventory",
                "options": {"host_file": hosts_file, "group_file": groups_file},
            },
            "runner": {"plugin": "threaded", "options": {"num_workers": 5}},
            "logging": {"enabled": True, "level": "INFO"},
        }

        config_file = os.path.join(temp_dir, "config.yaml")
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        return config_file

    def _initialize_nornir(self, devices: List[Device]):
        """Initialize Nornir with temporary config files"""
        config_file = self._create_nornir_files(devices)
        self.nr = InitNornir(config_file=config_file)

    async def execute_command(
        self, device_names: List[str], command: str, timeout: int = 30
    ) -> CommandResponse:
        """Execute a command on multiple devices"""

        # Get device objects from inventory
        devices = []
        missing_devices = []

        for device_name in device_names:
            device = inventory_service.get_device(device_name)
            if not device:
                logger.warning(f"Device {device_name} not found in inventory")
                missing_devices.append(device_name)
                continue
            devices.append(device)

        results = []

        # Handle missing devices
        for device_name in missing_devices:
            result = DeviceCommandResult(
                device=device_name,
                command=command,
                status=CommandStatus.FAILED,
                error_message="Device not found in inventory",
            )
            results.append(result)

        if not devices:
            return CommandResponse(
                total_devices=len(device_names),
                successful=0,
                failed=len(device_names),
                results=results,
            )

        # Initialize Nornir with our devices
        try:
            self._initialize_nornir(devices)
            logger.info(f"Initialized Nornir with {len(devices)} devices")
        except Exception as e:
            logger.error(f"Failed to initialize Nornir: {e}")
            # Create failed results for all devices
            for device in devices:
                result = DeviceCommandResult(
                    device=device.name,
                    command=command,
                    status=CommandStatus.FAILED,
                    error_message=f"Nornir initialization failed: {str(e)}",
                )
                results.append(result)

            return CommandResponse(
                total_devices=len(device_names),
                successful=0,
                failed=len(device_names),
                results=results,
            )

        # Execute the command
        start_time = time.time()

        try:
            logger.info(f"Executing command: {command}")

            # Run the command using Nornir
            nornir_result = self.nr.run(
                task=netmiko_send_command, command_string=command, read_timeout=timeout
            )

            logger.info("Command execution completed")

            # Process results
            for device_name, result in nornir_result.items():
                execution_time = time.time() - start_time

                if result.failed:
                    error_msg = (
                        str(result.exception) if result.exception else "Unknown error"
                    )
                    logger.error(f"Command failed on {device_name}: {error_msg}")

                    device_result = DeviceCommandResult(
                        device=device_name,
                        command=command,
                        status=CommandStatus.FAILED,
                        error_message=error_msg,
                        execution_time=execution_time,
                    )
                else:
                    logger.info(f"Command succeeded on {device_name}")

                    device_result = DeviceCommandResult(
                        device=device_name,
                        command=command,
                        status=CommandStatus.SUCCESS,
                        raw_output=result.result,
                        execution_time=execution_time,
                    )

                results.append(device_result)

        except Exception as e:
            logger.error(f"Error executing command: {e}")
            # Create failed results for all devices
            for device in devices:
                device_result = DeviceCommandResult(
                    device=device.name,
                    command=command,
                    status=CommandStatus.FAILED,
                    error_message=f"Execution error: {str(e)}",
                )
                results.append(device_result)

        # Calculate summary
        successful = len([r for r in results if r.status == CommandStatus.SUCCESS])
        failed = len(results) - successful

        logger.info(
            f"Command execution summary - Success: {successful}, Failed: {failed}"
        )

        return CommandResponse(
            total_devices=len(device_names),
            successful=successful,
            failed=failed,
            results=results,
        )


# Global executor instance
network_executor = NetworkExecutor()

# core_engine/executor.py
#
# Description:
# This script is the core execution engine for the net-chatbot project.
# It connects to a list of network devices defined in your containerlab
# inventory, executes a specified command, parses the output into structured
# data using Genie, and saves the result to the /results directory.
#
# FINAL FIX: Explicitly set auth method and handle enable secret.

import argparse
import json
import logging
import os

# This will print detailed connection logs from Netmiko to the screen.
import sys

import hvac
from nornir import InitNornir
from nornir_netmiko.tasks import netmiko_send_command
from nornir_utils.plugins.functions import print_result

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # Changed to INFO to be less noisy on success
logger = logging.getLogger("netmiko")


# --- Vault Integration ---
def get_credentials_from_vault(vault_addr, vault_token, mount_point, secret_path):
    """
    Fetches network device credentials from HashiCorp Vault.
    """
    try:
        print("--> Connecting to Vault to fetch credentials...")
        client = hvac.Client(url=vault_addr, token=vault_token)

        if not client.is_authenticated():
            print("!!! Vault authentication failed. Check VAULT_ADDR and VAULT_TOKEN.")
            return None

        response = client.secrets.kv.v2.read_secret_version(
            mount_point=mount_point,
            path=secret_path,
        )

        credentials = response["data"]["data"]
        print("--> Successfully fetched credentials from Vault.")
        return credentials

    except Exception as e:
        print(f"!!! An error occurred while fetching credentials from Vault: {e}")
        return None


# --- Nornir Task ---
def execute_and_parse_command(task, command_to_run, enable_secret):
    """
    Nornir task to send a command to a device and parse it with Genie.
    """
    print(f"-----> [{task.host.name}] Attempting to connect and execute...")
    try:
        result = task.run(
            task=netmiko_send_command,
            command_string=command_to_run,
            use_genie=True,
            enable=True,
            # ==================================================================
            # START: FIX
            # ==================================================================
            auth_method="keyboard-interactive",
            secret=enable_secret,
            # ==================================================================
            # END: FIX
            # ==================================================================
        )
        return result
    except Exception:
        return None


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Setup Argument Parser ---
    parser = argparse.ArgumentParser(description="Execute a command on network devices.")
    parser.add_argument(
        "--command",
        type=str,
        required=True,
        help="The command to execute on the devices.",
    )
    args = parser.parse_args()

    # --- Configuration ---
    VAULT_ADDR = os.getenv("VAULT_ADDR", "http://127.0.0.1:8200")
    VAULT_TOKEN = os.getenv("VAULT_TOKEN")
    VAULT_SECRET_MOUNT_POINT = "secret"
    VAULT_SECRET_PATH = "net_creds"

    # --- Fetch Credentials ---
    if not VAULT_TOKEN:
        print("!!! Environment variable VAULT_TOKEN is not set. Exiting.")
        exit(1)

    credentials = get_credentials_from_vault(
        vault_addr=VAULT_ADDR,
        vault_token=VAULT_TOKEN,
        mount_point=VAULT_SECRET_MOUNT_POINT,
        secret_path=VAULT_SECRET_PATH,
    )

    if not credentials:
        print("!!! Could not retrieve credentials from Vault. Exiting.")
        exit(1)

    # ==========================================================================
    # START: FIX - Get the enable secret from credentials
    # ==========================================================================
    enable_secret = credentials.get("enable_secret", "")  # Defaults to empty string
    # ==========================================================================
    # END: FIX
    # ==========================================================================

    # --- Initialize Nornir ---
    nr = InitNornir(config_file="config.yaml")
    print(f"--> Nornir initialized. Inventory has {len(nr.inventory.hosts)} hosts.")

    # --- Update Inventory with Credentials ---
    nr.inventory.defaults.username = credentials.get("username")
    nr.inventory.defaults.password = credentials.get("password")

    # --- Run the Nornir Task ---
    print(f"\n--> Running task for command: '{args.command}'")
    result = nr.run(
        name=f"Execute '{args.command}' and parse with Genie",
        task=execute_and_parse_command,
        command_to_run=args.command,
        enable_secret=enable_secret,  # Pass the secret to the task
    )

    # --- Process and Output Results ---
    print("\n--> Task execution summary:")
    print_result(result)  # Re-enabled print_result for clean output

    structured_output = {}
    for host, multi_result in result.items():
        if not multi_result.failed and multi_result[0].result is not None:
            structured_output[host] = multi_result[0].result
        else:
            structured_output[host] = {
                "error": f"Task failed for host {host}. Exception: {multi_result.exception}"
            }

    # --- Save Output ---
    output_filename = "results/structured_output.json"
    print(f"\n--> Saving structured output to '{output_filename}'...")
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(structured_output, f, indent=4)

    print("--> Script finished successfully.")

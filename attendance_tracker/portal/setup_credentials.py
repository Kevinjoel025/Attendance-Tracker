"""
portal/setup_credentials.py
───────────────────────────
Interactive CLI utility to securely save VNR VJIET portal credentials in Windows Credential Manager.

Usage:
    python -m attendance_tracker.portal.setup_credentials
"""

from __future__ import annotations

import getpass
import sys

from attendance_tracker.core.config import config
from attendance_tracker.portal.credentials import (
    get_credential_service_name,
    get_default_username,
    has_portal_credentials,
    set_portal_credentials,
)


def run_setup() -> None:
    print("=" * 60)
    print(" VNR VJIET Student Portal Credential Setup")
    print("=" * 60)
    print(f"Keyring Service: {get_credential_service_name()}")
    default_user = get_default_username()
    print(f"Default Student ID: {default_user}")
    print()

    if has_portal_credentials(default_user):
        print(f"[*] Note: Credentials are ALREADY configured for {default_user}.")
        overwrite = input("Do you want to overwrite them? (y/N): ").strip().lower()
        if overwrite not in ("y", "yes"):
            print("Setup cancelled. Existing credentials retained.")
            return

    username_input = input(f"Enter Student ID [{default_user}]: ").strip()
    username = username_input if username_input else default_user

    password = getpass.getpass("Enter Portal Password: ").strip()
    if not password:
        print("[!] Error: Password cannot be empty.")
        sys.exit(1)

    confirm_password = getpass.getpass("Confirm Portal Password: ").strip()
    if password != confirm_password:
        print("[!] Error: Passwords do not match.")
        sys.exit(1)

    set_portal_credentials(username, password)
    print()
    print(f"[+] Credentials for '{username}' successfully saved to Windows Credential Manager!")
    print("    You can now perform live attendance syncs with 'python -m attendance_tracker.cli sync'.")


if __name__ == "__main__":
    run_setup()

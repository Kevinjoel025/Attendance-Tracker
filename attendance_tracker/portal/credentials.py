"""
portal/credentials.py
─────────────────────
Secure credential management via Windows Credential Manager / keyring.

SECURITY RULES:
- Credentials are NEVER hardcoded in source code, logs, or stored in SQLite.
- Credentials are NEVER printed to stdout/stderr.
- Passwords are only retrieved in memory when initiating authentication.
"""

from __future__ import annotations

from typing import Optional, Tuple
import keyring
from keyring.errors import PasswordDeleteError

from attendance_tracker.core.config import config
from attendance_tracker.core.logging_setup import get_logger
from attendance_tracker.portal.exceptions import CredentialNotFoundError

logger = get_logger(__name__)


def get_credential_service_name() -> str:
    """Return the configured keyring service name."""
    return config.portal.credential_service


def get_default_username() -> str:
    """Return the configured default portal username (Student ID)."""
    return config.portal.username


def has_portal_credentials(username: Optional[str] = None) -> bool:
    """
    Check if credentials exist in keyring for the specified username.
    Does not log or expose the password.
    """
    user = username or get_default_username()
    service = get_credential_service_name()
    try:
        pwd = keyring.get_password(service, user)
        return bool(pwd and len(pwd.strip()) > 0)
    except Exception as e:
        logger.warning(f"Error checking credential store for user '{user}': {e}")
        return False


def get_portal_credentials(username: Optional[str] = None) -> tuple[str, str]:
    """
    Retrieve username and password from keyring.
    
    Raises
    ------
    CredentialNotFoundError
        If no credentials are saved in the keyring.
    """
    user = username or get_default_username()
    service = get_credential_service_name()

    try:
        pwd = keyring.get_password(service, user)
    except Exception as e:
        raise CredentialNotFoundError(
            f"Failed to access secure credential store for user '{user}': {e}"
        ) from e

    if not pwd or not pwd.strip():
        raise CredentialNotFoundError(
            f"No portal credentials found in keyring for user '{user}' under service '{service}'. "
            f"Please run 'python -m attendance_tracker.portal.setup_credentials' to configure."
        )

    return user, pwd


def set_portal_credentials(username: str, password: str) -> None:
    """
    Store username and password securely in Windows Credential Manager via keyring.
    """
    if not username or not username.strip():
        raise ValueError("Username cannot be empty.")
    if not password or not password.strip():
        raise ValueError("Password cannot be empty.")

    user = username.strip()
    service = get_credential_service_name()
    keyring.set_password(service, user, password.strip())
    logger.info(f"Successfully stored credentials for user '{user}' in service '{service}'.")


def delete_portal_credentials(username: Optional[str] = None) -> bool:
    """
    Remove stored portal credentials from keyring.
    Returns True if removed or not found, False on error.
    """
    user = username or get_default_username()
    service = get_credential_service_name()
    try:
        keyring.delete_password(service, user)
        logger.info(f"Removed credentials for user '{user}' from service '{service}'.")
        return True
    except PasswordDeleteError:
        logger.info(f"No credentials existed for user '{user}' in service '{service}'.")
        return True
    except Exception as e:
        logger.error(f"Error deleting credentials for user '{user}': {e}")
        return False

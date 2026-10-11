"""
portal/exceptions.py
────────────────────
Domain exceptions for the VNR VJIET portal integration.
"""

from __future__ import annotations


class PortalError(Exception):
    """Base exception for all portal-related failures."""
    pass


class CredentialNotFoundError(PortalError):
    """Raised when required portal credentials are not found in the credential store."""
    pass


class PortalAuthError(PortalError):
    """Raised when portal authentication fails (invalid credentials, lockout, CAPTCHA)."""
    pass


class PortalNavigationError(PortalError):
    """Raised when portal pages fail to load or timeout."""
    pass


class PortalScrapeError(PortalError):
    """Raised when attendance tables or data cannot be located or parsed from the portal HTML."""
    pass


class PortalValidationError(PortalError):
    """Raised when scraped attendance data violates invariants (e.g. attended > conducted)."""
    pass

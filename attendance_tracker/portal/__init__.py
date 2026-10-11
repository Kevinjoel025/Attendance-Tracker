"""
portal package
──────────────
VNR VJIET portal integration module.
"""

from attendance_tracker.portal.auth import PortalAuthenticator
from attendance_tracker.portal.attendance import AttendanceParser, PortalAttendanceScraper
from attendance_tracker.portal.credentials import (
    delete_portal_credentials,
    get_credential_service_name,
    get_default_username,
    get_portal_credentials,
    has_portal_credentials,
    set_portal_credentials,
)
from attendance_tracker.portal.exceptions import (
    CredentialNotFoundError,
    PortalAuthError,
    PortalError,
    PortalNavigationError,
    PortalScrapeError,
    PortalValidationError,
)
from attendance_tracker.portal.models import (
    PortalSyncResult,
    RawSubjectRecord,
    ScrapedAttendanceData,
)
from attendance_tracker.portal.service import (
    PortalAttendanceService,
    get_portal_service,
)

__all__ = [
    "PortalAuthenticator",
    "AttendanceParser",
    "PortalAttendanceScraper",
    "PortalAttendanceService",
    "get_portal_service",
    "PortalError",
    "PortalAuthError",
    "PortalNavigationError",
    "PortalScrapeError",
    "PortalValidationError",
    "CredentialNotFoundError",
    "PortalSyncResult",
    "RawSubjectRecord",
    "ScrapedAttendanceData",
    "get_portal_credentials",
    "set_portal_credentials",
    "delete_portal_credentials",
    "has_portal_credentials",
    "get_credential_service_name",
    "get_default_username",
]

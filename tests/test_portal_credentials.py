"""
tests/test_portal_credentials.py
─────────────────────────────────
Unit tests for secure portal credential handling via keyring.
"""

from unittest.mock import patch
import pytest

from attendance_tracker.portal.credentials import (
    delete_portal_credentials,
    get_credential_service_name,
    get_default_username,
    get_portal_credentials,
    has_portal_credentials,
    set_portal_credentials,
)
from attendance_tracker.portal.exceptions import CredentialNotFoundError


class MockKeyring:
    def __init__(self):
        self.vault = {}

    def get_password(self, service, username):
        return self.vault.get((service, username))

    def set_password(self, service, username, password):
        self.vault[(service, username)] = password

    def delete_password(self, service, username):
        key = (service, username)
        if key in self.vault:
            del self.vault[key]
        else:
            import keyring.errors
            raise keyring.errors.PasswordDeleteError("Not found")


def test_credential_service_name_and_default_user():
    assert get_credential_service_name() == "attendance_tracker_portal"
    assert get_default_username() == "24071A6947"


def test_get_credentials_success():
    mock_vault = MockKeyring()
    mock_vault.set_password("attendance_tracker_portal", "24071A6947", "SecretPass123!")

    with patch("attendance_tracker.portal.credentials.keyring", mock_vault):
        user, pwd = get_portal_credentials("24071A6947")
        assert user == "24071A6947"
        assert pwd == "SecretPass123!"
        assert has_portal_credentials("24071A6947") is True


def test_get_credentials_missing_raises_exception():
    mock_vault = MockKeyring()

    with patch("attendance_tracker.portal.credentials.keyring", mock_vault):
        assert has_portal_credentials("24071A6947") is False
        with pytest.raises(CredentialNotFoundError) as exc_info:
            get_portal_credentials("24071A6947")
        assert "No portal credentials found" in str(exc_info.value)


def test_set_credentials_validations():
    mock_vault = MockKeyring()

    with patch("attendance_tracker.portal.credentials.keyring", mock_vault):
        set_portal_credentials("24071A6947", "NewPassword456!")
        assert mock_vault.get_password("attendance_tracker_portal", "24071A6947") == "NewPassword456!"

        with pytest.raises(ValueError):
            set_portal_credentials("", "pass")

        with pytest.raises(ValueError):
            set_portal_credentials("24071A6947", "  ")


def test_delete_credentials():
    mock_vault = MockKeyring()
    mock_vault.set_password("attendance_tracker_portal", "24071A6947", "TempPass")

    with patch("attendance_tracker.portal.credentials.keyring", mock_vault):
        assert delete_portal_credentials("24071A6947") is True
        assert mock_vault.get_password("attendance_tracker_portal", "24071A6947") is None
        # Deleting again should be safe and return True
        assert delete_portal_credentials("24071A6947") is True

"""
tests/test_portal_auth.py
─────────────────────────
Unit tests for single-attempt portal authentication.
"""

from unittest.mock import MagicMock, patch
import pytest

from attendance_tracker.portal.auth import PortalAuthenticator
from attendance_tracker.portal.exceptions import (
    PortalAuthError,
    PortalNavigationError,
)


class MockElement:
    def __init__(self, visible=True, text=""):
        self._visible = visible
        self._text = text
        self.filled_value = None

    def fill(self, value):
        self.filled_value = value

    def click(self):
        pass

    def is_visible(self):
        return self._visible

    def inner_text(self):
        return self._text


class MockPage:
    def __init__(self, initial_url="https://automation.vnrvjiet.ac.in", post_url="https://automation.vnrvjiet.ac.in/App"):
        self.url = initial_url
        self.post_url = post_url
        self._response_callbacks = []
        self.user_input = MockElement(visible=True)
        self.pass_input = MockElement(visible=True)
        self.submit_btn = MockElement(visible=True)
        self.login_msg = MockElement(visible=False, text="")
        self.login_challenge = MockElement(visible=False)
        self.clicked = False

    def goto(self, url, timeout=30000, wait_until="domcontentloaded"):
        self.url = url

    def wait_for_selector(self, selector, timeout=10000, state="visible"):
        if selector == "#UserName":
            return self.user_input
        if selector == "#xPassword":
            return self.pass_input
        if selector == "#btnSubmit":
            return self.submit_btn
        raise Exception(f"Unknown selector {selector}")

    def on(self, event, callback):
        if event == "response":
            self._response_callbacks.append(callback)

    def wait_for_timeout(self, ms):
        # Simulate transition on submit
        if self.clicked:
            self.url = self.post_url

    def query_selector(self, selector):
        if selector == "#frm-login":
            return MockElement(visible=True) if "/App" not in self.url else None
        if selector == "#loginMessage":
            return self.login_msg
        if selector == "#loginChallenge":
            return self.login_challenge
        if selector == ".standard-login__head":
            return MockElement(visible=True) if "/App" not in self.url else None
        return None

    def wait_for_url(self, pattern, timeout=10000):
        pass


def test_login_success_redirect():
    page = MockPage()
    authenticator = PortalAuthenticator()

    def submit_action():
        page.clicked = True
        page.url = "https://automation.vnrvjiet.ac.in/App"

    page.submit_btn.click = submit_action

    success = authenticator.login(page, username="24071A6947", password="ValidPassword123")
    assert success is True
    assert page.user_input.filled_value == "24071A6947"
    assert page.pass_input.filled_value == "ValidPassword123"


def test_login_failure_invalid_credentials():
    page = MockPage(post_url="https://automation.vnrvjiet.ac.in/")
    page.login_msg = MockElement(visible=True, text="Invalid Username or Password.")
    authenticator = PortalAuthenticator()

    with pytest.raises(PortalAuthError) as exc_info:
        authenticator.login(page, username="24071A6947", password="WrongPassword")

    assert "Invalid Username or Password" in str(exc_info.value)


def test_login_failure_challenge_triggered():
    page = MockPage(post_url="https://automation.vnrvjiet.ac.in/")
    page.login_challenge = MockElement(visible=True)
    authenticator = PortalAuthenticator()

    with pytest.raises(PortalAuthError) as exc_info:
        authenticator.login(page, username="24071A6947", password="SomePassword")

    assert "CAPTCHA or DOB required" in str(exc_info.value)


def test_login_navigation_timeout():
    page = MockPage()
    def fail_goto(*args, **kwargs):
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        raise PlaywrightTimeoutError("Connection timed out")

    page.goto = fail_goto
    authenticator = PortalAuthenticator()

    with pytest.raises(PortalNavigationError) as exc_info:
        authenticator.login(page, username="24071A6947", password="SomePassword")

    assert "Timeout navigating" in str(exc_info.value)

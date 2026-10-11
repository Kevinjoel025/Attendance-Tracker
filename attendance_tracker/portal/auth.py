"""
portal/auth.py
──────────────
Secure authentication handler for VNR VJIET portal (https://automation.vnrvjiet.ac.in).

CRITICAL SAFETY CONSTRAINTS:
- EXACTLY ONE login attempt per execution.
- NEVER implement retry loops or brute-force attempts to avoid account lockout.
- NEVER log passwords, cookies, or session tokens.
- On authentication failure or CAPTCHA prompt, abort immediately with PortalAuthError.
"""

from __future__ import annotations

import json
from typing import Optional, Tuple
from playwright.sync_api import Page, Response, TimeoutError as PlaywrightTimeoutError

from attendance_tracker.core.config import config
from attendance_tracker.core.logging_setup import get_logger
from attendance_tracker.portal.credentials import get_portal_credentials
from attendance_tracker.portal.exceptions import (
    PortalAuthError,
    PortalNavigationError,
)

logger = get_logger(__name__)


class PortalAuthenticator:
    """
    Handles single-attempt automated login to the VNR VJIET student portal.
    """

    def __init__(
        self,
        portal_url: Optional[str] = None,
        timeout_ms: int = 30000,
    ) -> None:
        self.portal_url = portal_url or config.portal.url
        self.timeout_ms = timeout_ms

    def login(
        self,
        page: Page,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> bool:
        """
        Perform a single login attempt on the portal.

        Parameters
        ----------
        page : Page
            Active Playwright page.
        username : Optional[str]
            Student username (defaults to credentials in keyring / config).
        password : Optional[str]
            Student portal password (retrieved from keyring).

        Returns
        -------
        bool
            True if login succeeded and user is authenticated.

        Raises
        ------
        PortalAuthError
            If authentication failed (invalid credentials, CAPTCHA challenge, lockout).
        PortalNavigationError
            If network / page navigation timed out.
        """
        if username is None or password is None:
            user, pwd = get_portal_credentials(username)
        else:
            user, pwd = username, password

        logger.info(f"Initiating single login attempt for student ID '{user}' at {self.portal_url}...")

        # 1. Navigate to portal home/login page
        try:
            page.goto(self.portal_url, timeout=self.timeout_ms, wait_until="domcontentloaded")
        except PlaywrightTimeoutError as e:
            msg = f"Timeout navigating to portal URL: {self.portal_url}"
            logger.error(msg)
            raise PortalNavigationError(msg) from e
        except Exception as e:
            msg = f"Failed to connect to portal URL {self.portal_url}: {e}"
            logger.error(msg)
            raise PortalNavigationError(msg) from e

        # Check if already authenticated (e.g. if session is active)
        if "/App" in page.url or page.query_selector("#frm-login") is None:
            if page.query_selector(".standard-login__head") is None:
                logger.info("Session already active and authenticated.")
                return True

        # 2. Locate form elements
        try:
            user_input = page.wait_for_selector("#UserName", timeout=10000, state="visible")
            pass_input = page.wait_for_selector("#xPassword", timeout=10000, state="visible")
            submit_btn = page.wait_for_selector("#btnSubmit", timeout=10000, state="visible")
        except PlaywrightTimeoutError as e:
            msg = "Login form inputs (#UserName, #xPassword, #btnSubmit) not found or not visible."
            logger.error(msg)
            raise PortalNavigationError(msg) from e

        # 3. Listen for /Login response or redirection
        login_response_data: dict = {}

        def handle_response(resp: Response) -> None:
            nonlocal login_response_data
            if "/Login" in resp.url and resp.request.method.upper() == "POST":
                try:
                    login_response_data = resp.json()
                except Exception:
                    pass

        page.on("response", handle_response)

        # 4. Fill credentials and submit — STRICTLY ONE ATTEMPT
        try:
            user_input.fill(user)
            pass_input.fill(pwd)
            submit_btn.click()
        except Exception as e:
            msg = f"Failed to fill or submit login form: {e}"
            logger.error(msg)
            raise PortalNavigationError(msg) from e

        # 5. Wait for result (redirect to /App or /App/Party, or AJAX response, or alert message)
        try:
            # Wait for either URL change to /App or presence of error message
            for _ in range(50):  # Poll within 10 seconds (50 * 200ms)
                page.wait_for_timeout(200)

                # Check URL redirect
                if "/App" in page.url:
                    logger.info("Authentication successful. Redirected to workspace.")
                    return True

                # Check captured JSON response
                if login_response_data:
                    status = login_response_data.get("Status")
                    if status == 1:
                        # Wait for page navigation to complete
                        try:
                            page.wait_for_url("**/App**", timeout=10000)
                        except Exception:
                            pass
                        logger.info("Authentication successful (Status 1).")
                        return True
                    else:
                        # Authentication failed according to portal API
                        data = login_response_data.get("Data", {})
                        if isinstance(data, dict):
                            err_msg = data.get("Message") or data.get("message") or "Invalid credentials."
                            if data.get("BlockedUntil"):
                                err_msg += f" (Account blocked until {data['BlockedUntil']})"
                            if data.get("CaptchaRequired"):
                                err_msg += " (CAPTCHA verification triggered)"
                        else:
                            err_msg = str(data) or "Invalid username or password."
                        
                        logger.error(f"Portal authentication rejected: {err_msg}")
                        raise PortalAuthError(f"Portal login failed: {err_msg}")

                # Check on-page error message element
                msg_el = page.query_selector("#loginMessage")
                if msg_el and msg_el.is_visible():
                    err_text = msg_el.inner_text().strip()
                    if err_text:
                        logger.error(f"Portal login error message displayed: {err_text}")
                        raise PortalAuthError(f"Portal login failed: {err_text}")

                # Check if CAPTCHA challenge is visible
                challenge_el = page.query_selector("#loginChallenge")
                if challenge_el and challenge_el.is_visible():
                    msg = "Portal login challenged: CAPTCHA or DOB required. Aborting to avoid lockout."
                    logger.error(msg)
                    raise PortalAuthError(msg)

        except (PortalAuthError, PortalNavigationError):
            raise
        except Exception as e:
            msg = f"Unexpected error during login verification: {e}"
            logger.error(msg)
            raise PortalAuthError(msg) from e

        # Final check if redirect occurred
        if "/App" in page.url:
            logger.info("Authentication verified via URL.")
            return True

        # If we reached here without success, report failure
        raise PortalAuthError("Portal login did not succeed within the timeout window.")

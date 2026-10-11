"""
portal/attendance.py
────────────────────
HTML parsing, DOM extraction, and subject resolution for portal attendance tables.

VERIFIED VNR VJIET PORTAL ARCHITECTURE:
- Dashboard URL: https://automation.vnrvjiet.ac.in/App
- Action tile: #attp (.student-action-widget__item--attendance)
- Modal dialog: #dynamicModal
- Attendance table: #dynamicModal table
- Cell structure: <th>SUBJECT_CODE</th> <td>MONTHLY</td> <td>CUMULATIVE</td>
- The scraper extracts CUMULATIVE attendance (e.g. 22 / 33 -> attended=22, conducted=33).
- Validates exactly 9 canonical subjects, no duplicates, and Total row consistency.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional, Tuple
from bs4 import BeautifulSoup
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError

from attendance_tracker.attendance.engine import current_percentage
from attendance_tracker.core.logging_setup import get_logger
from attendance_tracker.core.subject_resolver import SubjectResolver, get_subject_resolver
from attendance_tracker.portal.exceptions import (
    PortalNavigationError,
    PortalScrapeError,
    PortalValidationError,
)
from attendance_tracker.portal.models import RawSubjectRecord, ScrapedAttendanceData

logger = get_logger(__name__)

# Regex to match attendance ratios like "22 / 33", "22/33", " 22 / 33 "
_RATIO_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)")


class AttendanceParser:
    """
    Parses attendance HTML tables into validated RawSubjectRecord objects.
    """

    def __init__(
        self,
        resolver: Optional[SubjectResolver] = None,
        expected_subject_count: int = 9,
    ) -> None:
        self.resolver = resolver or get_subject_resolver()
        self.expected_subject_count = expected_subject_count

    def parse_html(
        self,
        html_content: str,
        scraped_at: Optional[datetime] = None,
        strict_count: bool = True,
    ) -> ScrapedAttendanceData:
        """
        Parse raw HTML string containing the attendance table.

        Parameters
        ----------
        html_content : str
            Full or partial HTML of the attendance modal/view.
        scraped_at : Optional[datetime]
            Timestamp of the scrape (defaults to datetime.now()).
        strict_count : bool
            Whether to strictly enforce the expected canonical subject count (default True).

        Returns
        -------
        ScrapedAttendanceData
            Container of extracted and resolved records.
        """
        if not html_content or not html_content.strip():
            raise PortalScrapeError("Empty HTML content provided for attendance parsing.")

        timestamp = scraped_at or datetime.now()
        soup = BeautifulSoup(html_content, "html.parser")

        tables = soup.find_all("table")
        if not tables:
            raise PortalScrapeError("No <table> elements found in attendance page HTML.")

        extracted_records: list[RawSubjectRecord] = []
        portal_total_attended: Optional[int] = None
        portal_total_conducted: Optional[int] = None

        for table in tables:
            rows = table.find_all("tr")
            if not rows or len(rows) < 2:
                continue

            # Header inspection
            header_row = rows[0]
            header_cells = header_row.find_all(["th", "td"])
            headers = [c.get_text(strip=True).lower() for c in header_cells]

            # Detect column layout
            cumulative_col = -1
            monthly_col = -1
            attended_col = -1
            conducted_col = -1
            percentage_col = -1
            subject_col = 0

            for idx, h in enumerate(headers):
                if "cumulative" in h or "overall" in h:
                    cumulative_col = idx
                elif any(m in h for m in ["monthly", "month", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]):
                    monthly_col = idx
                elif any(k in h for k in ["classes attended", "attended", "present"]):
                    if attended_col == -1:
                        attended_col = idx
                elif any(k in h for k in ["classes conducted", "classes held", "conducted", "held", "total classes"]):
                    if conducted_col == -1:
                        conducted_col = idx
                elif any(k in h for k in ["%", "percent", "percentage"]):
                    if percentage_col == -1:
                        percentage_col = idx
                elif any(k in h for k in ["subject", "course", "paper", "title", "code", "name"]):
                    subject_col = idx

            # If headers include "Cumulative", default to the cumulative column index
            # In VNR VJIET portal: Row 0 has ["", "October, 2026", "Cumulative"]
            if cumulative_col == -1 and len(headers) >= 3 and any("cumulative" in h for h in headers):
                cumulative_col = 2

            is_ratio_table = (cumulative_col != -1 or monthly_col != -1)

            for row in rows[1:]:
                all_cells = row.find_all(["th", "td"])
                all_text_cells = [c.get_text(strip=True) for c in all_cells]

                if not all_text_cells or len(all_text_cells) < 2:
                    continue

                first_cell_text = all_text_cells[0]

                # Check for Total / Summary row
                if first_cell_text.lower() == "total" or any("total" in c.lower() for c in all_text_cells[:2]):
                    # Parse CUMULATIVE portal reported total for consistency verification
                    target_total_cell = ""
                    if cumulative_col != -1 and cumulative_col < len(all_text_cells):
                        target_total_cell = all_text_cells[cumulative_col]
                    else:
                        ratio_candidates = [c for c in all_text_cells[1:] if _RATIO_RE.search(c)]
                        if ratio_candidates:
                            target_total_cell = ratio_candidates[-1]

                    if target_total_cell:
                        match = _RATIO_RE.search(target_total_cell)
                        if match:
                            try:
                                portal_total_attended = int(match.group(1))
                                portal_total_conducted = int(match.group(2))
                            except ValueError:
                                pass
                    continue

                subject_name_raw = first_cell_text
                att_val: Optional[int] = None
                cond_val: Optional[int] = None
                pct_val: Optional[float] = None

                if is_ratio_table or any("/" in c for c in all_text_cells[1:]):
                    # Format: <th>22PC1IT202</th> <td>1 / 1</td> <td>22 / 33</td>
                    target_ratio_cell = ""
                    if cumulative_col != -1 and cumulative_col < len(all_text_cells):
                        target_ratio_cell = all_text_cells[cumulative_col]
                    else:
                        ratio_candidates = [c for c in all_text_cells[1:] if _RATIO_RE.search(c)]
                        if ratio_candidates:
                            target_ratio_cell = ratio_candidates[-1]

                    if not target_ratio_cell:
                        raise PortalValidationError(
                            f"Malformed attendance value for subject '{subject_name_raw}': "
                            f"No valid ratio 'X / Y' found in row {all_text_cells}."
                        )

                    ratio_match = _RATIO_RE.search(target_ratio_cell)
                    if not ratio_match:
                        raise PortalValidationError(
                            f"Malformed attendance ratio '{target_ratio_cell}' for subject '{subject_name_raw}'."
                        )

                    try:
                        att_val = int(ratio_match.group(1))
                        cond_val = int(ratio_match.group(2))
                    except ValueError as e:
                        raise PortalValidationError(
                            f"Invalid numeric values in ratio '{target_ratio_cell}' for '{subject_name_raw}': {e}"
                        ) from e

                elif attended_col != -1 and conducted_col != -1:
                    # Separate column format
                    if len(all_text_cells) > max(attended_col, conducted_col):
                        att_clean = re.sub(r"[^\d]", "", all_text_cells[attended_col])
                        cond_clean = re.sub(r"[^\d]", "", all_text_cells[conducted_col])
                        if not att_clean.isdigit() or not cond_clean.isdigit():
                            raise PortalValidationError(
                                f"Non-numeric values in row for '{subject_name_raw}': "
                                f"attended='{all_text_cells[attended_col]}', conducted='{all_text_cells[conducted_col]}'"
                            )
                        att_val = int(att_clean)
                        cond_val = int(cond_clean)

                        if percentage_col != -1 and percentage_col < len(all_text_cells):
                            try:
                                pct_clean = re.sub(r"[^\d.]", "", all_text_cells[percentage_col])
                                pct_val = float(pct_clean)
                            except ValueError:
                                pass
                else:
                    # Generic fallback: look for ratio in any cell
                    for c in reversed(all_text_cells[1:]):
                        m = _RATIO_RE.search(c)
                        if m:
                            att_val = int(m.group(1))
                            cond_val = int(m.group(2))
                            break

                if att_val is None or cond_val is None:
                    raise PortalValidationError(
                        f"Could not parse attendance values for subject '{subject_name_raw}' from row: {all_text_cells}"
                    )

                # Strict Invariant Validations
                if att_val < 0 or cond_val < 0:
                    raise PortalValidationError(
                        f"Negative counts for '{subject_name_raw}': attended={att_val}, conducted={cond_val}"
                    )
                if att_val > cond_val:
                    raise PortalValidationError(
                        f"Attended exceeds conducted for '{subject_name_raw}': attended={att_val}, conducted={cond_val}"
                    )

                # Canonical Subject Resolution
                resolved = self.resolver.resolve(subject_name_raw)
                if resolved is None:
                    raise PortalScrapeError(
                        f"Unrecognized subject '{subject_name_raw}' extracted from attendance table."
                    )

                record = RawSubjectRecord(
                    subject_name_raw=subject_name_raw,
                    attended=att_val,
                    conducted=cond_val,
                    percentage_raw=pct_val or current_percentage(att_val, cond_val),
                    subject_code=resolved.code,
                )
                extracted_records.append(record)

        if not extracted_records:
            raise PortalScrapeError("Could not extract any valid attendance records from portal HTML.")

        # Validation: Verify canonical subject count & no duplicates
        unique_codes = {r.subject_code for r in extracted_records}
        if len(extracted_records) != len(unique_codes):
            raise PortalValidationError(
                f"Duplicate subject entries found in attendance table: {len(extracted_records)} rows for {len(unique_codes)} unique subjects."
            )

        if strict_count:
            if len(unique_codes) < self.expected_subject_count:
                raise PortalScrapeError(
                    f"Expected {self.expected_subject_count} canonical subjects, but extracted {len(unique_codes)}: {unique_codes}"
                )
            if len(unique_codes) > self.expected_subject_count:
                raise PortalScrapeError(
                    f"Unexpected extra subjects extracted: expected {self.expected_subject_count}, got {len(unique_codes)}: {unique_codes}"
                )

        # Validation: Consistency check with portal Total row
        if portal_total_attended is not None and portal_total_conducted is not None:
            sum_att = sum(r.attended for r in extracted_records)
            sum_cond = sum(r.conducted for r in extracted_records)
            if sum_att != portal_total_attended or sum_cond != portal_total_conducted:
                raise PortalValidationError(
                    f"Portal reported Total ({portal_total_attended}/{portal_total_conducted}) "
                    f"does not match sum of extracted subjects ({sum_att}/{sum_cond})."
                )
            logger.info(
                f"Total consistency check passed: sum={sum_att}/{sum_cond} matches portal total."
            )

        return ScrapedAttendanceData(
            scraped_at=timestamp,
            source="portal",
            records=extracted_records,
            raw_html_snippet=html_content[:2000],
            metadata={
                "records_found": len(extracted_records),
                "portal_total": f"{portal_total_attended}/{portal_total_conducted}" if portal_total_attended is not None else None,
            },
        )


class PortalAttendanceScraper:
    """
    Automates navigation inside the portal to find and extract attendance data.
    """

    def __init__(
        self,
        parser: Optional[AttendanceParser] = None,
        resolver: Optional[SubjectResolver] = None,
        timeout_ms: int = 30000,
    ) -> None:
        self.resolver = resolver or get_subject_resolver()
        self.parser = parser or AttendanceParser(resolver=self.resolver)
        self.timeout_ms = timeout_ms

    def scrape_attendance(self, page: Page) -> ScrapedAttendanceData:
        """
        Navigate to the student attendance view and extract current attendance tables.

        VERIFIED NAVIGATION FLOW:
        1. On /App, wait for #attp (.student-action-widget__item--attendance).
        2. Click #attp.
        3. Wait for #dynamicModal to become visible.
        4. Wait for table rows to be rendered inside #dynamicModal.
        5. Extract HTML and parse cumulative attendance.
        """
        logger.info("Locating attendance action tile (#attp) on /App...")

        # 1. Wait for the Attendance action tile on /App
        try:
            att_tile = page.wait_for_selector(
                "#attp, .student-action-widget__item--attendance",
                timeout=self.timeout_ms,
                state="visible",
            )
        except PlaywrightTimeoutError as e:
            msg = f"Timed out waiting for attendance action tile (#attp) on page: {page.url}"
            logger.error(msg)
            raise PortalNavigationError(msg) from e

        if not att_tile:
            msg = f"Attendance action tile (#attp) not found on page: {page.url}"
            logger.error(msg)
            raise PortalNavigationError(msg)

        logger.info("Found #attp tile. Clicking to open attendance modal...")

        # 2. Click the attendance action tile
        try:
            att_tile.click()
        except Exception as e:
            msg = f"Failed to click #attp attendance action tile: {e}"
            logger.error(msg)
            raise PortalNavigationError(msg) from e

        # 3. Wait for #dynamicModal to open
        logger.info("Waiting for #dynamicModal to become visible...")
        try:
            modal = page.wait_for_selector(
                "#dynamicModal",
                timeout=self.timeout_ms,
                state="visible",
            )
        except PlaywrightTimeoutError as e:
            msg = "Timed out waiting for attendance modal (#dynamicModal) to open."
            logger.error(msg)
            raise PortalNavigationError(msg) from e

        # 4. Wait for the table rows inside #dynamicModal to be populated via AJAX
        logger.info("Waiting for attendance table rows inside #dynamicModal...")
        try:
            page.wait_for_function(
                "() => document.querySelectorAll('#dynamicModal table tr').length >= 2",
                timeout=self.timeout_ms,
            )
        except PlaywrightTimeoutError as e:
            msg = "Timed out waiting for attendance table rows to populate inside #dynamicModal."
            logger.error(msg)
            raise PortalNavigationError(msg) from e

        # 5. Extract the HTML from #dynamicModal
        modal_html = modal.inner_html()
        logger.info("Successfully extracted attendance modal HTML. Parsing...")

        return self.parser.parse_html(modal_html)

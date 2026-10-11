"""
tests/test_portal_parser.py
────────────────────────────
Comprehensive unit tests for verified VNR VJIET portal attendance parsing and validation.

Covers:
- Exact verified VNR VJIET dynamicModal table HTML
- Cumulative attendance extraction (ignoring monthly values)
- Canonical subject resolution for all 9 subjects
- Total row exclusion and consistency check
- Invalid invariants (attended > conducted)
- Missing subjects rejection (fewer than 9)
- Malformed cell strings ("abc / xyz")
- Total mismatch rejection
"""

from datetime import datetime
from pathlib import Path
import pytest

from attendance_tracker.attendance.repository import AttendanceRepository
from attendance_tracker.core.subject_resolver import SubjectResolver
from attendance_tracker.portal.attendance import AttendanceParser
from attendance_tracker.portal.exceptions import (
    PortalScrapeError,
    PortalValidationError,
)

VERIFIED_PORTAL_MODAL_HTML = """
<div id="dynamicModal" class="modal fade show">
    <div class="modal-dialog">
        <div class="modal-content">
            <div class="table-responsive">
                <table class="table table-sm ">
                    <tr>
                        <td></td>
                        <td>October, 2026</td>
                        <td>Cumulative</td>
                    </tr>
                    <tr>
                        <th>22MN6HS302</th>
                        <td style="text-align:center">2 / 2</td>
                        <td style="text-align:center">4 / 12</td>
                    </tr>
                    <tr>
                        <th>22OE1EN301</th>
                        <td style="text-align:center">1 / 1</td>
                        <td style="text-align:center">12 / 26</td>
                    </tr>
                    <tr>
                        <th>22PC1AM202</th>
                        <td style="text-align:center">1 / 1</td>
                        <td style="text-align:center">16 / 34</td>
                    </tr>
                    <tr>
                        <th>22PC1AM203</th>
                        <td style="text-align:center">0 / 0</td>
                        <td style="text-align:center">26 / 39</td>
                    </tr>
                    <tr>
                        <th>22PC1IN301</th>
                        <td style="text-align:center">1 / 1</td>
                        <td style="text-align:center">21 / 33</td>
                    </tr>
                    <tr>
                        <th>22PC1IT202</th>
                        <td style="text-align:center">1 / 1</td>
                        <td style="text-align:center">22 / 33</td>
                    </tr>
                    <tr>
                        <th>22PC2AM202</th>
                        <td style="text-align:center">3 / 3</td>
                        <td style="text-align:center">24 / 36</td>
                    </tr>
                    <tr>
                        <th>22PC2IN301</th>
                        <td style="text-align:center">0 / 0</td>
                        <td style="text-align:center">15 / 33</td>
                    </tr>
                    <tr>
                        <th>22PE1AM301</th>
                        <td style="text-align:center">0 / 1</td>
                        <td style="text-align:center">23 / 42</td>
                    </tr>
                    <tr>
                        <th>Total</th>
                        <td>9 / 10 (90.00)</td>
                        <td>163 / 288 (56.60)</td>
                    </tr>
                </table>
            </div>
        </div>
    </div>
</div>
"""

HTML_INCONSISTENT_TOTAL = """
<table>
    <tr><td></td><td>October, 2026</td><td>Cumulative</td></tr>
    <tr><th>22MN6HS302</th><td>2 / 2</td><td>4 / 12</td></tr>
    <tr><th>22OE1EN301</th><td>1 / 1</td><td>12 / 26</td></tr>
    <tr><th>22PC1AM202</th><td>1 / 1</td><td>16 / 34</td></tr>
    <tr><th>22PC1AM203</th><td>0 / 0</td><td>26 / 39</td></tr>
    <tr><th>22PC1IN301</th><td>1 / 1</td><td>21 / 33</td></tr>
    <tr><th>22PC1IT202</th><td>1 / 1</td><td>22 / 33</td></tr>
    <tr><th>22PC2AM202</th><td>3 / 3</td><td>24 / 36</td></tr>
    <tr><th>22PC2IN301</th><td>0 / 0</td><td>15 / 33</td></tr>
    <tr><th>22PE1AM301</th><td>0 / 1</td><td>23 / 42</td></tr>
    <tr><th>Total</th><td>9 / 10</td><td>999 / 999 (100.00)</td></tr>
</table>
"""

HTML_ATTENDED_EXCEEDS_CONDUCTED = """
<table>
    <tr><td></td><td>Monthly</td><td>Cumulative</td></tr>
    <tr><th>22PC1IT202</th><td>1 / 1</td><td>35 / 30</td></tr>
</table>
"""

HTML_MALFORMED_CELL = """
<table>
    <tr><td></td><td>Monthly</td><td>Cumulative</td></tr>
    <tr><th>22PC1IT202</th><td>1 / 1</td><td>abc / xyz</td></tr>
</table>
"""

HTML_MISSING_SUBJECTS = """
<table>
    <tr><td></td><td>October, 2026</td><td>Cumulative</td></tr>
    <tr><th>22MN6HS302</th><td>2 / 2</td><td>4 / 12</td></tr>
    <tr><th>22PC1IT202</th><td>1 / 1</td><td>22 / 33</td></tr>
    <tr><th>Total</th><td>3 / 3</td><td>26 / 45</td></tr>
</table>
"""


@pytest.fixture
def parser(tmp_path: Path):
    db_file = tmp_path / "test_parser.db"
    repo = AttendanceRepository(db_path=db_file)
    repo.seed_canonical_subjects()
    repo.seed_aliases()
    resolver = SubjectResolver(repository=repo)
    return AttendanceParser(resolver=resolver, expected_subject_count=9)


def test_parse_verified_vnr_portal_modal(parser):
    """Verify parsing of the exact verified VNR VJIET attendance table."""
    now = datetime(2026, 10, 4, 23, 30, 0)
    data = parser.parse_html(VERIFIED_PORTAL_MODAL_HTML, scraped_at=now)

    assert data.scraped_at == now
    assert len(data.records) == 9

    # Cumulative check: OS must have 22/33, not monthly 1/1
    os_rec = next(r for r in data.records if r.subject_code == "22PC1IT202")
    assert os_rec.attended == 22
    assert os_rec.conducted == 33
    assert os_rec.percentage_raw == pytest.approx(66.666, rel=1e-2)

    # Check all 9 canonical subjects are mapped
    expected_values = {
        "22MN6HS302": (4, 12),
        "22OE1EN301": (12, 26),
        "22PC1AM202": (16, 34),
        "22PC1AM203": (26, 39),
        "22PC1IN301": (21, 33),
        "22PC1IT202": (22, 33),
        "22PC2AM202": (24, 36),
        "22PC2IN301": (15, 33),
        "22PE1AM301": (23, 42),
    }

    for code, (expected_att, expected_cond) in expected_values.items():
        rec = next(r for r in data.records if r.subject_code == code)
        assert rec.attended == expected_att, f"Mismatch for {code} attended"
        assert rec.conducted == expected_cond, f"Mismatch for {code} conducted"

    # Verify Total row was excluded from subject records
    assert not any("total" in r.subject_name_raw.lower() for r in data.records)

    # Verify sum matches 163 / 288 (56.60%)
    total_att = sum(r.attended for r in data.records)
    total_cond = sum(r.conducted for r in data.records)
    assert total_att == 163
    assert total_cond == 288
    assert (total_att / total_cond) * 100 == pytest.approx(56.597, rel=1e-2)


def test_parse_total_row_inconsistency_fails(parser):
    """Verify that an inconsistent Total row triggers a validation error."""
    with pytest.raises(PortalValidationError) as exc_info:
        parser.parse_html(HTML_INCONSISTENT_TOTAL)
    assert "does not match sum" in str(exc_info.value)


def test_parse_attended_exceeds_conducted_fails(parser):
    """Verify that attended > conducted raises PortalValidationError."""
    with pytest.raises(PortalValidationError) as exc_info:
        parser.parse_html(HTML_ATTENDED_EXCEEDS_CONDUCTED, strict_count=False)
    assert "Attended exceeds conducted" in str(exc_info.value)


def test_parse_malformed_cell_fails(parser):
    """Verify that malformed text ('abc / xyz') raises PortalValidationError."""
    with pytest.raises(PortalValidationError) as exc_info:
        parser.parse_html(HTML_MALFORMED_CELL, strict_count=False)
    assert "Malformed attendance" in str(exc_info.value)


def test_parse_missing_subjects_fails(parser):
    """Verify that fewer than 9 canonical subjects raises PortalScrapeError in strict mode."""
    with pytest.raises(PortalScrapeError) as exc_info:
        parser.parse_html(HTML_MISSING_SUBJECTS, strict_count=True)
    assert "Expected 9 canonical subjects" in str(exc_info.value)


def test_parse_empty_html_fails(parser):
    """Verify that empty HTML raises PortalScrapeError."""
    with pytest.raises(PortalScrapeError):
        parser.parse_html("")

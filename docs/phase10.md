# Phase 10 — Production Hardening & Reliability

## 1. Overview

Phase 10 hardens the College Attendance Tracker for resilient, deterministic daily production use. It establishes end-to-end failure safety, database transaction atomicity, strict invariant verification, credential hygiene, observability, and robust NLU inference handling without modifying the frozen Phase 7.6 SLM checkpoint or adding extraneous user-facing features.

```mermaid
flowchart TD
    subgraph Portal Safety Layer
        A[Portal Login Attempt] -->|1 Attempt Max| B{Auth Outcome}
        B -->|Success| C[Navigate to #attp / #dynamicModal]
        B -->|Failure / Lockout / CAPTCHA| D[Immediate Abort + Log Failure in scrape_log]
        C --> E[Extract Cumulative Attendance Table]
    end

    subgraph Data Integrity & Validation
        E --> F{Validate 7 Invariants}
        F -->|1. Exactly 9 Canonical Subjects| G[Check Pass]
        F -->|2. No Duplicate Subject Codes| G
        F -->|3. No Unexpected Subjects| G
        F -->|4. Cumulative Column Used| G
        F -->|5. 0 <= Attended <= Conducted| G
        F -->|6. Total Sum Consistency| G
        F -->|Any Violation| H[Raise PortalValidationError + Abort]
    end

    subgraph Database Safety
        G --> I[Begin Atomic Transaction]
        I --> J[Insert Snapshot & 9 Subject Records]
        J --> K[Compute & Persist Changes]
        K --> L[Commit Transaction + Log SUCCESS in scrape_log]
        H --> M[Rollback Transaction: Database Preserved]
        D --> M
    end

    subgraph Application & NLU Layer
        N[User Query] --> O[Input Normalization: Whitespace / Punctuation / Casing]
        O --> P[Frozen Phase 7.6 SLM (5.85M params)]
        P --> Q[Deterministic Tool Router]
        Q --> R[Safe Formatted Response / CLI Output]
    end
```

---

## 2. Production Safety Invariants & Hardening Measures

### 2.1 Portal Failure Safety & Lockout Protection
- **Strict Single-Attempt Rule**: Exactly **ONE** login attempt per execution (`attendance_tracker/portal/auth.py`). Automated retry loops are strictly prohibited to prevent student account lockout.
- **Immediate Abort on Security Prompts**: Detects CAPTCHA challenges, Date of Birth prompts, and lockout notifications (`BlockedUntil`), raising `PortalAuthError` immediately.
- **DOM & Navigation Resilience**: Handles page timeouts, missing action tiles (`#attp`), missing modal dialogs (`#dynamicModal`), and unpopulated AJAX tables gracefully via `PortalNavigationError`.
- **Fail-Safe Snapshot Preservation**: Any authentication, network, or parsing failure aborts before modifying the database. Existing historical snapshots remain completely untouched.

### 2.2 Data Integrity & Table Parsing Invariants
`attendance_tracker/portal/attendance.py` enforces 7 invariants before any data reaches SQLite:
1. **Canonical Subject Count**: Exactly 9 canonical VNR VJIET subjects must be extracted.
2. **Zero Duplicates**: Multiple rows mapping to the same subject code are rejected.
3. **No Unrecognized Subjects**: Unknown courses or extra subjects trigger `PortalScrapeError`.
4. **Cumulative Column Parsing**: Extracts the overall cumulative ratio (`X / Y`) rather than the monthly slice.
5. **Count Invariants**: Attended and conducted counts must be non-negative, and $\text{attended} \le \text{conducted}$.
6. **Total Consistency Cross-Check**: If a `Total` summary row is reported by the portal, the sum of subject counts ($\sum \text{attended}, \sum \text{conducted}$) must match the portal total exactly.
7. **Domain Exception Hierarchy**: All parsing violations raise `PortalValidationError` or `PortalScrapeError`, inheriting from `PortalError`.

### 2.3 Database Safety & Transaction Atomicity
- **Atomic Multi-Row Writes**: `AttendanceRepository.create_snapshot()` wraps snapshot metadata and subject records within a single SQLite transaction (`with conn:`). Any validation or foreign-key error triggers complete rollback.
- **Foreign Key Enforcement**: `PRAGMA foreign_keys=ON;` is enforced on every connection, preventing orphaned records or invalid alias associations.
- **Audit Logging**: Every sync attempt (successful or failed) records `attempted_at`, `completed_at`, `status` (`SUCCESS` / `FAILED`), `snapshot_id`, and `error_msg` in `scrape_log`.

### 2.4 Credential Security & Hygiene
- **Zero Hardcoded Secrets**: Verified that no passwords, API tokens, or session cookies exist in repository files, configuration files, test fixtures, SQLite database records, or CLI outputs.
- **Windows Credential Manager**: Credentials reside securely in Windows Credential Manager via the `keyring` library (`service="attendance_tracker_portal"`).

### 2.5 NLU Robustness & Inference Normalization
- **Frozen Model Preservation**: Phase 7.6 SLM (`checkpoints/phase7_6_boundary/best.pt`, 5,853,184 parameters) is 100% frozen.
- **Input Query Normalization**: `normalize_query_for_slm()` handles duplicate whitespace, repeated punctuation (`???`), and casing normalization (preserving acronyms like `OS`, `ACD`, `DE`, `WP`, `GS`, `OE-1`, `IoTAP`) before BPE tokenization.
- **Threshold Resolution**: Resolves explicit targets (e.g. 80%), college threshold (75.0%), and personal target (70.0%) deterministically.
- **Graceful Fallback**: Unrecognized, malformed, or out-of-domain queries fallback cleanly to helpful guidance without crashing.

---

## 3. Test Suite Verification & Coverage

The test suite consists of **34 test modules** with **580 passing tests** (1 skipped calendar test):

| Category | Test Module | Passed Tests | Description |
| :--- | :--- | :---: | :--- |
| **Database Safety** | `tests/test_database_safety.py` | 8 | Transaction rollback, negative counts, attended > conducted, foreign keys, audit log |
| **Portal Hardening** | `tests/test_portal_hardening.py` | 7 | Lockout, CAPTCHA challenge, navigation timeout, duplicate subjects, extra subjects |
| **NLU Robustness** | `tests/test_nlu_robustness.py` | 6 | Intent casing, 8 canonical intents, threshold resolution, noise resilience, unknown fallback |
| **CLI Hardening** | `tests/test_cli_hardening.py` | 9 | Ask queries, quiet mode, overview tables, sync feedback, credentials management |
| **Portal Parser** | `tests/test_portal_parser.py` | 6 | DOM table extraction, cumulative ratio parsing, Total consistency cross-check |
| **Portal Service** | `tests/test_portal_service.py` | 4 | End-to-end sync, change detection persistence, safe failure handling |
| **Portal Auth** | `tests/test_portal_auth.py` | 4 | Single attempt auth, credential errors, CAPTCHA, navigation timeouts |
| **Portal Credentials** | `tests/test_portal_credentials.py` | 5 | Keyring retrieval, isolation, missing credentials, deletion |
| **E2E Pipeline** | `tests/test_e2e_pipeline.py` | 14 | Dynamic live repository E2E queries, calculations, projections |
| **NLU & Router** | `tests/test_tool_router.py` | 12 | Intent routing, tool execution, structured responses |
| **Thresholds** | `tests/test_thresholds.py` | 9 | College vs personal thresholds, target overrides |
| **Engine & Math** | `tests/test_engine.py` | 229 | Attendance calculation formulas, margin computations |
| **Other Modules** | Core, SLM, Calendar, Timetable, Model | 267 | Tokenizer, generator, attention, checkpointing, training, evaluator |
| **Total** | **34 modules** | **580 passed, 1 skipped, 0 failed** | **Execution time: 70.64s** |

---

## 4. Phase 10 Verification Status

Phase 10 is **complete, hardened, and verified**:
- Real VNR VJIET portal sync is reliable and failure-safe.
- Database integrity and transaction atomicity are fully protected.
- Credentials remain 100% secure in Windows Credential Manager.
- Application CLI and NLU pipeline handle diverse inputs gracefully.
- All 580 automated regression tests pass cleanly.

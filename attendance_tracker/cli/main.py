"""
cli/main.py
───────────
Interactive command-line interface for the College Attendance Tracker.

Commands:
    overview             Render the authoritative attendance overview table.
    ask "<query>"        Ask a single natural language attendance question.
    chat                 Enter interactive conversational assistant session.
    sync                 Synchronize latest attendance from the college portal.
    credentials          Manage portal credentials in Windows Credential Manager.

Usage:
    python -m attendance_tracker.cli.main overview
    python -m attendance_tracker.cli.main ask "What is my attendance in OS?"
    python -m attendance_tracker.cli.main sync
    python -m attendance_tracker.cli.main credentials check
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from attendance_tracker.attendance.repository import OverallAttendance, get_repository
from attendance_tracker.core.config import config
from attendance_tracker.core.logging_setup import get_logger
from attendance_tracker.core.tool_router import PipelineResponse, get_tool_router
from attendance_tracker.portal.credentials import (
    delete_portal_credentials,
    get_credential_service_name,
    get_default_username,
    has_portal_credentials,
)
from attendance_tracker.portal.service import get_portal_service
from attendance_tracker.portal.setup_credentials import run_setup

logger = get_logger(__name__)

# Reconfigure stdout/stderr for UTF-8 on Windows terminals
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

console = Console(highlight=False, legacy_windows=False)


def render_attendance_table(overall: Optional[OverallAttendance] = None) -> None:
    """
    Render a formatted attendance overview table using Rich.
    If overall is None, loads the latest snapshot from the repository.
    """
    repo = get_repository()

    if overall is None:
        overall = repo.get_all_current_attendance()

    if overall is None or not overall.subject_records:
        console.print(
            Panel.fit(
                "[bold yellow]No attendance records found in the database.[/bold yellow]\n"
                "Run [bold cyan]python -m attendance_tracker.cli.main sync[/bold cyan] or "
                "[bold cyan]python scripts/seed_subjects.py[/bold cyan] to populate data.",
                title="College Attendance Tracker",
            )
        )
        return

    college_thresh = config.attendance.college_threshold
    personal_tgt = config.attendance.personal_target

    table = Table(
        title=f"Authoritative Attendance Overview (Snapshot #{overall.snapshot_id} - {overall.scraped_at})",
        caption=f"College Threshold: {college_thresh:.1f}% | Personal Target: {personal_tgt:.1f}%",
        show_header=True,
        header_style="bold magenta",
        show_footer=True,
    )

    table.add_column("Subject Name", style="cyan", no_wrap=False, footer="[bold]Overall[/bold]")
    table.add_column("Code", style="dim", footer="-")
    table.add_column("Short", style="bold blue", footer="-")
    table.add_column("Attended", justify="right", footer=f"[bold]{overall.total_attended}[/bold]")
    table.add_column("Conducted", justify="right", footer=f"[bold]{overall.total_conducted}[/bold]")
    table.add_column("Percentage", justify="right", footer=f"[bold]{overall.overall_percentage:.2f}%[/bold]")
    table.add_column("College (75%)", justify="center", footer="[red]Below[/red]" if overall.overall_percentage < college_thresh else "[green]Above[/green]")
    table.add_column("Target (70%)", justify="center", footer="[red]Below[/red]" if overall.overall_percentage < personal_tgt else "[green]Above[/green]")

    for rec in overall.subject_records:
        s = rec.subject
        pct = rec.percentage
        status_college = "[green]Above[/green]" if pct >= college_thresh else f"[red]-{college_thresh - pct:.1f}%[/red]"
        status_target = "[green]Above[/green]" if pct >= personal_tgt else f"[yellow]-{personal_tgt - pct:.1f}%[/yellow]"

        if pct >= college_thresh:
            pct_str = f"[bold green]{pct:.2f}%[/bold green]"
        elif pct >= personal_tgt:
            pct_str = f"[bold yellow]{pct:.2f}%[/bold yellow]"
        else:
            pct_str = f"[bold red]{pct:.2f}%[/bold red]"

        table.add_row(
            s.full_name,
            s.code,
            s.short_name,
            str(rec.attended),
            str(rec.conducted),
            pct_str,
            status_college,
            status_target,
        )

    console.print()
    console.print(table)
    console.print()


def handle_ask_query(query: str, verbose: bool = True) -> PipelineResponse:
    """Process a single natural language query and print the result."""
    router = get_tool_router()
    response = router.process(query)

    if verbose:
        nlu = response.nlu
        info_lines = [
            f"[dim]Intent:[/dim] [bold cyan]{nlu.intent}[/bold cyan] "
            f"([dim]conf:[/dim] {nlu.confidence:.2%}) | "
            f"[dim]Subject:[/dim] [bold]{nlu.subject or 'None'}[/bold] | "
            f"[dim]Target:[/dim] [bold]{nlu.target or 'None'}[/bold] | "
            f"[dim]Count:[/dim] [bold]{nlu.count or 'None'}[/bold] | "
            f"[dim]Time:[/dim] {response.total_time_ms:.1f}ms",
        ]
        console.print(" ".join(info_lines))

    console.print()
    console.print(Panel(response.response_text, title="Attendance Assistant", border_style="blue"))
    console.print()
    return response


def run_interactive_chat() -> None:
    """Run an interactive REPL chat session with the attendance assistant."""
    console.print(
        Panel.fit(
            "[bold cyan]College Attendance Tracker — Interactive Assistant[/bold cyan]\n"
            "Ask any question about your attendance, projections, or bunk margin.\n"
            "Type [bold yellow]'exit'[/bold yellow], [bold yellow]'quit'[/bold yellow], or [bold yellow]'q'[/bold yellow] to leave.",
            title="Welcome",
            border_style="cyan",
        )
    )

    router = get_tool_router()

    while True:
        try:
            user_input = console.input("[bold green]Student ❯ [/bold green]").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q", ":q"):
                console.print("[dim]Goodbye![/dim]")
                break

            response = router.process(user_input)
            nlu = response.nlu

            # Display parse metadata
            console.print(
                f"[dim]Intent: {nlu.intent} ({nlu.confidence:.1%}) | "
                f"Subject: {nlu.subject or '-'} | "
                f"Count: {nlu.count or '-'} | "
                f"Target: {nlu.target or '-'} | "
                f"Latency: {response.total_time_ms:.1f}ms[/dim]"
            )
            console.print()
            console.print(Panel(response.response_text, title="Assistant", border_style="blue"))
            console.print()

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session closed.[/dim]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")


def handle_sync_portal(headless: bool = True) -> None:
    """Synchronize live attendance data from the college portal."""
    console.print("[bold cyan]Connecting to VNR VJIET portal for attendance synchronization...[/bold cyan]")
    portal_svc = get_portal_service()
    result = portal_svc.sync_attendance(headless=headless)

    if result.success:
        console.print(
            Panel.fit(
                f"[bold green]✓ Portal Synchronization Successful[/bold green]\n\n"
                f"• Snapshot ID: [bold]{result.snapshot_id}[/bold]\n"
                f"• Subjects Synced: [bold]{result.records_count}[/bold]\n"
                f"• Duration: [bold]{result.duration_ms:.1f} ms[/bold]",
                title="Sync Complete",
                border_style="green",
            )
        )
        if result.changes_detected:
            changes_table = Table(title="Detected Attendance Changes", show_header=True, header_style="bold yellow")
            changes_table.add_column("Subject", style="cyan")
            changes_table.add_column("Status", style="magenta")
            changes_table.add_column("Delta", justify="center")
            changes_table.add_column("Explanation", style="dim")

            for c in result.changes_detected:
                delta_str = f"+{c['delta_attended']}/+{c['delta_conducted']}"
                changes_table.add_row(c["subject_code"], c["status"], delta_str, c["explanation"])

            console.print(changes_table)

        render_attendance_table()
    else:
        console.print(
            Panel.fit(
                f"[bold red]✗ Portal Synchronization Failed[/bold red]\n\n"
                f"[white]{result.error_message}[/white]",
                title="Sync Error",
                border_style="red",
            )
        )


def handle_credentials_command(action: str) -> None:
    """Manage Windows Credential Manager entries for the portal."""
    user = get_default_username()
    svc = get_credential_service_name()

    if action == "check" or action == "status":
        exists = has_portal_credentials(user)
        if exists:
            console.print(
                f"[green]✓ Credentials are configured for user '[bold]{user}[/bold]' in service '[bold]{svc}[/bold]'.[/green]"
            )
        else:
            console.print(
                f"[yellow]! No credentials configured for user '[bold]{user}[/bold]'.[/yellow]\n"
                f"Run '[bold cyan]python -m attendance_tracker.cli.main credentials setup[/bold cyan]' to configure."
            )
    elif action == "setup" or action == "set":
        run_setup()
    elif action == "delete" or action == "remove":
        success = delete_portal_credentials(user)
        if success:
            console.print(f"[green]Removed credentials for '[bold]{user}[/bold]' from Windows Credential Manager.[/green]")
        else:
            console.print(f"[red]Failed to remove credentials for '[bold]{user}[/bold]'.[/red]")
    else:
        console.print(f"[red]Unknown credentials action '{action}'. Use setup, check, or delete.[/red]")


def main() -> None:
    """Main CLI router."""
    parser = argparse.ArgumentParser(
        description="College Attendance Tracker CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Overview
    subparsers.add_parser("overview", help="Display attendance overview table")

    # Ask
    ask_parser = subparsers.add_parser("ask", help="Ask a single natural language question")
    ask_parser.add_argument("query", type=str, help="Question to ask")
    ask_parser.add_argument("--quiet", "-q", action="store_true", help="Omit metadata")

    # Chat
    subparsers.add_parser("chat", help="Start interactive conversational session")

    # Sync
    sync_parser = subparsers.add_parser("sync", help="Synchronize attendance from the college portal")
    sync_parser.add_argument("--headed", action="store_true", help="Run browser in visible mode (default: headless)")

    # Credentials
    cred_parser = subparsers.add_parser("credentials", help="Manage portal credentials")
    cred_parser.add_argument(
        "action",
        nargs="?",
        default="check",
        choices=["check", "status", "setup", "set", "delete", "remove"],
        help="Action to perform (default: check)",
    )

    args = parser.parse_args()

    if args.command == "overview" or args.command is None:
        render_attendance_table()
    elif args.command == "ask":
        handle_ask_query(args.query, verbose=not args.quiet)
    elif args.command == "chat":
        run_interactive_chat()
    elif args.command == "sync":
        handle_sync_portal(headless=not args.headed)
    elif args.command == "credentials":
        handle_credentials_command(args.action)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

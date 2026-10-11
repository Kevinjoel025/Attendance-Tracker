"""
core/config.py
──────────────
Loads config.yaml and exposes a single typed AppConfig object.
All other modules import from here — never read config.yaml directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field


# ── Sub-models ────────────────────────────────────────────────────────────────

class ProjectConfig(BaseModel):
    name: str = "College Attendance Tracker"
    version: str = "0.1.0"


class PathsConfig(BaseModel):
    database: str = "data/attendance.db"
    timetable: str = "data/timetable.json"
    holidays: str = "data/holidays.json"
    logs_dir: str = "logs"
    checkpoints: str = "checkpoints"
    datasets: str = "datasets"


class AttendanceConfig(BaseModel):
    college_threshold: float = 75.0
    personal_target: float = 70.0
    default_threshold: float = 75.0


class PortalConfig(BaseModel):
    url: str = "https://automation.vnrvjiet.ac.in"
    credential_service: str = "attendance_tracker_portal"
    username: str = "24071A6947"
    observation_window_end: str = "23:00"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    max_file_size_mb: int = 10
    backup_count: int = 5


class ModelConfig(BaseModel):
    n_layers: int = 6
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 1024
    context_length: int = 256
    vocab_size: int = 4096
    dropout: float = 0.1
    pos_encoding: str = "learned"
    activation: str = "gelu"
    layer_norm: str = "pre"
    weight_tying: bool = True


class TrainingConfig(BaseModel):
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 500
    max_epochs: int = 100
    early_stop_patience: int = 10
    mixed_precision: bool = True
    device: str = "cuda"
    seed: int = 42


class NLUConfig(BaseModel):
    max_tokens_generated: int = 64
    fallback_intent: str = "UNKNOWN"


class TimetableConfig(BaseModel):
    effective_from: str = "2026-07-16"
    non_attendance_activities: List[str] = Field(
        default_factory=lambda: ["LIB", "SPORTS", "ECA/CCA", "SL", "MTP"]
    )


# ── Root config ───────────────────────────────────────────────────────────────

class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    attendance: AttendanceConfig = Field(default_factory=AttendanceConfig)
    portal: PortalConfig = Field(default_factory=PortalConfig)
    timezone: str = "Asia/Kolkata"
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    nlu: NLUConfig = Field(default_factory=NLUConfig)
    timetable: TimetableConfig = Field(default_factory=TimetableConfig)


# ── Loader ────────────────────────────────────────────────────────────────────

def _find_project_root() -> Path:
    """Walk up from this file to find the directory containing config.yaml."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "config.yaml").exists():
            return parent
    # Fallback: cwd
    return Path.cwd()


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """
    Load and validate config.yaml.

    If config_path is not provided, searches upward from this file for config.yaml.
    Falls back to defaults if the file is missing (useful in tests).
    """
    if config_path is None:
        root = _find_project_root()
        config_path = root / "config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        # Return defaults — callers can override specific fields if needed.
        return AppConfig()

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return AppConfig.model_validate(raw)


# ── Singleton ─────────────────────────────────────────────────────────────────
# Modules import `config` directly:
#   from attendance_tracker.core.config import config

config: AppConfig = load_config()


# ── Convenience helpers ───────────────────────────────────────────────────────

def get_db_path() -> Path:
    """Return the absolute path to the SQLite database file."""
    root = _find_project_root()
    return root / config.paths.database


def get_data_dir() -> Path:
    """Return the absolute path to the data/ directory."""
    root = _find_project_root()
    return root / "data"

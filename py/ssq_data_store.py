"""Compatibility facade for draw-history validation and persistence."""

from ssq_csv_store import atomic_write_csv, read_existing_csv
from ssq_history_store import update_csv_file
from ssq_snapshot_validation import (
    MIN_FULL_SNAPSHOT_RECORDS,
    normalize_lottery_frame,
    validate_authoritative_snapshot,
)

__all__ = [
    'MIN_FULL_SNAPSHOT_RECORDS',
    'atomic_write_csv',
    'normalize_lottery_frame',
    'read_existing_csv',
    'update_csv_file',
    'validate_authoritative_snapshot',
]

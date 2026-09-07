"""Validated merge workflow for persisted draw history."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from ssq_csv_store import atomic_write_csv, read_existing_csv
from ssq_snapshot_validation import (
    normalize_lottery_frame,
    validate_authoritative_snapshot,
)

logger = logging.getLogger('ssq_data_processor')


@dataclass(frozen=True)
class HistoryStoreDependencies:
    """Replaceable boundaries used by the draw-history merge workflow."""

    normalize_frame: Callable[[Any], pd.DataFrame] = normalize_lottery_frame
    read_csv: Callable[[Any], pd.DataFrame] = read_existing_csv
    validate_snapshot: Callable[..., None] = validate_authoritative_snapshot
    write_csv: Callable[[pd.DataFrame, Any], None] = atomic_write_csv


def update_csv_file(
    csv_path,
    all_new_data,
    require_full_snapshot=False,
    dependencies=None,
):
    """Validate and merge new records, preferring new values by issue."""
    if not all_new_data:
        logger.info('No new records; leaving the CSV unchanged.')
        return False

    dependencies = dependencies or HistoryStoreDependencies()
    try:
        new_data = dependencies.normalize_frame(pd.DataFrame(all_new_data))
        existing_data = dependencies.read_csv(csv_path)
        if not existing_data.empty:
            existing_data = dependencies.normalize_frame(existing_data)
        if require_full_snapshot:
            dependencies.validate_snapshot(new_data, existing_data)

        combined = (
            pd.concat([existing_data, new_data], ignore_index=True)
            if not existing_data.empty else new_data
        )
        final_data = dependencies.normalize_frame(
            combined.drop_duplicates(subset=['期号'], keep='last')
        )
        dependencies.write_csv(final_data, csv_path)
        logger.info(
            'CSV file updated at %s with %s records.',
            csv_path,
            len(final_data),
        )
        return True
    except (
        OSError, UnicodeError, TypeError, ValueError, pd.errors.ParserError
    ) as exc:
        logger.error('Unable to update CSV file: %s', exc)
        return False

"""Validated merge and atomic persistence for draw history."""

import logging

import pandas as pd
from ssq_csv_store import atomic_write_csv, read_existing_csv
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

logger = logging.getLogger('ssq_data_processor')


def update_csv_file(csv_path, all_new_data, require_full_snapshot=False):
    """Validate and merge new records, preferring new values by issue."""
    if not all_new_data:
        logger.info('No new records; leaving the CSV unchanged.')
        return False

    try:
        new_data = normalize_lottery_frame(pd.DataFrame(all_new_data))
        existing_data = read_existing_csv(csv_path)
        if not existing_data.empty:
            existing_data = normalize_lottery_frame(existing_data)
        if require_full_snapshot:
            validate_authoritative_snapshot(new_data, existing_data)

        combined = (
            pd.concat([existing_data, new_data], ignore_index=True)
            if not existing_data.empty else new_data
        )
        final_data = normalize_lottery_frame(
            combined.drop_duplicates(subset=['期号'], keep='last')
        )
        atomic_write_csv(final_data, csv_path)
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

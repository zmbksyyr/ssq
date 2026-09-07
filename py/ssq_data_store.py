"""Validated merge and atomic persistence for draw history."""

import csv
import logging
import os

import pandas as pd
from ssq_file_io import atomic_text_writer
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


def read_existing_csv(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        logger.info(
            'CSV file does not exist or is empty; a new file will be created.'
        )
        return pd.DataFrame()
    logger.info('Reading existing CSV file: %s', csv_path)
    try:
        return pd.read_csv(csv_path, dtype={'期号': str}, encoding='utf-8')
    except (UnicodeDecodeError, pd.errors.ParserError):
        logger.warning('Unable to read CSV as UTF-8; trying GBK.')
        return pd.read_csv(csv_path, dtype={'期号': str}, encoding='gbk')
    except pd.errors.EmptyDataError:
        logger.warning('Existing CSV file is empty.')
        return pd.DataFrame()


def atomic_write_csv(frame, csv_path):
    with atomic_text_writer(csv_path) as temporary_file:
        frame.to_csv(
            temporary_file,
            index=False,
            quoting=csv.QUOTE_MINIMAL,
        )


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

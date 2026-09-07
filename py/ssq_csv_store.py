"""CSV persistence for normalized draw history."""

import csv
import logging
import os

import pandas as pd
from ssq_file_io import atomic_text_writer

logger = logging.getLogger('ssq_data_processor')


def read_existing_csv(csv_path):
    """Read an existing draw CSV, returning an empty frame when absent."""
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
    """Atomically replace a CSV with the supplied frame."""
    with atomic_text_writer(csv_path) as temporary_file:
        frame.to_csv(
            temporary_file,
            index=False,
            quoting=csv.QUOTE_MINIMAL,
        )

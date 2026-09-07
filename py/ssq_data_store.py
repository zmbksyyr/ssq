"""Validated merge and atomic persistence for draw history."""

import csv
import logging
import os
import tempfile

import pandas as pd
from ssq_draw_data import serialize_draw_frame, validate_draw_dates_not_future

MIN_FULL_SNAPSHOT_RECORDS = 100
logger = logging.getLogger('ssq_data_processor')


def normalize_lottery_frame(frame):
    """Validate and normalize a draw DataFrame before it reaches the CSV."""
    return serialize_draw_frame(frame)


def validate_authoritative_snapshot(new_data, existing_data, today=None):
    """Ensure the advertised full snapshot cannot truncate local history."""
    if len(new_data) < MIN_FULL_SNAPSHOT_RECORDS:
        raise ValueError(
            f'权威全量快照仅有 {len(new_data)} 条，少于最低要求 '
            f'{MIN_FULL_SNAPSHOT_RECORDS} 条'
        )
    validate_draw_dates_not_future(new_data, today=today)
    if existing_data.empty:
        return
    missing_issues = sorted(set(existing_data['期号']) - set(new_data['期号']))
    if missing_issues:
        preview = ', '.join(str(issue) for issue in missing_issues[:5])
        suffix = ' ...' if len(missing_issues) > 5 else ''
        raise ValueError(f'权威全量快照缺少本地已有期号: {preview}{suffix}')


def read_existing_csv(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        logger.info('CSV文件不存在或为空，将创建新文件。')
        return pd.DataFrame()
    logger.info(f'正在读取现有CSV文件: {csv_path}')
    try:
        return pd.read_csv(csv_path, dtype={'期号': str}, encoding='utf-8')
    except (UnicodeDecodeError, pd.errors.ParserError):
        logger.warning('UTF-8编码读取失败，尝试GBK编码...')
        return pd.read_csv(csv_path, dtype={'期号': str}, encoding='gbk')
    except pd.errors.EmptyDataError:
        logger.warning('现有CSV文件为空。')
        return pd.DataFrame()


def atomic_write_csv(frame, csv_path):
    target = os.path.abspath(os.fspath(csv_path))
    target_directory = os.path.dirname(target)
    os.makedirs(target_directory, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            newline='',
            dir=target_directory,
            prefix='.ssq-',
            suffix='.tmp',
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            frame.to_csv(
                temporary_file,
                index=False,
                quoting=csv.QUOTE_MINIMAL,
            )
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def update_csv_file(csv_path, all_new_data, require_full_snapshot=False):
    """Validate and merge new records, preferring new values by issue."""
    if not all_new_data:
        logger.info('没有新的数据可供更新，CSV文件保持不变。')
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
            f'CSV文件已成功更新并保存至: {csv_path}。总计 {len(final_data)} 条记录。'
        )
        return True
    except (
        OSError, UnicodeError, TypeError, ValueError, pd.errors.ParserError
    ) as exc:
        logger.error(f'更新CSV文件时发生严重错误: {exc}')
        return False

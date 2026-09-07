"""Shared validation and normalization for lottery draw tables."""

import hashlib
from itertools import pairwise

import pandas as pd
from ssq_domain import DRAW_COLUMNS, DRAW_WEEKDAYS
from ssq_draw_schedule import local_today
from ssq_parsing import parse_blue_ball, parse_issue, parse_red_balls


def validate_draw_dates_not_future(frame, today=None):
    """Reject draw records dated after the current local calendar date."""
    if frame.empty:
        return
    today = today or local_today()
    latest_date = pd.to_datetime(
        frame['日期'],
        format='%Y-%m-%d',
        errors='raise',
    ).max().date()
    if latest_date > today:
        raise ValueError(f'开奖记录包含未来开奖日期: {latest_date}')


def normalize_draw_frame(frame):
    """Return a validated draw table in chronological, typed form."""
    actual_columns = set(frame.columns)
    expected_columns = set(DRAW_COLUMNS)
    if actual_columns != expected_columns or len(frame.columns) != len(DRAW_COLUMNS):
        missing = sorted(expected_columns - actual_columns)
        unexpected = sorted(actual_columns - expected_columns)
        raise ValueError(f'字段不匹配: 缺少 {missing}, 多余 {unexpected}')

    normalized = frame.loc[:, list(DRAW_COLUMNS)].copy()
    normalized['期号'] = normalized['期号'].apply(parse_issue)
    if normalized['期号'].duplicated().any():
        duplicates = sorted(int(issue) for issue in normalized.loc[
            normalized['期号'].duplicated(keep=False), '期号'
        ].unique())
        raise ValueError(f'存在重复期号: {duplicates}')

    parsed_dates = pd.to_datetime(
        normalized['日期'], format='%Y-%m-%d', errors='raise'
    )
    normalized['日期'] = parsed_dates
    normalized['红球'] = normalized['红球'].apply(parse_red_balls)
    normalized['蓝球'] = normalized['蓝球'].apply(parse_blue_ball)
    normalized = normalized.sort_values('期号').reset_index(drop=True)

    if ((normalized['期号'] // 1000) != normalized['日期'].dt.year).any():
        raise ValueError('期号年份与开奖日期不一致')
    if normalized['日期'].duplicated().any():
        raise ValueError('同一开奖日期不能对应多个期号')
    if not normalized['日期'].is_monotonic_increasing:
        raise ValueError('期号与开奖日期顺序不一致')
    if not normalized['日期'].dt.weekday.isin(DRAW_WEEKDAYS).all():
        raise ValueError('开奖日期必须为周二、周四或周日')

    issues = normalized['期号'].tolist()
    for previous, current in pairwise(issues):
        previous_year, previous_sequence = divmod(previous, 1000)
        current_year, current_sequence = divmod(current, 1000)
        continuous = (
            current_year == previous_year
            and current_sequence == previous_sequence + 1
        ) or (
            current_year == previous_year + 1
            and current_sequence == 1
        )
        if not continuous:
            raise ValueError(f'开奖期号不连续: {previous} -> {current}')

    normalized['日期'] = normalized['日期'].dt.strftime('%Y-%m-%d')
    return normalized


def serialize_draw_frame(frame):
    """Return a validated draw table using the stable on-disk CSV format."""
    serialized = normalize_draw_frame(frame)
    serialized['红球'] = serialized['红球'].apply(
        lambda numbers: ','.join(f'{number:02d}' for number in numbers)
    )
    serialized['蓝球'] = serialized['蓝球'].apply(lambda number: f'{number:02d}')
    return serialized


def fingerprint_draw_frame(frame):
    """Return a stable fingerprint of normalized, serialized draw history."""
    canonical_csv = serialize_draw_frame(frame).to_csv(
        index=False,
        lineterminator='\n',
    )
    return hashlib.sha256(canonical_csv.encode('utf-8')).hexdigest()

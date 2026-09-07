"""Shared validation and normalization for lottery draw tables."""

import pandas as pd
from ssq_core import (
    DRAW_COLUMNS,
    DRAW_WEEKDAYS,
    parse_blue_ball,
    parse_issue,
    parse_red_balls,
)


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
    if not normalized['日期'].is_monotonic_increasing:
        raise ValueError('期号与开奖日期顺序不一致')
    if not normalized['日期'].dt.weekday.isin(DRAW_WEEKDAYS).all():
        raise ValueError('开奖日期必须为周二、周四或周日')

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

"""Validated preparation of draw history for one analysis run."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from ssq_core import infer_next_issue
from ssq_draw_data import (
    fingerprint_draw_frame,
    normalize_draw_frame,
    validate_draw_dates_not_future,
)
from ssq_features import FEATURE_COLUMNS, feature_engineer
from ssq_workflow_models import PreparedHistory


@dataclass(frozen=True)
class HistoryPreparationDependencies:
    load_data: Callable[..., Any]
    fingerprint_history: Callable[..., str]
    engineer_features: Callable[..., pd.DataFrame]
    infer_target_issue: Callable[..., int]


def load_and_preprocess_data(
    filepath,
    normalize=None,
    validate_dates=None,
):
    """Load, validate, parse, and chronologically order draw history."""
    normalize = normalize or normalize_draw_frame
    validate_dates = validate_dates or validate_draw_dates_not_future
    try:
        frame = pd.read_csv(filepath, header=0)
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        print(f"错误: 无法加载数据文件 '{filepath}': {exc}")
        return None

    try:
        normalized = normalize(frame)
        validate_dates(normalized)
        return normalized
    except (TypeError, ValueError) as exc:
        print(f"错误: 数据文件 '{filepath}' 校验失败: {exc}")
        return None


def default_history_dependencies():
    return HistoryPreparationDependencies(
        load_data=load_and_preprocess_data,
        fingerprint_history=fingerprint_draw_frame,
        engineer_features=feature_engineer,
        infer_target_issue=infer_next_issue,
    )


def prepare_history(filepath, dependencies=None):
    """Derive immutable analysis inputs from validated historical draws."""
    if dependencies is None:
        dependencies = default_history_dependencies()
    full_df = dependencies.load_data(filepath)
    if full_df is None or len(full_df) < 50:
        raise SystemExit('错误: 历史数据加载失败或数据量过少（至少需要50期），程序终止。')
    history_sha256 = dependencies.fingerprint_history(full_df)
    full_df = dependencies.engineer_features(full_df)
    latest_issue = str(full_df.iloc[-1]['期号'])
    try:
        target_issue = dependencies.infer_target_issue(
            latest_issue,
            full_df.iloc[-1]['日期'],
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(f'错误: 无法推导下一期期号: {exc}') from exc
    return PreparedHistory(
        frame=full_df,
        feature_columns=FEATURE_COLUMNS,
        latest_issue=latest_issue,
        target_issue=target_issue,
        sha256=history_sha256,
    )

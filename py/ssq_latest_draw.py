"""Loading of the latest validated lottery draw from persisted history."""

import pandas as pd
from ssq_draw_data import normalize_draw_frame, validate_draw_dates_not_future


def load_latest_draw(
    filepath,
    normalize=normalize_draw_frame,
    validate_dates=validate_draw_dates_not_future,
):
    """Return the latest draw after applying the shared history contract."""
    frame = normalize(pd.read_csv(filepath, header=0))
    validate_dates(frame)
    if frame.empty:
        raise ValueError('开奖数据为空')
    latest = frame.iloc[-1]
    return {
        'issue': int(latest['期号']),
        'red': set(latest['红球']),
        'blue': latest['蓝球'],
    }

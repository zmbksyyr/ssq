"""Feature contract and leakage-safe feature engineering."""

from itertools import combinations, pairwise

from ssq_domain import PRIME_RED_BALLS

FEATURE_COLUMNS = (
    'red_sum',
    'red_span',
    'odd_count',
    'blue_lag1',
    'red_zone_small',
    'red_zone_medium',
    'red_zone_large',
    'red_big_count',
    'red_prime_count',
    'red_sum_tail',
    'red_consecutive_groups',
    'red_ac_value',
    'red_tail_uniques',
    'red_sum_lag1',
    'odd_count_lag1',
    'red_sum_ma5',
    'odd_count_ma5',
    'blue_ma5',
)


def _count_consecutive_groups(numbers):
    groups = 0
    in_group = False
    for current, following in pairwise(numbers):
        if following - current == 1:
            if not in_group:
                groups += 1
                in_group = True
        else:
            in_group = False
    return groups


def feature_engineer(df):
    """Return a copy of draw history with the declared model input features."""
    df = df.copy()
    df['red_sum'] = df['红球'].apply(sum)
    df['red_span'] = df['红球'].apply(lambda numbers: max(numbers) - min(numbers))
    df['odd_count'] = df['红球'].apply(
        lambda numbers: sum(number % 2 != 0 for number in numbers)
    )
    df['blue_lag1'] = df['蓝球'].shift(1)
    df['red_zone_small'] = df['红球'].apply(
        lambda numbers: sum(1 <= number <= 11 for number in numbers)
    )
    df['red_zone_medium'] = df['红球'].apply(
        lambda numbers: sum(12 <= number <= 22 for number in numbers)
    )
    df['red_zone_large'] = df['红球'].apply(
        lambda numbers: sum(23 <= number <= 33 for number in numbers)
    )
    df['red_big_count'] = df['红球'].apply(
        lambda numbers: sum(number > 16 for number in numbers)
    )
    df['red_prime_count'] = df['红球'].apply(
        lambda numbers: sum(number in PRIME_RED_BALLS for number in numbers)
    )
    df['red_sum_tail'] = df['red_sum'].apply(lambda value: value % 10)
    df['red_consecutive_groups'] = df['红球'].apply(_count_consecutive_groups)
    df['red_ac_value'] = df['红球'].apply(
        lambda numbers: len({
            abs(first - second) for first, second in combinations(numbers, 2)
        })
    )
    df['red_tail_uniques'] = df['红球'].apply(
        lambda numbers: len({number % 10 for number in numbers})
    )

    window_size = 5
    df['red_sum_lag1'] = df['red_sum'].shift(1)
    df['odd_count_lag1'] = df['odd_count'].shift(1)
    df['red_sum_ma5'] = df['red_sum'].shift(1).rolling(window=window_size).mean()
    df['odd_count_ma5'] = df['odd_count'].shift(1).rolling(window=window_size).mean()
    df['blue_ma5'] = df['蓝球'].shift(1).rolling(window=window_size).mean()
    return df


def validate_feature_columns(frame, feature_columns=FEATURE_COLUMNS):
    """Reject missing, duplicate, or undeclared model feature names."""
    columns = tuple(feature_columns)
    if len(columns) != len(set(columns)):
        raise ValueError('模型特征列不能重复')
    missing = sorted(set(FEATURE_COLUMNS) - set(columns))
    undeclared = sorted(set(columns) - set(FEATURE_COLUMNS))
    if missing:
        raise ValueError(f'缺少声明的模型特征列: {missing}')
    if undeclared:
        raise ValueError(f'包含未声明的模型特征列: {undeclared}')
    if columns != FEATURE_COLUMNS:
        raise ValueError('模型特征列顺序与声明不一致')
    missing_from_frame = sorted(set(columns) - set(frame.columns))
    if missing_from_frame:
        raise ValueError(f'模型数据缺少特征列: {missing_from_frame}')
    return columns

# --- 核心库导入 ---
import argparse
import json
import os
import random
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from math import comb

import lightgbm as lgb
import numpy as np
import pandas as pd
from ssq_core import (
    PRIZE_NAMES,
    PRIZE_RULES,
    atomic_write_text,
    infer_next_issue,
    local_now,
    parse_blue_ball,
    parse_issue,
    parse_red_balls,
)
from ssq_rules import (
    FILTER_NAMES,
    HARD_FILTER_NAMES,
    RED_RULES,
    build_rank_center_scores,
    explain_filter_failures,
    filter_pipeline_stats,
    passes_red_filters,
    score_red_combination,
    select_recommendations,
)
from tqdm import tqdm

# --- 平台特定模块导入, 用于实现非阻塞的键盘输入监听 ---
try:
    # 尝试导入 msvcrt, 这是 Windows 平台专用的库
    import msvcrt
except ImportError:
    # 如果导入失败 (说明不是 Windows 平台), 则导入 select, 适用于 Linux/Mac
    import select

# --- 1. 全局可调参数与路径设置 ---

# --- 核心策略参数 ---
# 红球大底号码池的大小。机器学习评分后，选出分数最高的N个红球进入这个池子。
# (建议15-18)。值越大，后续生成的候选组合越多，计算时间越长，但覆盖面也更广。
# The pool deliberately mixes model ranks.  The old implementation discarded
# the highest-ranked numbers, which made the strategy both opaque and brittle.
POOL_SIZE_RED = 17
RED_HIGH_COUNT = 4
RED_LOW_COUNT = 4

# 最终在报告中推荐的蓝球个数。
NUM_BLUE_BALLS = 7          

# 随机抛弃库的大小。程序会预先生成大量随机组合，在最后过滤时，
# 如果一个精心筛选出的组合也存在于这个随机库中，我们认为它“不够独特”，予以排除。
# (建议10000-100000)。值越大，排他性越强，但生成库的时间越长。
REJECTION_LIB_SIZE = 500000  
RANDOM_SEED = 42
REJECTION_SEED_MULTIPLIER = 1_000_000_007

# --- 回测与输出参数 ---
# 执行历史回测时，使用最近的多少期数据进行验证。
BACKTEST_PERIODS = 200

# 当通过所有规则检验的组合数量超过此阈值时，程序会暂停并询问用户是否要全部显示。
# 这是一个防止刷屏的机制。
INTERACTIVE_THRESHOLD = 100 

# 在上面的交互式询问中，给用户的倒计时秒数。
COUNTDOWN_SECONDS = 10      

# 在最终报告里展示多少注高分单式推荐。
NUM_RECOMMENDATIONS = 10
MAX_SHARED_RED_BALLS = 4
RULE_AUDIT_PERIODS = 200
TOTAL_RED_COMBINATIONS = 1_107_568

DEFAULT_PARAMS = {
    'decay_factor': 0.999,
    'weight_freq': 0.4,
    'weight_omission': 0.5,
    'weight_ml': 0.1,
    'hot_lookback': 10,
    'hot_threshold': 2,
    'hot_bonus': 1.2,
    'cold_lookback': 30,
    'cold_bonus': 1.05,
    'repeat_bonus': 1.15,
    'weight_blue_freq': 0.6,
    'weight_blue_ml': 0.4,
}
INTEGER_PARAM_NAMES = ('hot_lookback', 'hot_threshold', 'cold_lookback')
FLOAT_PARAM_NAMES = (
    'decay_factor', 'weight_freq', 'weight_omission', 'weight_ml',
    'hot_bonus', 'cold_bonus', 'repeat_bonus',
    'weight_blue_freq', 'weight_blue_ml',
)


def normalize_integer_param(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f'{name} 必须为整数')
    return int(value)


def normalize_float_param(name, value):
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f'{name} 必须为数字')
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f'{name} 必须为有限数值')
    return normalized


def validate_weight_group(params, names):
    values = [params[name] for name in names]
    if any(value < 0 for value in values) or not np.isclose(sum(values), 1.0):
        raise ValueError(f"权重 {', '.join(names)} 必须非负且总和为 1")


def validate_param_ranges(params):
    if not 0 < params['decay_factor'] <= 1:
        raise ValueError('decay_factor 必须在 (0, 1] 范围内')
    for name in INTEGER_PARAM_NAMES:
        if params[name] < 0:
            raise ValueError(f'{name} 不能为负数')
    for name in ('hot_bonus', 'cold_bonus', 'repeat_bonus'):
        if params[name] <= 0:
            raise ValueError(f'{name} 必须大于 0')


def validate_strategy_params(params):
    if not isinstance(params, dict):
        raise TypeError('strategy params must be a JSON object')
    unknown = sorted(set(params) - set(DEFAULT_PARAMS))
    if unknown:
        raise ValueError(f"未知策略参数: {', '.join(unknown)}")
    merged = {**DEFAULT_PARAMS, **params}
    for name in INTEGER_PARAM_NAMES:
        merged[name] = normalize_integer_param(name, merged[name])
    for name in FLOAT_PARAM_NAMES:
        merged[name] = normalize_float_param(name, merged[name])
    for group in (
        ('weight_freq', 'weight_omission', 'weight_ml'),
        ('weight_blue_freq', 'weight_blue_ml'),
    ):
        validate_weight_group(merged, group)
    validate_param_ranges(merged)
    return merged


@dataclass(frozen=True)
class StrategyConfig:
    """All selection knobs live here so backtests and live runs share them."""
    pool_size_red: int = POOL_SIZE_RED
    high_count: int = RED_HIGH_COUNT
    low_count: int = RED_LOW_COUNT
    blue_count: int = NUM_BLUE_BALLS
    recommendation_count: int = NUM_RECOMMENDATIONS
    rejection_lib_size: int = REJECTION_LIB_SIZE
    random_seed: int = RANDOM_SEED

    def __post_init__(self):
        if not 6 <= self.pool_size_red <= 33:
            raise ValueError('pool_size_red must be between 6 and 33')
        if min(self.high_count, self.low_count) < 0:
            raise ValueError('high_count and low_count cannot be negative')
        if self.high_count + self.low_count > self.pool_size_red:
            raise ValueError('high_count and low_count exceed pool_size_red')
        if not 1 <= self.blue_count <= 16:
            raise ValueError('blue_count must be between 1 and 16')
        if self.recommendation_count < 1:
            raise ValueError('recommendation_count must be positive')
        if not 0 <= self.rejection_lib_size <= TOTAL_RED_COMBINATIONS:
            raise ValueError(
                f'rejection_lib_size must be between 0 and {TOTAL_RED_COMBINATIONS}'
            )


DEFAULT_STRATEGY_CONFIG = StrategyConfig()


@dataclass(frozen=True)
class BacktestResult:
    periods: int
    active_periods: int
    tickets: int
    cost: int
    winnings: int
    prize_counts: Counter
    evaluated_periods: int = 0
    pool_red_hits: int = 0
    ticket_red_hit_counts: Counter = field(default_factory=Counter)
    candidate_tickets: int = 0
    candidate_red_hit_counts: Counter = field(default_factory=Counter)
    blue_hit_periods: int = 0
    rank_band_hits: Counter = field(default_factory=Counter)

    @property
    def profit(self):
        return self.winnings - self.cost

    @property
    def roi(self):
        return self.winnings / self.cost if self.cost else 0.0

    @property
    def average_pool_red_hits(self):
        return self.pool_red_hits / self.evaluated_periods if self.evaluated_periods else 0.0

    @property
    def average_ticket_red_hits(self):
        if not self.tickets:
            return 0.0
        total_hits = sum(hits * count for hits, count in self.ticket_red_hit_counts.items())
        return total_hits / self.tickets

    @property
    def three_plus_red_tickets(self):
        return sum(count for hits, count in self.ticket_red_hit_counts.items() if hits >= 3)

    @property
    def three_plus_red_rate(self):
        return self.three_plus_red_tickets / self.tickets if self.tickets else 0.0

    @property
    def average_candidate_red_hits(self):
        if not self.candidate_tickets:
            return 0.0
        total_hits = sum(
            hits * count for hits, count in self.candidate_red_hit_counts.items()
        )
        return total_hits / self.candidate_tickets

    @property
    def candidate_three_plus_red_rate(self):
        if not self.candidate_tickets:
            return 0.0
        three_plus = sum(
            count for hits, count in self.candidate_red_hit_counts.items() if hits >= 3
        )
        return three_plus / self.candidate_tickets

    @property
    def ranking_red_hit_delta(self):
        return self.average_ticket_red_hits - self.average_candidate_red_hits

    @property
    def ranking_three_plus_delta(self):
        return self.three_plus_red_rate - self.candidate_three_plus_red_rate

    @property
    def blue_hit_rate(self):
        return self.blue_hit_periods / self.evaluated_periods if self.evaluated_periods else 0.0

    def rank_band_rate(self, band):
        total_actual_reds = self.evaluated_periods * 6
        return self.rank_band_hits[band] / total_actual_reds if total_actual_reds else 0.0

    def rank_band_lift(self, band):
        band_width = len(RANK_BANDS[band]) if band in RANK_BANDS else RANK_OTHER_WIDTH
        expected_rate = band_width / 33
        return self.rank_band_rate(band) / expected_rate if expected_rate else 0.0


@dataclass
class BacktestAccumulator:
    prize_counts: Counter = field(default_factory=Counter)
    cost: int = 0
    winnings: int = 0
    active_periods: int = 0
    evaluated_periods: int = 0
    tickets: int = 0
    pool_red_hits: int = 0
    ticket_red_hit_counts: Counter = field(default_factory=Counter)
    candidate_tickets: int = 0
    candidate_red_hit_counts: Counter = field(default_factory=Counter)
    blue_hit_periods: int = 0
    rank_band_hits: Counter = field(default_factory=Counter)

    def to_result(self, periods):
        return BacktestResult(
            periods=periods,
            active_periods=self.active_periods,
            tickets=self.tickets,
            cost=self.cost,
            winnings=self.winnings,
            prize_counts=self.prize_counts,
            evaluated_periods=self.evaluated_periods,
            pool_red_hits=self.pool_red_hits,
            ticket_red_hit_counts=self.ticket_red_hit_counts,
            candidate_tickets=self.candidate_tickets,
            candidate_red_hit_counts=self.candidate_red_hit_counts,
            blue_hit_periods=self.blue_hit_periods,
            rank_band_hits=self.rank_band_hits,
        )


# --- 文件路径设置 ---
# 获取当前脚本文件所在的目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 假设脚本在 'py' 这样的子目录中, 数据和报告文件都位于其上一级目录
root_dir = os.path.dirname(script_dir) 
# 构造历史数据 CSV 文件的绝对路径 (例如: .../ssq/shuangseqiu.csv)
CSV_PATH = os.path.join(root_dir, 'shuangseqiu.csv')
# 构造机器学习优化参数 JSON 文件的绝对路径 (例如: .../ssq/best_params.json)
PARAMS_JSON_PATH = os.path.join(root_dir, 'best_params.json')
# 构造报告输出目录的绝对路径 (例如: .../ssq/report/)
REPORT_DIR = os.path.join(root_dir, 'report')

# --- 2. 辅助函数库 ---

def load_and_preprocess_data(filepath=CSV_PATH):
    """
    从CSV文件加载并预处理双色球历史数据。

    Args:
        filepath (str): CSV文件的完整路径。

    Returns:
        DataFrame: 处理好并按期号升序排列的数据。如果加载或处理失败，返回None。
    """
    try:
        # 尝试使用 pandas 读取 CSV 文件，假设第一行是表头
        df = pd.read_csv(filepath, header=0)
        # 为了代码的健壮性，强制重命名列
        df.columns = ['期号', '日期', '红球', '蓝球']
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as e:
        # 如果文件读取失败（例如文件不存在、格式错误），打印错误信息并中断程序
        print(f"错误: 无法加载数据文件 '{filepath}': {e}")
        return None
    
    try:
        df['期号'] = df['期号'].apply(parse_issue)
        if df['期号'].duplicated().any():
            raise ValueError("存在重复期号")
        df['_parsed_date'] = pd.to_datetime(
            df['日期'], format='%Y-%m-%d', errors='raise'
        )
        df['红球'] = df['红球'].apply(parse_red_balls)
        df['蓝球'] = df['蓝球'].apply(parse_blue_ball)
    except (TypeError, ValueError) as exc:
        print(f"错误: 数据文件 '{filepath}' 校验失败: {exc}")
        return None

    df = df.sort_values('期号').reset_index(drop=True)
    if ((df['期号'] // 1000) != df['_parsed_date'].dt.year).any():
        print(f"错误: 数据文件 '{filepath}' 校验失败: 期号年份与开奖日期不一致")
        return None
    if not df['_parsed_date'].is_monotonic_increasing:
        print(f"错误: 数据文件 '{filepath}' 校验失败: 期号与开奖日期顺序不一致")
        return None
    return df.drop(columns=['_parsed_date'])

def feature_engineer(df):
    """
    为数据集进行特征工程，基于历史数据计算出各种可能影响下一期结果的统计指标。
    这些指标将作为机器学习模型的输入特征。

    Args:
        df (DataFrame): 输入的包含'红球'和'蓝球'列的数据。

    Returns:
        DataFrame: 增加了18个新特征列的数据。
    """
    df = df.copy()
    # 特征1: 和值 - 6个红球号码之和
    df['red_sum'] = df['红球'].apply(sum)
    # 特征2: 跨度 - 6个红球中最大号码与最小号码的差
    df['red_span'] = df['红球'].apply(lambda x: max(x) - min(x))
    # 特征3: 奇数个数 - 6个红球中奇数的数量
    df['odd_count'] = df['红球'].apply(lambda x: sum(1 for i in x if i % 2 != 0))
    # 特征4: 蓝球滞后1期 - 上一期的蓝球号码
    df['blue_lag1'] = df['蓝球'].shift(1)
    # 特征5: 小区(1-11)号码个数
    df['red_zone_small'] = df['红球'].apply(lambda x: sum(1 for i in x if 1 <= i <= 11))
    # 特征6: 中区(12-22)号码个数
    df['red_zone_medium'] = df['红球'].apply(lambda x: sum(1 for i in x if 12 <= i <= 22))
    # 特征7: 大区(23-33)号码个数
    df['red_zone_large'] = df['红球'].apply(lambda x: sum(1 for i in x if 23 <= i <= 33))
    # 特征8: 大数(>16)个数
    df['red_big_count'] = df['红球'].apply(lambda x: sum(1 for i in x if i > 16))
    # 预先定义 1-33 中的所有质数，提高计算效率；1 不是质数。
    RED_PRIME_NUMBERS = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
    # 特征9: 质数个数
    df['red_prime_count'] = df['红球'].apply(lambda x: sum(1 for i in x if i in RED_PRIME_NUMBERS))
    # 特征10: 和尾 - 和值的个位数
    df['red_sum_tail'] = df['red_sum'].apply(lambda x: x % 10)
    
    def count_consecutive_groups(nums):
        """计算一组号码中的连号组数 (例如 [1,2, 4,5] 有2组连号)"""
        groups = 0
        in_group = False
        for i in range(len(nums) - 1):
            if nums[i+1] - nums[i] == 1:
                if not in_group:
                    groups = groups + 1
                    in_group = True
            else:
                in_group = False
        return groups
    # 特征11: 连号组数
    df['red_consecutive_groups'] = df['红球'].apply(count_consecutive_groups)
    # 特征12: AC值 - 号码间两两之差的绝对值的唯一数量，反映号码的离散程度
    df['red_ac_value'] = df['红球'].apply(
        lambda nums: len({abs(n1 - n2) for n1, n2 in combinations(nums, 2)})
    )
    # 特征13: 尾数唯一值个数 - 6个号码的个位数有多少种不同的值
    df['red_tail_uniques'] = df['红球'].apply(
        lambda numbers: len({number % 10 for number in numbers})
    )
    
    # --- 涉及多期数据的移动平均(MA)和滞后(Lag)特征 ---
    window_size = 5 # 定义移动平均的窗口大小为5期
    # 特征14: 和值滞后1期 - 上一期的和值
    df['red_sum_lag1'] = df['red_sum'].shift(1)
    # 特征15: 奇数个数滞后1期 - 上一期的奇数个数
    df['odd_count_lag1'] = df['odd_count'].shift(1)
    # 特征16: 和值5期移动平均 - (不包含当期)过去5期的和值平均数
    df['red_sum_ma5'] = df['red_sum'].shift(1).rolling(window=window_size).mean()
    # 特征17: 奇数个数5期移动平均
    df['odd_count_ma5'] = df['odd_count'].shift(1).rolling(window=window_size).mean()
    # 特征18: 蓝球5期移动平均
    df['blue_ma5'] = df['蓝球'].shift(1).rolling(window=window_size).mean()
    
    return df

def get_omission(df):
    """
    计算截至当前最新一期，每个红球号码的遗漏值。
    遗漏值指一个号码距离上次开出所隔的期数。

    Args:
        df (DataFrame): 包含历史开奖数据的DataFrame。

    Returns:
        dict: 一个字典，键为红球号码(1-33)，值为其对应的遗漏值。
    """
    total_draws = len(df)
    last_positions = {}
    for position, draw in enumerate(df['红球']):
        for ball in draw:
            last_positions[ball] = position
    return {
        ball: total_draws - last_positions[ball] - 1
        if ball in last_positions else total_draws
        for ball in range(1, 34)
    }

def get_weighted_frequency(series, decay_factor):
    """
    计算时间衰减加权频率。越近的期数权重越高，越远的期数权重越低。
    这比简单的频率统计更能反映号码的近期热度。

    Args:
        series (pd.Series): 一个包含号码列表的Series (例如df['红球'])。
        decay_factor (float): 衰减因子，越接近1，时间权重衰减越慢 (建议0.99-0.999)。

    Returns:
        pd.Series: 每个号码的加权频率。
    """
    N = len(series)
    # 创建一个权重数组，最近的期数权重最高 (decay_factor^0=1)，最远的最低
    weights = np.array([decay_factor ** (N - i - 1) for i in range(N)])
    weighted_counts = {}
    # 遍历每一期的号码列表
    for i, sublist in enumerate(series):
        # 为该期的每个号码，累加上其对应的权重
        for ball in sublist:
            weighted_counts[ball] = weighted_counts.get(ball, 0) + weights[i]
    return pd.Series(weighted_counts)


def apply_red_score_adjustments(red_scores, df_history, params):
    """Apply the hot, cold, and previous-draw bonuses from the original strategy."""
    adjusted = dict(red_scores)
    hot_lookback = int(params.get('hot_lookback', 0))
    hot_threshold = int(params.get('hot_threshold', 0))
    if hot_lookback > 0 and hot_threshold > 0:
        recent = df_history.tail(hot_lookback)['红球']
        hot_counts = Counter(ball for draw in recent for ball in draw)
        for ball, count in hot_counts.items():
            if count >= hot_threshold:
                adjusted[ball] *= params.get('hot_bonus', 1.0)

    cold_lookback = int(params.get('cold_lookback', 0))
    if cold_lookback > 0:
        recent_numbers = {
            ball for draw in df_history.tail(cold_lookback)['红球'] for ball in draw
        }
        for ball in set(range(1, 34)) - recent_numbers:
            adjusted[ball] *= params.get('cold_bonus', 1.0)

    if not df_history.empty:
        for ball in df_history.iloc[-1]['红球']:
            adjusted[ball] *= params.get('repeat_bonus', 1.0)
    return adjusted


def train_ball_models(training_df, feature_columns, candidates, outcome_column,
                      contains_candidate, description=None):
    """Train one binary next-draw model per candidate without mutating the frame."""
    models = {}
    iterator = tqdm(candidates, desc=description, ncols=80) if description else candidates
    features = training_df[feature_columns]
    for candidate in iterator:
        target = training_df[outcome_column].apply(
            lambda outcome, current=candidate: int(
                contains_candidate(outcome, current)
            )
        ).shift(-1)
        valid_rows = target.notna() & features.notna().all(axis=1)
        if not valid_rows.any():
            continue
        model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        model.fit(features.loc[valid_rows], target.loc[valid_rows])
        models[candidate] = model
    return models


def train_prediction_models(training_df, feature_columns, show_progress=False):
    """Train the complete red and blue model sets used by both run modes."""
    red_models = train_ball_models(
        training_df, feature_columns, range(1, 34), '红球',
        lambda draw, ball: ball in draw,
        '训练红球模型' if show_progress else None,
    )
    blue_models = train_ball_models(
        training_df, feature_columns, range(1, 17), '蓝球',
        lambda drawn, ball: drawn == ball,
        '训练蓝球模型' if show_progress else None,
    )
    return red_models, blue_models


def validate_model_sets(red_models, blue_models):
    missing_red = sorted(set(range(1, 34)) - set(red_models))
    missing_blue = sorted(set(range(1, 17)) - set(blue_models))
    if missing_red or missing_blue:
        raise ValueError(f"模型训练不完整: 红球缺失 {missing_red}, 蓝球缺失 {missing_blue}")


def predict_positive_probability(model, features):
    """Return P(class=1), including correct behavior for one-class models."""
    classes = np.asarray(model.classes_)
    if len(classes) == 1:
        return np.full(len(features), float(classes[0] == 1))
    positive_columns = np.flatnonzero(classes == 1)
    if len(positive_columns) != 1:
        raise ValueError(f"模型类别缺少唯一正类 1: {classes.tolist()}")
    probabilities = np.asarray(model.predict_proba(features))
    return probabilities[:, int(positive_columns[0])]


def run_strategy_and_get_scores(df_history, params, ml_models_red, ml_models_blue, feature_columns):
    """
    核心评分函数：结合时间加权频率、遗漏值和机器学习预测概率，为所有号码生成综合评分。

    Args:
        df_history (DataFrame): 用于计算指标的历史数据。
        params (dict): 包含各种权重的参数字典。
        ml_models_red (dict): 预训练好的红球模型。
        ml_models_blue (dict): 预训练好的蓝球模型。
        feature_columns (list): 用于机器学习预测的特征列名。

    Returns:
        tuple: (red_scores, blue_scores) 两个字典，分别包含红球和蓝球的综合评分。
    """
    validate_model_sets(ml_models_red, ml_models_blue)

    # 1. 准备用于ML预测的最新一行特征数据
    # .iloc[[-1]] 确保返回的是DataFrame而不是Series，以适配模型输入
    last_features = df_history.iloc[[-1]][feature_columns].copy()
    # 如果最新特征中有空值（通常是由于移动平均窗口不足），用历史均值填充
    for col in last_features.columns:
        if last_features[col].isnull().any():
            last_features[col] = last_features[col].fillna(df_history[col].mean())
    
    # 2. 红球评分
    # 计算红球的时间衰减加权频率
    red_weighted_freq = get_weighted_frequency(df_history['红球'], params['decay_factor'])
    # 计算红球的当前遗漏值
    red_omission = get_omission(df_history)
    # 使用ML模型预测每个红球下一期出现的概率
    red_ml_probs = {
        ball: predict_positive_probability(ml_models_red[ball], last_features)[0]
        for ball in range(1, 34)
    }
    
    red_scores = {}
    # 为了避免不同指标量纲差异过大，先进行归一化处理
    max_red_freq = red_weighted_freq.max() or 1 # or 1 防止数据为空时除以0
    max_red_omission = max(red_omission.values()) or 1
    
    for ball in range(1, 34):
        # 归一化频率 (0-1之间)
        norm_freq = red_weighted_freq.get(ball, 0) / max_red_freq
        # 归一化遗漏值 (0-1之间)
        norm_omission = red_omission.get(ball, 0) / max_red_omission
        # 综合评分 = 频率分 * 权重 + 遗漏分 * 权重 + ML预测分 * 权重
        red_scores[ball] = (norm_freq * params['weight_freq'] + 
                            norm_omission * params['weight_omission'] + 
                            red_ml_probs[ball] * params['weight_ml'])
    red_scores = apply_red_score_adjustments(red_scores, df_history, params)
    
    # 3. 蓝球评分
    # 计算蓝球的时间衰减加权频率 (注意蓝球每期只有一个，所以用apply将其包装成列表)
    blue_weighted_freq = get_weighted_frequency(df_history['蓝球'].apply(lambda x: [x]), params['decay_factor'])
    # 使用ML模型预测每个蓝球下一期出现的概率
    blue_ml_probs = {
        ball: predict_positive_probability(ml_models_blue[ball], last_features)[0]
        for ball in range(1, 17)
    }
    
    blue_scores = {}
    max_blue_freq = blue_weighted_freq.max() or 1
    
    for ball in range(1, 17):
        # 归一化蓝球频率
        norm_blue_freq = blue_weighted_freq.get(ball, 0) / max_blue_freq
        # 蓝球综合评分 (简单结合频率和ML预测)
        blue_scores[ball] = (norm_blue_freq * params['weight_blue_freq'] + 
                             blue_ml_probs[ball] * params['weight_blue_ml'])
        
    return red_scores, blue_scores


# --- 规则过滤函数库 (每个函数都是一条独立的过滤规则) ---
# r: 代表一个已排序的6红球组合元组, e.g., (1, 5, 10, 12, 23, 31)

RANK_BANDS = {
    "high": range(1, RED_HIGH_COUNT + 1),
    "middle": range(13, 22),
    "low": range(34 - RED_LOW_COUNT, 34),
}
RANK_OTHER_WIDTH = 33 - sum(len(ranks) for ranks in RANK_BANDS.values())


def count_actual_reds_by_rank_band(red_scores, actual_reds):
    """Count actual red balls by their model-score rank band."""
    ranked = [
        ball for ball, _ in sorted(
            red_scores.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    rank_by_ball = {ball: rank for rank, ball in enumerate(ranked, 1)}
    counts = Counter({"high": 0, "middle": 0, "low": 0, "other": 0})
    for ball in actual_reds:
        rank = rank_by_ball[ball]
        band = next((name for name, ranks in RANK_BANDS.items() if rank in ranks), "other")
        counts[band] += 1
    return counts


def build_red_pool(red_scores, config=DEFAULT_STRATEGY_CONFIG, mode="mixed"):
    """Build a red pool from one score band or a high/middle/low mixture."""
    ranked = [ball for ball, _ in sorted(red_scores.items(), key=lambda item: (-item[1], item[0]))]
    if mode == "high":
        return sorted(ranked[:config.pool_size_red])
    if mode == "low":
        return sorted(ranked[-config.pool_size_red:])
    if mode == "middle":
        start = max(0, (len(ranked) - config.pool_size_red) // 2)
        return sorted(ranked[start:start + config.pool_size_red])
    if mode != "mixed":
        raise ValueError(f"unknown pool mode: {mode}")
    high = ranked[:config.high_count]
    low = ranked[-config.low_count:] if config.low_count else []
    middle_count = max(0, config.pool_size_red - len(high) - len(low))
    available_start = config.high_count
    available_end = len(ranked) - config.low_count if config.low_count else len(ranked)
    middle_start = available_start + max(0, (available_end - available_start - middle_count) // 2)
    middle = ranked[middle_start:middle_start + middle_count]
    return sorted(dict.fromkeys(high + middle + low))


def make_rejection_set(size, rng=None):
    """Create a reproducible anti-crowding sample of six-red combinations."""
    if not 0 <= size <= TOTAL_RED_COMBINATIONS:
        raise ValueError(f"rejection size must be between 0 and {TOTAL_RED_COMBINATIONS}")
    rng = rng or random.Random(RANDOM_SEED)
    rejection_set = set()
    while len(rejection_set) < size:
        rejection_set.add(tuple(sorted(rng.sample(range(1, 34), 6))))
    return rejection_set


def rejection_seed_for_issue(base_seed, issue):
    """Derive a reproducible anti-crowding seed that changes every issue."""
    return int(base_seed) * REJECTION_SEED_MULTIPLIER + parse_issue(issue)


def historical_rule_context(full_df, index):
    """Build rule inputs using only draws before the audited issue."""
    history = full_df.iloc[:index]
    recent = [set(draw) for draw in history.iloc[-10:]['红球']]
    return get_omission(history), recent, recent[-1], recent[-2]


def audit_historical_rule_coverage(full_df, periods=RULE_AUDIT_PERIODS):
    """Measure how often each strategy rule accepts actual historical draws."""
    if periods <= 0 or len(full_df) < 11:
        return {name: {'passed': 0, 'total': 0, 'rate': 0.0} for name in FILTER_NAMES}
    start = max(10, len(full_df) - periods)
    passed_counts = Counter()
    total = 0
    for index in range(start, len(full_df)):
        combo = tuple(full_df.iloc[index]['红球'])
        failures = set(explain_filter_failures(
            combo, *historical_rule_context(full_df, index), None
        ))
        for name in FILTER_NAMES:
            if name not in failures:
                passed_counts[name] += 1
        total += 1
    return {
        name: {'passed': passed_counts[name], 'total': total,
               'rate': passed_counts[name] / total if total else 0.0}
        for name in FILTER_NAMES
    }


def audit_historical_hard_pipeline(full_df, periods=RULE_AUDIT_PERIODS):
    """Measure cumulative survival of actual draws through ordered hard rules."""
    hard_rules = [rule for rule in RED_RULES if rule.hard]
    if periods <= 0 or len(full_df) < 11:
        return {
            'total': 0,
            'passed': 0,
            'rate': 0.0,
            'stages': [
                {'rule': rule.name, 'before': 0, 'removed': 0, 'remaining': 0}
                for rule in hard_rules
            ],
        }

    start = max(10, len(full_df) - periods)
    total = len(full_df) - start
    remaining_counts = Counter()
    for index in range(start, len(full_df)):
        combo = tuple(full_df.iloc[index]['红球'])
        context = historical_rule_context(full_df, index)
        for rule in hard_rules:
            if not rule.evaluator(combo, *context):
                break
            remaining_counts[rule.name] += 1

    stages = []
    before = total
    for rule in hard_rules:
        remaining = remaining_counts[rule.name]
        stages.append({
            'rule': rule.name,
            'before': before,
            'removed': before - remaining,
            'remaining': remaining,
        })
        before = remaining
    return {
        'total': total,
        'passed': before,
        'rate': before / total if total else 0.0,
        'stages': stages,
    }


def find_best_7_red_combinations(passed_combos_tuples, red_pool, red_scores=None,
                                 last_draw_set=None, last_2_draw_set=None):
    """Rank every 7-red ticket by valid subticket coverage, then strategy score."""
    if not passed_combos_tuples:
        return []

    passed_combos_set = set(passed_combos_tuples)
    rank_center_scores = build_rank_center_scores(red_scores) if red_scores else None
    ranked = []
    seven_ball_combos = combinations(sorted(red_pool), 7)
    for seven_combo in tqdm(
        seven_ball_combos, total=comb(len(red_pool), 7),
        desc="生成7红球大底", leave=False, ncols=80,
    ):
        subtickets = list(combinations(seven_combo, 6))
        coverage = sum(subticket in passed_combos_set for subticket in subtickets)
        if not coverage:
            continue
        quality = 0.0
        if red_scores:
            quality = sum(
                score_red_combination(
                    subticket, red_scores, last_draw_set, last_2_draw_set,
                    rank_center_scores,
                )
                for subticket in subtickets
            ) / len(subtickets)
        ranked.append((seven_combo, coverage, quality))
    ranked.sort(key=lambda item: (-item[1], -item[2], item[0]))
    return [(combo, coverage) for combo, coverage, _ in ranked]

# --- 3. 交互式输入模块 ---
user_input_lock = threading.Lock() # 线程锁，确保对全局变量的访问安全
user_input_flag = None # 全局标志，用于记录用户输入或超时状态

def get_user_input_with_timeout(timeout):
    """
    在一个独立的线程中运行，用于在指定秒数内等待用户输入'y'。
    这是一个非阻塞的输入实现，不会卡住主程序。

    Args:
        timeout (int): 等待用户输入的秒数。
    """
    global user_input_flag
    start_time = time.time()
    
    prompt = f"\n发现大量高质量组合。输入 'y' 并回车可在 {timeout} 秒内查看全部，否则将仅输出随机推荐...\n"
    sys.stdout.write(prompt)
    sys.stdout.flush() # 强制刷新输出缓冲区

    # --- 根据不同操作系统选择不同的实现方式 ---
    if 'msvcrt' in sys.modules: # Windows 实现
        while time.time() - start_time < timeout and user_input_flag is None:
            if msvcrt.kbhit(): # 如果检测到键盘敲击
                char = msvcrt.getch().decode(errors='ignore').lower()
                if char in ('y', '\r', '\n'): # 接受 'y' 或直接回车
                    with user_input_lock:
                        user_input_flag = 'y'
                    break
            time.sleep(0.1) # 短暂休眠，避免CPU空转
    else: # Linux/Mac 实现
        # 使用select监听标准输入流(sys.stdin)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist: # 如果在超时前监听到输入
            s = sys.stdin.readline().strip().lower()
            if s == 'y':
                with user_input_lock:
                    user_input_flag = 'y'
    
    # 倒计时结束后，检查标志位
    with user_input_lock:
        if user_input_flag is None: # 如果用户始终没有输入
            user_input_flag = 'timeout' # 标记为超时
    
    sys.stdout.write("\n倒计时结束。\n")
    sys.stdout.flush()

# --- 4. 核心功能模块 (回测与预测) ---

def evaluate_backtest_mode(mode, current, actual_red_set, actual_blue,
                           recommended_blue, rank_band_hits, red_scores,
                           omission, last_10, last_1, last_2, rejection_set,
                           config):
    """Evaluate one pool mode for one historical issue."""
    red_pool = build_red_pool(red_scores, config=config, mode=mode)
    current.evaluated_periods += 1
    current.pool_red_hits += len(set(red_pool) & actual_red_set)
    current.rank_band_hits.update(rank_band_hits)
    if recommended_blue == actual_blue:
        current.blue_hit_periods += 1

    passed_combos = [
        combo for combo in combinations(sorted(red_pool), 6)
        if passes_red_filters(
            combo, omission, last_10, last_1, last_2, rejection_set
        )
    ]
    if not passed_combos:
        return

    red_hits_by_combo = {
        combo: len(set(combo) & actual_red_set) for combo in passed_combos
    }
    current.candidate_tickets += len(passed_combos)
    current.candidate_red_hit_counts.update(red_hits_by_combo.values())
    selected_combos = select_recommendations(
        passed_combos, red_scores, last_1, last_2,
        limit=config.recommendation_count,
    )
    current.active_periods += 1
    current.tickets += len(selected_combos)
    current.cost += len(selected_combos) * 2
    blue_hits = int(recommended_blue == actual_blue)
    for combo in selected_combos:
        red_hits = red_hits_by_combo[combo]
        current.ticket_red_hit_counts[red_hits] += 1
        hit_key = (red_hits, blue_hits)
        prize = PRIZE_RULES.get(hit_key, 0)
        if prize > 0:
            current.winnings += prize
            current.prize_counts[hit_key] += 1


def run_full_backtest(full_df, params, feature_columns, num_periods,
                      pool_modes=("mixed",), config=DEFAULT_STRATEGY_CONFIG):
    """
    对最近 N 期执行滚动策略回测。
    每一步仅使用当前期之前的数据重新训练模型，避免未来数据泄露。
    返回每种候选池模式对应的 BacktestResult。
    """
    print("\n" + "="*70)
    print(f"        最近 {num_periods} 期完整策略滚动回测")
    print("="*70)
    
    # 检查是否有足够的数据进行回测 (需要回测期数 + 至少50期用于模型初次训练)
    if len(full_df) < num_periods + 50:
        print(f"历史数据不足 {num_periods + 50} 期，无法执行回测。跳过此步骤。")
        return {mode: BacktestResult(0, 0, 0, 0, 0, Counter()) for mode in pool_modes}

    metrics = {mode: BacktestAccumulator() for mode in pool_modes}
    
    # 定义回测的时间范围，从倒数第N期到倒数第1期
    backtest_range = range(len(full_df) - num_periods, len(full_df))
    
    # 开始回测循环，使用tqdm显示进度条
    with tqdm(total=len(backtest_range), desc="执行严谨回测", ncols=80) as pbar:
        for i in backtest_range:
            # 1. 准备当期的数据：i之前是历史，i是当期的开奖结果
            history_df_for_step = full_df.iloc[:i]
            actual_draw = full_df.iloc[i]
            actual_red_set = set(actual_draw['红球'])
            actual_blue = actual_draw['蓝球']
            
            # --- 核心修正部分: 在每次循环内部，仅使用当前的历史数据重新训练一套全新的模型 ---
            training_data_for_step = history_df_for_step.iloc[5:].copy()
            if len(training_data_for_step) < 20: # 如果用于训练的数据太少，则跳过本期回测
                pbar.update(1)
                continue
            
            local_ml_models_red, local_ml_models_blue = train_prediction_models(
                training_data_for_step, feature_columns
            )
            
            if len(local_ml_models_red) != 33 or len(local_ml_models_blue) != 16: # 如果模型训练不完整，跳过
                pbar.update(1)
                continue
            # --- 核心修正部分结束 ---

            # 2. 使用刚刚训练好的【局部模型】进行评分和筛选
            red_scores, blue_scores = run_strategy_and_get_scores(history_df_for_step, params, local_ml_models_red, local_ml_models_blue, feature_columns)
            
            # 在回测中，我们假设每期只追评分最高的那个蓝球
            recommended_blue = max(blue_scores, key=blue_scores.get)
            rank_band_hits = count_actual_reds_by_rank_band(red_scores, actual_red_set)
            
            # --- 在回测的每一步都重新应用完整的过滤流程 ---
            rejection_seed = rejection_seed_for_issue(
                config.random_seed, actual_draw['期号']
            )
            rejection_set = make_rejection_set(
                config.rejection_lib_size, random.Random(rejection_seed)
            )
            omission = get_omission(history_df_for_step)
            last_10 = [set(d) for d in history_df_for_step.iloc[-10:]['红球'].tolist()]
            last_1 = last_10[-1]; last_2 = last_10[-2]
            
            for mode in pool_modes:
                evaluate_backtest_mode(
                    mode, metrics[mode], actual_red_set, actual_blue,
                    recommended_blue, rank_band_hits, red_scores, omission,
                    last_10, last_1, last_2, rejection_set, config,
                )
            pbar.update(1)

    print("回测完成。\n")
    return {
        mode: value.to_result(len(backtest_range))
        for mode, value in metrics.items()
    }


# --- 5. 主执行逻辑 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='双色球策略分析与推荐')
    parser.add_argument('--backtest-periods', type=int, default=BACKTEST_PERIODS,
                        help='回测期数，默认 %(default)s')
    parser.add_argument('--rejection-size', type=int, default=REJECTION_LIB_SIZE,
                        help='反撞号随机库大小，默认 %(default)s')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help='随机种子，默认 %(default)s')
    parser.add_argument('--pool-mode', choices=('mixed', 'high', 'middle', 'low'),
                        default='mixed', help='红球候选池模式，默认 %(default)s')
    parser.add_argument('--compare-pools', action='store_true',
                        help='在一次回测中对比四种候选池模式')
    parser.add_argument('--non-interactive', action='store_true',
                        help='不等待键盘输入，适用于自动化运行')
    parser.add_argument('--rule-audit-periods', type=int, default=RULE_AUDIT_PERIODS,
                        help='统计规则对真实开奖覆盖率的期数，默认 %(default)s')
    args = parser.parse_args()
    if args.backtest_periods < 0 or args.rule_audit_periods < 0:
        parser.error('backtest-periods 和 rule-audit-periods 不能为负数')
    if not 0 <= args.rejection_size <= TOTAL_RED_COMBINATIONS:
        parser.error(f'rejection-size 必须在 0 到 {TOTAL_RED_COMBINATIONS} 之间')
    config = StrategyConfig(
        rejection_lib_size=args.rejection_size,
        random_seed=args.seed,
    )

    print("="*70)
    print("         双色球策略分析器 v7.0")
    print("="*70)

    # --- [阶段 1/8] 加载与特征工程 ---
    print("\n[阶段 1/8] 正在加载和处理历史数据...")
    full_df = load_and_preprocess_data()
    if full_df is None or len(full_df) < 50:
        raise SystemExit("错误: 历史数据加载失败或数据量过少（至少需要50期），程序终止。")
    full_df = feature_engineer(full_df)
    FEATURE_COLUMNS = [col for col in full_df.columns if col not in ['期号', '日期', '红球', '蓝球']]
    rule_coverage = audit_historical_rule_coverage(full_df, args.rule_audit_periods)
    hard_pipeline_coverage = audit_historical_hard_pipeline(
        full_df, args.rule_audit_periods
    )
    latest_issue = str(full_df.iloc[-1]['期号'])
    try:
        target_issue = infer_next_issue(latest_issue, full_df.iloc[-1]['日期'])
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"错误: 无法推导下一期期号: {exc}")
    print("数据加载与特征工程完成。")

    # --- [阶段 2/8] 执行严谨的历史回测 ---
    params_loaded = True
    try:
        with open(PARAMS_JSON_PATH, 'r') as f: 
            params = validate_strategy_params(json.load(f))
    except FileNotFoundError:
        params_loaded = False
        print(f"警告: 未找到参数文件 {PARAMS_JSON_PATH}，将使用内置的默认参数。")
        params = validate_strategy_params({})
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SystemExit(f"错误: 参数文件 {PARAMS_JSON_PATH} 无效: {exc}")
    # 执行回测并捕获其返回的统计结果
    pool_modes = ('mixed', 'high', 'middle', 'low') if args.compare_pools else (args.pool_mode,)
    backtests = run_full_backtest(
        full_df, params, FEATURE_COLUMNS, args.backtest_periods,
        pool_modes=pool_modes, config=config,
    )
    backtest = backtests[args.pool_mode]
    
    # --- [阶段 3/8] 训练最终预测模型 ---
    print("\n[阶段 3/8] 正在使用全部历史数据，训练用于最终预测的模型...")
    ml_training_df = full_df.iloc[5:].copy()
    final_ml_models_red, final_ml_models_blue = train_prediction_models(
        ml_training_df, FEATURE_COLUMNS, show_progress=True
    )
    try:
        validate_model_sets(final_ml_models_red, final_ml_models_blue)
    except ValueError as exc:
        raise SystemExit(f"错误: {exc}")
    
    # --- [阶段 4/8] 执行对下一期的预测 ---
    print("\n[阶段 4/8] 正在为下一期号码进行机器学习评分...")
    red_scores, blue_scores = run_strategy_and_get_scores(full_df, params, final_ml_models_red, final_ml_models_blue, FEATURE_COLUMNS)
    red_pool = build_red_pool(red_scores, config=config, mode=args.pool_mode)
    recommended_blues = sorted(
        blue_scores, key=blue_scores.get, reverse=True
    )[:config.blue_count]
    print(f"已根据ML评分选出 {config.pool_size_red} 个红球大底: {sorted(red_pool)}")

    # --- [阶段 5/8] 规则过滤 ---
    print("\n[阶段 5/8] 正在从大底中生成组合并应用硬规则过滤...")
    
    # 同一期可复现，不同期变化；与滚动回测采用完全相同的派生规则。
    rejection_seed = rejection_seed_for_issue(config.random_seed, target_issue)
    rejection_set = make_rejection_set(
        config.rejection_lib_size, random.Random(rejection_seed)
    )

    # 提前计算过滤所需的历史数据
    omission_values = get_omission(full_df)
    last_10_draws_sets = [set(d) for d in full_df.iloc[-10:]['红球'].tolist()]
    last_draw_set = last_10_draws_sets[-1]
    last_2_draw_set = last_10_draws_sets[-2]
    
    # 从红球大底中生成所有可能的6球组合
    potential_combos = list(combinations(sorted(red_pool), 6))
    
    # --- 核心过滤流程 (展开形式) ---
    passed_combos_tuples = []
    # 遍历所有由大底生成的潜在组合
    for r in tqdm(potential_combos, desc="规则过滤进度", ncols=80):
        # 硬规则负责淘汰组合，软规则在最终排序时参与评分。
        is_passed = passes_red_filters(r, omission_values, last_10_draws_sets,
                                       last_draw_set, last_2_draw_set, rejection_set)

        # 组合通过全部硬规则后进入候选列表。
        if is_passed:
            passed_combos_tuples.append(r)

    print(f"过滤完成！共有 {len(passed_combos_tuples)} 组号码通过硬规则检验。")
    pipeline_stats = filter_pipeline_stats(
        potential_combos, omission_values, last_10_draws_sets,
        last_draw_set, last_2_draw_set, rejection_set
    )

    # --- [阶段 6/8] 交互式输出 ---
    if 0 < len(passed_combos_tuples) < INTERACTIVE_THRESHOLD:
        print(f"\n通过检验的组合数量为 {len(passed_combos_tuples)} (低于{INTERACTIVE_THRESHOLD})，全部输出如下：")
        for i, combo in enumerate(passed_combos_tuples, 1): 
            print(f"  组合 {i:>2}: {' '.join(f'{n:02d}' for n in combo)}")
    elif len(passed_combos_tuples) >= INTERACTIVE_THRESHOLD and not args.non_interactive:
        input_thread = threading.Thread(target=get_user_input_with_timeout, args=(COUNTDOWN_SECONDS,))
        input_thread.start()
        input_thread.join()
        if user_input_flag == 'y':
            print("\n根据您的确认，输出所有通过检验的组合：")
            for i, combo in enumerate(passed_combos_tuples, 1): 
                print(f"  组合 {i:>3}: {' '.join(f'{n:02d}' for n in combo)}")

    # --- [阶段 7/8] 高级推荐 ---
    print("\n[阶段 7/8] 正在从最终组合中，生成高重合度的7红球大底...")
    best_7_reds = find_best_7_red_combinations(
        passed_combos_tuples, red_pool, red_scores, last_draw_set, last_2_draw_set
    )

    # --- [阶段 8/8] 最终报告 ---
    print("\n--- 正在生成最终推荐报告 ---")
    report_lines = []
    report_lines.append("="*60); report_lines.append("          双色球策略分析与推荐报告 (高级过滤版)"); report_lines.append("="*60)
    
    report_lines.append("\n--- 0. 报告元数据 ---")
    report_lines.append(f"Data_Basis_Issue: {latest_issue}")
    report_lines.append(f"Prediction_Target_Issue: {target_issue}")
    report_lines.append(f"报告生成时间: {local_now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    report_lines.append("\n--- 1. 策略参数与回测 ---")
    mode_desc = "加载已固化的参数" if params_loaded else "使用内置的默认参数"
    report_lines.append(f"模式: {mode_desc}")
    report_lines.append(f"  - anti_crowding_size  : {config.rejection_lib_size}")
    report_lines.append(
        f"  - anti_crowding_seed  : {config.random_seed} -> "
        f"{rejection_seed} (目标期派生)"
    )
    for key, val in params.items(): 
        report_lines.append(f"  - {key:<20}: {val}")
    
    report_lines.append(
        f"\n单式策略滚动回测 ({backtest.periods}期，每期最多"
        f"{config.recommendation_count}注，不含复式):"
    )
    report_lines.append(f"  - 候选池模式: {args.pool_mode}")
    report_lines.append(f"  - 成功建模评估期数: {backtest.evaluated_periods}")
    report_lines.append(f"  - 实际投注期数: {backtest.active_periods}")
    report_lines.append(f"  - 投注注数: {backtest.tickets}")
    report_lines.append(f"  - 候选池平均覆盖红球: {backtest.average_pool_red_hits:.2f}/6")
    report_lines.append(f"  - 单注平均命中红球: {backtest.average_ticket_red_hits:.3f}/6")
    report_lines.append(
        f"  - 候选全集平均命中红球: {backtest.average_candidate_red_hits:.3f}/6，"
        f"最终排序增益 {backtest.ranking_red_hit_delta:+.3f}"
    )
    report_lines.append(
        f"  - 命中至少3个红球: {backtest.three_plus_red_tickets} 注 "
        f"({backtest.three_plus_red_rate:.2%})"
    )
    report_lines.append(
        f"  - 候选全集3+红比例: {backtest.candidate_three_plus_red_rate:.2%}，"
        f"最终排序增益 {backtest.ranking_three_plus_delta:+.2%}"
    )
    report_lines.append(
        f"  - 最高分蓝球命中: {backtest.blue_hit_periods}/"
        f"{backtest.evaluated_periods} ({backtest.blue_hit_rate:.2%})"
    )
    report_lines.append(f"  - 总投入: {backtest.cost:.2f} 元")
    report_lines.append(f"  - 固定参考奖金: {backtest.winnings:.2f} 元")
    report_lines.append(f"  - 参考净收益: {backtest.profit:.2f} 元")
    report_lines.append(f"  - 参考回报率: {backtest.roi:.2%}")
    if len(backtests) > 1:
        report_lines.append("  - 候选池对照:")
        for name, result in backtests.items():
            report_lines.append(
                f"    {name:<6} 投入 {result.cost:>6.0f} 元，参考奖金 {result.winnings:>6.0f} 元，"
                f"参考净收益 {result.profit:>7.0f} 元，参考回报率 {result.roi:>7.2%}，"
                f"池覆盖 {result.average_pool_red_hits:.2f}/6，"
                f"单注红球 {result.average_ticket_red_hits:.3f}/6 "
                f"({result.ranking_red_hit_delta:+.3f})，"
                f"3+红 {result.three_plus_red_rate:.2%} "
                f"({result.ranking_three_plus_delta:+.2%})"
            )
    report_lines.append("  - 实际红球在模型评分排名中的分布:")
    for band, label in (
        ("high", "高端(1-4)"), ("middle", "中段(13-21)"),
        ("low", "低端(30-33)"), ("other", "其他"),
    ):
        band_width = len(RANK_BANDS[band]) if band in RANK_BANDS else RANK_OTHER_WIDTH
        report_lines.append(
            f"    {label:<13}: {backtest.rank_band_hits[band]:>3} 个 "
            f"(占比 {backtest.rank_band_rate(band):.2%}，"
            f"随机基线 {band_width / 33:.2%}，相对 {backtest.rank_band_lift(band):.2f}x)"
        )
    report_lines.append("中奖详情如下：")
    aggregated_counts = {name: 0 for name in set(PRIZE_NAMES.values())}
    for (red, blue), count in backtest.prize_counts.items():
        if count > 0: 
            prize_name = PRIZE_NAMES.get((red, blue))
            aggregated_counts[prize_name] += count
    
    if sum(aggregated_counts.values()) == 0:
        report_lines.append("  - 未中任何奖项。")
    else:
        for prize_name in ["一等奖", "二等奖", "三等奖", "四等奖", "五等奖", "六等奖"]:
            count = aggregated_counts.get(prize_name, 0)
            if count > 0:
                report_lines.append(f"  - {prize_name:<5}: {count} 次")

    report_lines.append("\n硬规则过滤统计 (按流水线累计):")
    for item in pipeline_stats:
        report_lines.append(
            f"  - {item['rule']:<22}: {item['before']} -> {item['remaining']} "
            f"(remove {item['removed']})"
        )
    inactive_rules = [item['rule'] for item in pipeline_stats if item['removed'] == 0]
    aggressive_rules = [
        item['rule'] for item in pipeline_stats
        if item['before'] and item['removed'] / item['before'] >= 0.5
    ]
    if inactive_rules:
        report_lines.append(f"  提示：本轮未淘汰组合的规则: {', '.join(inactive_rules)}")
    if aggressive_rules:
        report_lines.append(f"  提示：淘汰比例达到或超过50%的规则: {', '.join(aggressive_rules)}")

    report_lines.append(f"\n真实开奖规则覆盖率 (最近 {args.rule_audit_periods} 期，逐条独立统计):")
    for name in FILTER_NAMES:
        result = rule_coverage[name]
        rule_type = '硬' if name in HARD_FILTER_NAMES else '软'
        report_lines.append(
            f"  - [{rule_type}] {name:<22}: {result['passed']}/{result['total']} ({result['rate']:.2%})"
        )

    report_lines.append(
        f"\n真实开奖硬规则累计覆盖率 (最近 {args.rule_audit_periods} 期，不含随机撞号):"
    )
    for item in hard_pipeline_coverage['stages']:
        report_lines.append(
            f"  - {item['rule']:<22}: {item['before']} -> {item['remaining']} "
            f"(新增排除 {item['removed']} 期)"
        )
    report_lines.append(
        f"  - 合计保留: {hard_pipeline_coverage['passed']}/"
        f"{hard_pipeline_coverage['total']} ({hard_pipeline_coverage['rate']:.2%})"
    )

    report_lines.append("\n--- 2. 推荐组合 ---")
    top_blue = recommended_blues[0] if recommended_blues else "N/A"
    
    report_lines.append(
        f"\n【单式推荐 ({config.recommendation_count}组)】"
    )
    if passed_combos_tuples:
        final_selection = select_recommendations(
            passed_combos_tuples, red_scores, last_draw_set, last_2_draw_set,
            limit=config.recommendation_count,
        )
        for i, combo in enumerate(final_selection, 1):
            report_lines.append(f"  组合 {i:>2}: 红球 {list(combo)!s:<24} 蓝球 [{top_blue:02d}]")
    else:
        report_lines.append("  - 未能生成足够的单式组合。")
        
    report_lines.append("\n【7+N 复式推荐 (1组)】")
    if best_7_reds and recommended_blues:
        best_7_red_combo = list(best_7_reds[0][0])
        report_lines.append(f"  红球: {best_7_red_combo}")
        report_lines.append(f"  蓝球: {recommended_blues}")
    else:
        report_lines.append("  - 未能生成足够的复式组合。")

    report_lines.append("\n" + "="*60 + "\n报告结束。祝您好运！\n" + "="*60)
    
    final_report_string = "\n".join(report_lines)
    print("\n\n" + final_report_string)

    try:
        os.makedirs(REPORT_DIR, exist_ok=True)
        timestamp = local_now().strftime("%Y%m%d_%H%M%S")
        filename = f"ssq_analysis_output_{timestamp}.txt"
        filepath = os.path.join(REPORT_DIR, filename)
        atomic_write_text(filepath, final_report_string)
        print(f"\n\n报告已成功保存到文件: {filepath}")
    except OSError as e:
        raise SystemExit(f"\n\n写入报告文件失败: {e}")

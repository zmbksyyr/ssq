# --- 核心库导入 ---
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from math import comb

import lightgbm as lgb
import numpy as np
import pandas as pd
from ssq_config import (  # noqa: F401 - compatibility exports
    BACKTEST_PERIODS,
    COUNTDOWN_SECONDS,
    DEFAULT_PARAMS,
    DEFAULT_STRATEGY_CONFIG,
    INTERACTIVE_THRESHOLD,
    MAX_SHARED_RED_BALLS,
    NUM_BLUE_BALLS,
    NUM_RECOMMENDATIONS,
    POOL_SIZE_RED,
    RANDOM_SEED,
    RED_HIGH_COUNT,
    RED_LOW_COUNT,
    RED_POOL_MODES,
    REJECTION_LIB_SIZE,
    REJECTION_SEED_MULTIPLIER,
    RULE_AUDIT_PERIODS,
    TOTAL_RED_COMBINATIONS,
    AnalyzerOptions,
    LoadedStrategyParams,
    StrategyConfig,
    build_argument_parser,
    parse_cli_options,
    validate_strategy_params,
)
from ssq_core import (
    PRIZE_RULES,
    atomic_write_text,
    infer_next_issue,
    local_now,
    parse_blue_ball,
    parse_issue,
    parse_red_balls,
)
from ssq_reporting import AnalysisReportData, build_analysis_report
from ssq_rules import (
    FILTER_NAMES,
    RED_RULES,
    RuleContext,
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

@dataclass(frozen=True)
class RedCandidateSelection:
    red_pool: tuple[int, ...]
    potential_combos: tuple[tuple[int, ...], ...]
    passed_combos: tuple[tuple[int, ...], ...]
    recommendations: tuple[tuple[int, ...], ...]


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
        expected_rate = RANK_BAND_WIDTHS[band] / 33
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


def load_strategy_params(filepath=PARAMS_JSON_PATH):
    try:
        with open(filepath, encoding='utf-8') as handle:
            values = validate_strategy_params(json.load(handle))
    except FileNotFoundError:
        return LoadedStrategyParams(validate_strategy_params({}), False)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"参数文件 {filepath} 无效: {exc}") from exc
    return LoadedStrategyParams(values, True)


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
RANK_BAND_WIDTHS = {
    **{name: len(ranks) for name, ranks in RANK_BANDS.items()},
    'other': RANK_OTHER_WIDTH,
}


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


def generate_red_candidates(
    red_scores,
    context,
    rejection_set,
    config=DEFAULT_STRATEGY_CONFIG,
    mode="mixed",
    show_progress=False,
):
    """Run the shared red-ball selection pipeline for live runs and backtests."""
    red_pool = tuple(build_red_pool(red_scores, config=config, mode=mode))
    potential_combos = tuple(combinations(red_pool, 6))
    iterator = (
        tqdm(potential_combos, desc="规则过滤进度", ncols=80)
        if show_progress else potential_combos
    )
    passed_combos = tuple(
        combo for combo in iterator
        if passes_red_filters(combo, context, rejection_set)
    )
    recommendations = tuple(select_recommendations(
        passed_combos,
        red_scores,
        context.last_draw,
        context.previous_draw,
        limit=config.recommendation_count,
    ))
    return RedCandidateSelection(
        red_pool=red_pool,
        potential_combos=potential_combos,
        passed_combos=passed_combos,
        recommendations=recommendations,
    )


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
    return RuleContext(
        omission_values=get_omission(history),
        recent_draws=recent,
        last_draw=recent[-1],
        previous_draw=recent[-2],
    )


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
            combo, historical_rule_context(full_df, index)
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
            if not rule.evaluator(combo, context):
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


def is_confirmation_input(value):
    """Return whether terminal input explicitly confirms the prompt."""
    if isinstance(value, bytes):
        value = value.decode(errors='ignore')
    return value.strip().lower() == 'y'


def get_user_input_with_timeout(timeout):
    """Wait up to ``timeout`` seconds and return whether the user confirmed."""
    prompt = f"\n发现大量高质量组合。输入 'y' 并回车可在 {timeout} 秒内查看全部，否则将仅输出随机推荐...\n"
    sys.stdout.write(prompt)
    sys.stdout.flush()

    confirmed = False
    if 'msvcrt' in sys.modules:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if msvcrt.kbhit() and is_confirmation_input(msvcrt.getch()):
                confirmed = True
                break
            time.sleep(0.1)
    else:
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            confirmed = is_confirmation_input(sys.stdin.readline())

    sys.stdout.write("\n倒计时结束。\n")
    sys.stdout.flush()
    return confirmed

# --- 4. 核心功能模块 (回测与预测) ---

def evaluate_backtest_mode(mode, current, actual_red_set, actual_blue,
                           recommended_blue, rank_band_hits, red_scores,
                           context, rejection_set, config):
    """Evaluate one pool mode for one historical issue."""
    selection = generate_red_candidates(
        red_scores, context, rejection_set, config=config, mode=mode
    )
    current.evaluated_periods += 1
    current.pool_red_hits += len(set(selection.red_pool) & actual_red_set)
    current.rank_band_hits.update(rank_band_hits)
    if recommended_blue == actual_blue:
        current.blue_hit_periods += 1

    if not selection.passed_combos:
        return

    red_hits_by_combo = {
        combo: len(set(combo) & actual_red_set)
        for combo in selection.passed_combos
    }
    current.candidate_tickets += len(selection.passed_combos)
    current.candidate_red_hit_counts.update(red_hits_by_combo.values())
    current.active_periods += 1
    current.tickets += len(selection.recommendations)
    current.cost += len(selection.recommendations) * 2
    blue_hits = int(recommended_blue == actual_blue)
    for combo in selection.recommendations:
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
            context = RuleContext(
                omission_values=omission,
                recent_draws=last_10,
                last_draw=last_1,
                previous_draw=last_2,
            )
            
            for mode in pool_modes:
                evaluate_backtest_mode(
                    mode, metrics[mode], actual_red_set, actual_blue,
                    recommended_blue, rank_band_hits, red_scores, context,
                    rejection_set, config,
                )
            pbar.update(1)

    print("回测完成。\n")
    return {
        mode: value.to_result(len(backtest_range))
        for mode, value in metrics.items()
    }


# --- 5. 主执行逻辑 ---
if __name__ == '__main__':
    options = parse_cli_options()
    config = options.strategy_config

    print("="*70)
    print("         双色球策略分析器 v7.0")
    print("="*70)

    # --- [阶段 1/8] 加载与特征工程 ---
    print("\n[阶段 1/8] 正在加载和处理历史数据...")
    full_df = load_and_preprocess_data()
    if full_df is None or len(full_df) < 50:
        raise SystemExit("错误: 历史数据加载失败或数据量过少（至少需要50期），程序终止。")
    full_df = feature_engineer(full_df)
    feature_columns = [col for col in full_df.columns if col not in ['期号', '日期', '红球', '蓝球']]
    rule_coverage = audit_historical_rule_coverage(
        full_df, options.rule_audit_periods
    )
    hard_pipeline_coverage = audit_historical_hard_pipeline(
        full_df, options.rule_audit_periods
    )
    latest_issue = str(full_df.iloc[-1]['期号'])
    try:
        target_issue = infer_next_issue(latest_issue, full_df.iloc[-1]['日期'])
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"错误: 无法推导下一期期号: {exc}")
    print("数据加载与特征工程完成。")

    # --- [阶段 2/8] 执行严谨的历史回测 ---
    try:
        loaded_params = load_strategy_params()
    except ValueError as exc:
        raise SystemExit(f"错误: {exc}") from exc
    params = loaded_params.values
    params_loaded = loaded_params.loaded_from_file
    if not params_loaded:
        print(f"警告: 未找到参数文件 {PARAMS_JSON_PATH}，将使用内置的默认参数。")
    # 执行回测并捕获其返回的统计结果
    backtests = run_full_backtest(
        full_df, params, feature_columns, options.backtest_periods,
        pool_modes=options.backtest_pool_modes, config=config,
    )
    backtest = backtests[options.pool_mode]
    
    # --- [阶段 3/8] 训练最终预测模型 ---
    print("\n[阶段 3/8] 正在使用全部历史数据，训练用于最终预测的模型...")
    ml_training_df = full_df.iloc[5:].copy()
    final_ml_models_red, final_ml_models_blue = train_prediction_models(
        ml_training_df, feature_columns, show_progress=True
    )
    try:
        validate_model_sets(final_ml_models_red, final_ml_models_blue)
    except ValueError as exc:
        raise SystemExit(f"错误: {exc}")
    
    # --- [阶段 4/8] 执行对下一期的预测 ---
    print("\n[阶段 4/8] 正在为下一期号码进行机器学习评分...")
    red_scores, blue_scores = run_strategy_and_get_scores(
        full_df, params, final_ml_models_red, final_ml_models_blue,
        feature_columns,
    )
    recommended_blues = sorted(
        blue_scores, key=blue_scores.get, reverse=True
    )[:config.blue_count]

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
    context = RuleContext(
        omission_values=omission_values,
        recent_draws=last_10_draws_sets,
        last_draw=last_draw_set,
        previous_draw=last_2_draw_set,
    )
    
    selection = generate_red_candidates(
        red_scores,
        context,
        rejection_set,
        config=config,
        mode=options.pool_mode,
        show_progress=True,
    )
    red_pool = selection.red_pool
    potential_combos = selection.potential_combos
    passed_combos_tuples = selection.passed_combos

    print(f"已根据ML评分选出 {config.pool_size_red} 个红球大底: {list(red_pool)}")
    print(f"过滤完成！共有 {len(passed_combos_tuples)} 组号码通过硬规则检验。")
    pipeline_stats = filter_pipeline_stats(
        potential_combos, context, rejection_set
    )

    # --- [阶段 6/8] 交互式输出 ---
    if 0 < len(passed_combos_tuples) < INTERACTIVE_THRESHOLD:
        print(f"\n通过检验的组合数量为 {len(passed_combos_tuples)} (低于{INTERACTIVE_THRESHOLD})，全部输出如下：")
        for i, combo in enumerate(passed_combos_tuples, 1): 
            print(f"  组合 {i:>2}: {' '.join(f'{n:02d}' for n in combo)}")
    elif (
        len(passed_combos_tuples) >= INTERACTIVE_THRESHOLD
        and not options.non_interactive
    ):
        if get_user_input_with_timeout(COUNTDOWN_SECONDS):
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
    report_data = AnalysisReportData(
        latest_issue=latest_issue,
        target_issue=target_issue,
        generated_at=local_now(),
        params_loaded=params_loaded,
        params=params,
        config=config,
        rejection_seed=rejection_seed,
        backtest=backtest,
        backtests=backtests,
        pool_mode=options.pool_mode,
        rank_band_widths=RANK_BAND_WIDTHS,
        pipeline_stats=pipeline_stats,
        rule_coverage=rule_coverage,
        hard_pipeline_coverage=hard_pipeline_coverage,
        rule_audit_periods=options.rule_audit_periods,
        selection=selection,
        recommended_blues=recommended_blues,
        best_7_reds=best_7_reds,
    )
    final_report_string = build_analysis_report(report_data)
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

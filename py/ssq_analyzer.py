# --- 核心库导入 ---
import pandas as pd
import numpy as np
import lightgbm as lgb
from itertools import combinations
from collections import Counter
import datetime
from tqdm import tqdm
import os
import json
import random
import time
import threading
import sys
import argparse
from dataclasses import dataclass
from ssq_core import PRIZE_NAMES, PRIZE_RULES, parse_blue_ball, parse_red_balls

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

# --- 回测与输出参数 ---
# 执行历史回测时，使用最近的多少期数据进行验证。
BACKTEST_PERIODS = 50       

# 当通过所有规则检验的组合数量超过此阈值时，程序会暂停并询问用户是否要全部显示。
# 这是一个防止刷屏的机制。
INTERACTIVE_THRESHOLD = 100 

# 在上面的交互式询问中，给用户的倒计时秒数。
COUNTDOWN_SECONDS = 10      

# 在最终报告里展示多少注高分单式推荐。
NUM_RECOMMENDATIONS = 10
RULE_AUDIT_PERIODS = 200
TOTAL_RED_COMBINATIONS = 1_107_568

FILTER_NAMES = (
    'highly_regular', 'sum_value', 'span', 'consecutive_numbers', 'zones',
    'ac_value', 'prime_composite_ratio', 'big_small_ratio', 'recent_overlap',
    'all_cold', 'odd_even_ratio', 'modulo3_roads', 'ending_digits',
    'head_tail_range', 'sum_of_tails', 'related_numbers',
    'diagonal_consecutive',
)

HARD_FILTER_NAMES = (
    'highly_regular', 'sum_value', 'span', 'consecutive_numbers', 'zones',
    'recent_overlap', 'all_cold', 'ending_digits', 'sum_of_tails',
    'related_numbers',
)

SOFT_FILTER_NAMES = tuple(name for name in FILTER_NAMES if name not in HARD_FILTER_NAMES)

DEFAULT_PARAMS = {
    'decay_factor': 0.995,
    'weight_freq': 0.3,
    'weight_omission': 0.4,
    'weight_ml': 0.3,
    'hot_lookback': 10,
    'hot_threshold': 2,
    'hot_bonus': 1.0,
    'cold_lookback': 30,
    'cold_bonus': 1.0,
    'repeat_bonus': 1.0,
    'weight_blue_freq': 0.5,
    'weight_blue_ml': 0.5,
}


def validate_strategy_params(params):
    merged = {**DEFAULT_PARAMS, **params}
    if not 0 < merged['decay_factor'] <= 1:
        raise ValueError('decay_factor 必须在 (0, 1] 范围内')
    for group in (
        ('weight_freq', 'weight_omission', 'weight_ml'),
        ('weight_blue_freq', 'weight_blue_ml'),
    ):
        values = [float(merged[name]) for name in group]
        if any(value < 0 for value in values) or not np.isclose(sum(values), 1.0):
            raise ValueError(f"权重 {', '.join(group)} 必须非负且总和为 1")
    for name in ('hot_lookback', 'hot_threshold', 'cold_lookback'):
        if int(merged[name]) < 0:
            raise ValueError(f'{name} 不能为负数')
    for name in ('hot_bonus', 'cold_bonus', 'repeat_bonus'):
        if float(merged[name]) <= 0:
            raise ValueError(f'{name} 必须大于 0')
    return merged


@dataclass(frozen=True)
class StrategyConfig:
    """All selection knobs live here so backtests and live runs share them."""
    pool_size_red: int = POOL_SIZE_RED
    high_count: int = RED_HIGH_COUNT
    low_count: int = RED_LOW_COUNT
    blue_count: int = NUM_BLUE_BALLS
    rejection_lib_size: int = REJECTION_LIB_SIZE
    random_seed: int = RANDOM_SEED


@dataclass(frozen=True)
class BacktestResult:
    periods: int
    active_periods: int
    tickets: int
    cost: int
    winnings: int
    prize_counts: Counter

    @property
    def profit(self):
        return self.winnings - self.cost

    @property
    def roi(self):
        return self.winnings / self.cost if self.cost else 0.0

# --- 文件路径设置 (采纳 ssq2 的动态路径方案) ---
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
    except Exception as e:
        # 如果文件读取失败（例如文件不存在、格式错误），打印错误信息并中断程序
        print(f"错误: 无法加载数据文件 '{filepath}': {e}")
        return None
    
    try:
        df['期号'] = pd.to_numeric(df['期号'], errors='raise').astype(int)
        if df['期号'].duplicated().any():
            raise ValueError("存在重复期号")
        pd.to_datetime(df['日期'], format='%Y-%m-%d', errors='raise')
        df['红球'] = df['红球'].apply(parse_red_balls)
        df['蓝球'] = df['蓝球'].apply(parse_blue_ball)
    except (TypeError, ValueError) as exc:
        print(f"错误: 数据文件 '{filepath}' 校验失败: {exc}")
        return None

    return df.sort_values('期号').reset_index(drop=True)

def feature_engineer(df):
    """
    为数据集进行特征工程，基于历史数据计算出各种可能影响下一期结果的统计指标。
    这些指标将作为机器学习模型的输入特征。

    Args:
        df (DataFrame): 输入的包含'红球'和'蓝球'列的数据。

    Returns:
        DataFrame: 增加了18个新特征列的数据。
    """
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
    df['red_ac_value'] = df['红球'].apply(lambda nums: len(set(abs(n1-n2) for n1, n2 in combinations(nums, 2))))
    # 特征13: 尾数唯一值个数 - 6个号码的个位数有多少种不同的值
    df['red_tail_uniques'] = df['红球'].apply(lambda x: len(set(n % 10 for n in x)))
    
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
    red_omission = {}
    total_draws = len(df)
    # 遍历1到33号红球
    for i in range(1, 34):
        # 查找号码 i 最后一次出现的行的索引位置
        last_occurrence = df[df['红球'].apply(lambda x: i in x)].index.max()
        # 如果号码从未出现过(last_occurrence为NaN)，则其遗漏值为总期数
        if pd.isna(last_occurrence):
            omission = total_draws
        else:
            # 否则，遗漏值为 (总期数 - 最后出现位置的索引 - 1)
            omission = total_draws - last_occurrence - 1
        red_omission[i] = omission
    return red_omission

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
    red_ml_probs = {ball: ml_models_red[ball].predict_proba(last_features)[:, 1][0] for ball in range(1, 34)}
    
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
    blue_ml_probs = {ball: ml_models_blue[ball].predict_proba(last_features)[:, 1][0] for ball in range(1, 17)}
    
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

# 预计算1-33中的质数，避免在函数内重复计算
PRIMES_IN_33 = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
def build_red_pool(red_scores, config=StrategyConfig(), mode="mixed"):
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
    return sorted(list(dict.fromkeys(high + middle + low)))


def make_rejection_set(size, rng=None):
    """Create a reproducible anti-crowding sample of six-red combinations."""
    if not 0 <= size <= TOTAL_RED_COMBINATIONS:
        raise ValueError(f"rejection size must be between 0 and {TOTAL_RED_COMBINATIONS}")
    rng = rng or random.Random(RANDOM_SEED)
    rejection_set = set()
    while len(rejection_set) < size:
        rejection_set.add(tuple(sorted(rng.sample(range(1, 34), 6))))
    return rejection_set


def passes_red_filters(combo, omission_values, last_10_draws_sets, last_draw_set,
                       last_2_draw_set, rejection_set=None):
    """Single source of truth for live selection and historical backtests."""
    return (
        filter_highly_regular(combo) and filter_sum_value(combo) and
        filter_span(combo) and filter_consecutive_numbers(combo) and
        filter_zones(combo) and
        filter_recent_overlap(combo, last_10_draws_sets) and
        filter_all_cold(combo, omission_values) and filter_ending_digits(combo) and
        filter_sum_of_tails(combo) and
        filter_related_numbers(combo, last_draw_set) and
        (rejection_set is None or combo not in rejection_set)
    )




def is_prime(n):
    """判断一个数字是否是预定义的质数。"""
    return n in PRIMES_IN_33

def calculate_ac_value(r): 
    """计算AC值（算术复杂度），并进行一个自定义调整 (-5)。"""
    standard_ac = len(set(abs(n1 - n2) for n1, n2 in combinations(r, 2)))
    return standard_ac - 5 # 减5是一个经验调整

def filter_highly_regular(r): 
    """规则1: 过滤掉高度规律的组合 (例如等差数列 '2 4 6 8 10 12')。
       逻辑: 如果所有相邻数字的差值都一样，则该组合只有一个差值，集合长度为1，被过滤。
       返回: True=保留, False=过滤
    """
    return len(set(r[i+1] - r[i] for i in range(len(r) - 1))) > 1

def filter_sum_value(r): 
    """规则2: 过滤和值。大部分中奖号码的和值集中在中间区域。
       逻辑: 和值必须在 70 到 160 之间。
       返回: True=保留, False=过滤
    """
    return 70 <= sum(r) <= 160

def filter_span(r): 
    """规则3: 过滤跨度。跨度指最大号与最小号之差。
       逻辑: 跨度必须大于等于 15，过滤掉号码过于集中的组合。
       返回: True=保留, False=过滤
    """
    return (r[-1] - r[0]) >= 15

def filter_consecutive_numbers(r):
    """规则4: 过滤连号。不允许出现过多的连号。
       逻辑: 不允许出现3组连号，也不允许出现4连号及以上的情况。
       返回: True=保留, False=过滤
    """
    groups = 0      # 连号的组数
    max_c = 0       # 最大连号的长度
    current_c = 1   # 当前正在计算的连号长度
    for i in range(len(r) - 1):
        if r[i+1] - r[i] == 1: 
            current_c += 1
        else:
            if current_c >= 2: # 如果前一个数字是连号的结尾
                groups += 1
                max_c = max(max_c, current_c)
            current_c = 1 # 重置连号计数
    if current_c >= 2: # 检查最后一组数是否是连号
        groups += 1
        max_c = max(max_c, current_c)
    # 如果连号组数>=3 或 最大连号长度>=4，则过滤掉
    return not (groups >= 3 or max_c >= 4)

def filter_zones(r): 
    """规则5: 过滤三区分布。不允许所有号码都集中在同一个区域。
       逻辑: 检查是否所有号码都在小区(1-11)、或中区(12-22)、或大区(23-33)。
       返回: True=保留, False=过滤
    """
    all_in_small = all(b <= 11 for b in r)
    all_in_medium = all(12 <= b <= 22 for b in r)
    all_in_large = all(b >= 23 for b in r)
    return not (all_in_small or all_in_medium or all_in_large)

def filter_ac_value(r): 
    """规则6: 过滤AC值。
       逻辑: 自定义AC值必须在 6 到 10 之间 (相当于标准AC值的 11-15)。
       返回: True=保留, False=过滤
    """
    return 6 <= calculate_ac_value(r) <= 10

def filter_prime_composite_ratio(r): 
    """规则7: 过滤质合比。不允许质数个数极端（过多或过少）。
       逻辑: 排除质数个数为 0, 1, 5, 6 的组合。
       返回: True=保留, False=过滤
    """
    prime_count = sum(1 for ball in r if is_prime(ball))
    return prime_count not in [0, 1, 5, 6]

def filter_big_small_ratio(r): 
    """规则8: 过滤大小比。不允许大数或小数过多。以16为界。
       逻辑: 排除小数(<=16)个数为 0, 1, 5, 6 的组合。
       返回: True=保留, False=过滤
    """
    small_count = sum(1 for ball in r if ball <= 16)
    return small_count not in [0, 1, 5, 6]

def filter_recent_overlap(r, last_10_draws_sets):
    """规则9: 过滤历史重号。不允许与最近10期内任一期号码重合数过多。
       逻辑: 如果与最近10期任一期的交集达到4个或以上，则过滤。
       返回: True=保留, False=过滤
    """
    candidate_set = set(r)
    for draw_set in last_10_draws_sets:
        if len(candidate_set.intersection(draw_set)) >= 4:
            return False # 重合过多，过滤
    return True

def filter_all_cold(r, omission_values): 
    """规则10: 过滤全冷号组合。
       逻辑: 不允许组合中所有6个号码都是遗漏值大于15的冷号。
       返回: True=保留, False=过滤
    """
    return not all(omission_values.get(ball, 0) > 15 for ball in r)

def filter_odd_even_ratio(r): 
    """规则11: 过滤奇偶比。不允许奇偶比极端（全奇、全偶或接近全奇全偶）。
       逻辑: 排除偶数个数为 0, 1, 5, 6 的组合。
       返回: True=保留, False=过滤
    """
    even_count = sum(1 for n in r if n % 2 == 0)
    return even_count not in [0, 1, 5, 6]

def filter_modulo3_roads(r): 
    """规则12: 过滤除3余数。要求0路、1路、2路号码都必须出现。
       逻辑: {n % 3 for n in r} 会得到包含所有余数的集合，其长度必须为3。
       返回: True=保留, False=过滤
    """
    return len({n % 3 for n in r}) == 3

def filter_ending_digits(r):
    """规则13: 过滤尾数分布。不允许出现过多同尾号，或尾数种类过少。
       逻辑: 任一尾数出现次数不能超过2次，且不同尾数的种类不能少于3种。
       返回: True=保留, False=过滤
    """
    tails = [n % 10 for n in r]
    counts = Counter(tails) # 统计每个尾数出现的次数
    return not (max(counts.values()) >= 3 or len(counts) <= 2)

def filter_head_tail_range(r): 
    """规则14: 过滤首尾号码范围。龙头(第一个号)不宜过大，凤尾(最后一个号)不宜过小。
       逻辑: 龙头不能大于10，凤尾不能小于25。
       返回: True=保留, False=过滤
    """
    return not (r[0] > 10 or r[-1] < 25)

def filter_sum_of_tails(r): 
    """规则15: 过滤尾数和值。所有号码的个位数之和应在一个合理范围。
       逻辑: 尾数和应在 15 到 45 之间。
       返回: True=保留, False=过滤
    """
    return 15 <= sum(n % 10 for n in r) <= 45

def filter_related_numbers(r, last_draw_set):
    """规则16: 过滤关联码。要求组合与上期号码至少有1个关联。
       逻辑: 必须包含至少1个重号(与上期相同)或边号(与上期号码加减1)。
       返回: True=保留, False=过滤
    """
    combo_set = set(r)
    # 重号：与上期相同的号码
    repeats = combo_set.intersection(last_draw_set)
    # 边号：与上期号码加减1的号码
    adjacents = combo_set.intersection({n - 1 for n in last_draw_set} | {n + 1 for n in last_draw_set})
    # 如果重号和边号的总数大于0，则保留
    return (len(repeats) + len(adjacents)) > 0

def filter_diagonal_consecutive(r, last_draw_set, last_2_draw_set):
    """规则17: 过滤斜连号。例如，上上期有10，上期有11，本期组合中不应出现12。
       逻辑: 检查组合中是否存在号码 n，使得 n-1 在上期开奖中，n-2 在上上期开奖中。
       返回: True=保留, False=过滤
    """
    for n in r:
        if (n - 1) in last_draw_set and (n - 2) in last_2_draw_set:
            return False # 发现斜连号，过滤
    return True


def explain_filter_failures(combo, omission_values, last_10_draws_sets,
                            last_draw_set, last_2_draw_set, rejection_set=None):
    """Return the names of every rule that rejects a combination."""
    checks = (
        ("highly_regular", filter_highly_regular(combo)),
        ("sum_value", filter_sum_value(combo)),
        ("span", filter_span(combo)),
        ("consecutive_numbers", filter_consecutive_numbers(combo)),
        ("zones", filter_zones(combo)),
        ("ac_value", filter_ac_value(combo)),
        ("prime_composite_ratio", filter_prime_composite_ratio(combo)),
        ("big_small_ratio", filter_big_small_ratio(combo)),
        ("recent_overlap", filter_recent_overlap(combo, last_10_draws_sets)),
        ("all_cold", filter_all_cold(combo, omission_values)),
        ("odd_even_ratio", filter_odd_even_ratio(combo)),
        ("modulo3_roads", filter_modulo3_roads(combo)),
        ("ending_digits", filter_ending_digits(combo)),
        ("head_tail_range", filter_head_tail_range(combo)),
        ("sum_of_tails", filter_sum_of_tails(combo)),
        ("related_numbers", filter_related_numbers(combo, last_draw_set)),
        ("diagonal_consecutive", filter_diagonal_consecutive(combo, last_draw_set, last_2_draw_set)),
        ("anti_crowding", rejection_set is None or combo not in rejection_set),
    )
    return [name for name, passed in checks if not passed]


def filter_pipeline_stats(combos, omission_values, last_10_draws_sets,
                          last_draw_set, last_2_draw_set, rejection_set=None):
    """Measure each rule's incremental impact in the configured pipeline."""
    checks = (
        ("highly_regular", lambda c: filter_highly_regular(c)),
        ("sum_value", lambda c: filter_sum_value(c)),
        ("span", lambda c: filter_span(c)),
        ("consecutive_numbers", lambda c: filter_consecutive_numbers(c)),
        ("zones", lambda c: filter_zones(c)),
        ("recent_overlap", lambda c: filter_recent_overlap(c, last_10_draws_sets)),
        ("all_cold", lambda c: filter_all_cold(c, omission_values)),
        ("ending_digits", lambda c: filter_ending_digits(c)),
        ("sum_of_tails", lambda c: filter_sum_of_tails(c)),
        ("related_numbers", lambda c: filter_related_numbers(c, last_draw_set)),
        ("anti_crowding", lambda c: rejection_set is None or c not in rejection_set),
    )
    remaining = list(combos)
    stats = []
    for name, check in checks:
        before = len(remaining)
        remaining = [combo for combo in remaining if check(combo)]
        stats.append({"rule": name, "before": before, "removed": before - len(remaining),
                      "remaining": len(remaining)})
    return stats


def audit_historical_rule_coverage(full_df, periods=RULE_AUDIT_PERIODS):
    """Measure how often each strategy rule accepts actual historical draws."""
    if periods <= 0 or len(full_df) < 11:
        return {name: {'passed': 0, 'total': 0, 'rate': 0.0} for name in FILTER_NAMES}
    start = max(10, len(full_df) - periods)
    passed_counts = Counter()
    total = 0
    for index in range(start, len(full_df)):
        history = full_df.iloc[:index]
        combo = tuple(full_df.iloc[index]['红球'])
        last_10 = [set(draw) for draw in history.iloc[-10:]['红球']]
        failures = set(explain_filter_failures(
            combo, get_omission(history), last_10, last_10[-1], last_10[-2], None
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


def score_red_combination(combo, red_scores, last_draw_set=None, last_2_draw_set=None):
    """Rank already-valid combinations using soft, explainable preferences."""
    values = [red_scores.get(ball, 0.0) for ball in combo]
    max_score = max(red_scores.values(), default=1.0) or 1.0
    signal = sum(values) / (len(combo) * max_score)
    odd_even_balance = 1.0 - abs(sum(n % 2 for n in combo) - 3) / 3
    zone_counts = (
        sum(n <= 11 for n in combo),
        sum(12 <= n <= 22 for n in combo),
        sum(n >= 23 for n in combo),
    )
    zone_balance = 1.0 - (max(zone_counts) - min(zone_counts)) / 6
    prime_balance = 1.0 - abs(sum(is_prime(n) for n in combo) - 3) / 3
    small_count = sum(n <= 16 for n in combo)
    big_small_balance = 1.0 - abs(small_count - 3) / 3
    ac_score = 1.0 if filter_ac_value(combo) else 0.0
    modulo_score = 1.0 if filter_modulo3_roads(combo) else 0.0
    head_tail_score = 1.0 if filter_head_tail_range(combo) else 0.0
    diagonal_score = 1.0
    if last_draw_set is not None and last_2_draw_set is not None:
        diagonal_score = 1.0 if filter_diagonal_consecutive(
            combo, last_draw_set, last_2_draw_set
        ) else 0.0
    return (
        0.50 * signal + 0.10 * odd_even_balance + 0.10 * zone_balance +
        0.08 * prime_balance + 0.08 * big_small_balance + 0.05 * ac_score +
        0.04 * modulo_score + 0.03 * head_tail_score + 0.02 * diagonal_score
    )

def find_best_7_red_combinations(passed_combos_tuples, red_pool):
    """
    从所有通过规则检验的6红球组合中，提炼出覆盖度最高的7红球“小复式”组合。
    覆盖度指一个7红球组合能拆分出多少个有效的6红球组合。

    Args:
        passed_combos_tuples (list): 包含所有通过检验的6红球元组的列表。
        red_pool (list): 机器学习筛选出的红球大底。

    Returns:
        list: 一个排序后的列表，每个元素是 ((7红球元组), 覆盖度)。
    """
    if not passed_combos_tuples:
        return []
    
    # 将列表转为集合以获得O(1)的查找速度，极大提升效率
    passed_combos_set = set(passed_combos_tuples)
    
    best_7_red_combos = {} # 用于存储找到的7红球组合及其覆盖度
    
    # 遍历每一个通过筛选的6球组合作为“种子”
    for seed_combo in tqdm(passed_combos_set, desc="生成7红球大底", leave=False, ncols=80):
        seed_set = set(seed_combo)
        best_7th_ball = -1
        max_coverage = 0
        # 候选的第7个球，是红球大底中除了种子6个球之外的球
        candidate_balls = set(red_pool) - seed_set
        
        # 尝试将每个候选球加入种子，形成一个7球组合，并计算其覆盖度
        for ball in candidate_balls:
            temp_7_red_set = seed_set.union({ball})
            coverage = 0
            # 从这个7球组合中拆分出所有可能的6球组合
            for sub_combo in combinations(temp_7_red_set, 6):
                # 如果拆分出的组合在“通过检验的组合集合”中，则覆盖度+1
                if sub_combo in passed_combos_set:
                    coverage = coverage + 1
            # 找到能使覆盖度最大的第7个球
            if coverage > max_coverage:
                max_coverage = coverage
                best_7th_ball = ball
        
        # 如果找到了一个有效的第7球
        if best_7th_ball != -1:
            # 形成最终的7球组合
            final_7_combo = tuple(sorted(list(seed_set) + [best_7th_ball]))
            # 存入字典，如果已存在则不更新（避免重复计算）
            if final_7_combo not in best_7_red_combos:
                best_7_red_combos[final_7_combo] = max_coverage
                
    # 按覆盖度从高到低排序
    sorted_results = sorted(best_7_red_combos.items(), key=lambda item: item[1], reverse=True)
    return sorted_results

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

def run_full_backtest(full_df, params, feature_columns, num_periods, pool_modes=("mixed",)):
    """
    [逻辑已修正] 对最近N期执行逻辑严谨的策略回测。
    核心修正: 此函数在回测的每一步，都仅使用当前步之前的数据重新训练模型，避免了“未来数据”的泄露。
    返回每种候选池模式对应的 BacktestResult。
    """
    print("\n" + "="*70)
    print(f"        [新功能] 最近 {num_periods} 期完整策略回测 (逻辑严谨版)")
    print("="*70)
    
    # 检查是否有足够的数据进行回测 (需要回测期数 + 至少50期用于模型初次训练)
    if len(full_df) < num_periods + 50:
        print(f"历史数据不足 {num_periods + 50} 期，无法执行回测。跳过此步骤。")
        return {mode: BacktestResult(0, 0, 0, 0, 0, Counter()) for mode in pool_modes}

    # 初始化统计变量
    metrics = {
        mode: {"prize_counts": Counter(), "cost": 0, "winnings": 0,
               "active_periods": 0, "tickets": 0}
        for mode in pool_modes
    }
    
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
            
            local_ml_models_red, local_ml_models_blue = {}, {} # 使用本次循环的局部模型
            
            # 训练红球模型
            for ball_r in range(1, 34):
                training_data_for_step[f'red_{ball_r}_next'] = training_data_for_step['红球'].apply(lambda x: 1 if ball_r in x else 0).shift(-1)
                df_temp_r = training_data_for_step.dropna(subset=feature_columns + [f'red_{ball_r}_next'])
                X_r, y_r = df_temp_r[feature_columns], df_temp_r[f'red_{ball_r}_next']
                if not X_r.empty: 
                    lgb_clf_r = lgb.LGBMClassifier(random_state=42, verbose=-1)
                    lgb_clf_r.fit(X_r, y_r)
                    local_ml_models_red[ball_r] = lgb_clf_r
            
            # 训练蓝球模型
            for ball_b in range(1, 17):
                training_data_for_step[f'blue_{ball_b}_next'] = training_data_for_step['蓝球'].apply(lambda x: 1 if x == ball_b else 0).shift(-1)
                df_temp_b = training_data_for_step.dropna(subset=feature_columns + [f'blue_{ball_b}_next'])
                X_b, y_b = df_temp_b[feature_columns], df_temp_b[f'blue_{ball_b}_next']
                if not X_b.empty: 
                    lgb_clf_b = lgb.LGBMClassifier(random_state=42, verbose=-1)
                    lgb_clf_b.fit(X_b, y_b)
                    local_ml_models_blue[ball_b] = lgb_clf_b
            
            if len(local_ml_models_red) != 33 or len(local_ml_models_blue) != 16: # 如果模型训练不完整，跳过
                pbar.update(1)
                continue
            # --- 核心修正部分结束 ---

            # 2. 使用刚刚训练好的【局部模型】进行评分和筛选
            red_scores, blue_scores = run_strategy_and_get_scores(history_df_for_step, params, local_ml_models_red, local_ml_models_blue, feature_columns)
            
            # 在回测中，我们假设每期只追评分最高的那个蓝球
            recommended_blue = sorted(blue_scores, key=blue_scores.get, reverse=True)[0]
            
            # --- 在回测的每一步都重新应用完整的过滤流程 ---
            rejection_set = make_rejection_set(REJECTION_LIB_SIZE, random.Random(RANDOM_SEED + i))
            omission = get_omission(history_df_for_step)
            last_10 = [set(d) for d in history_df_for_step.iloc[-10:]['红球'].tolist()]
            last_1 = last_10[-1]; last_2 = last_10[-2]
            
            for mode in pool_modes:
                red_pool = build_red_pool(red_scores, mode=mode)
                passed_combos = [
                    combo for combo in combinations(sorted(red_pool), 6)
                    if passes_red_filters(combo, omission, last_10, last_1, last_2, rejection_set)
                ]
                if not passed_combos:
                    continue
                selected_combos = sorted(
                    passed_combos,
                    key=lambda combo: (
                        -score_red_combination(combo, red_scores, last_1, last_2), combo
                    )
                )[:NUM_RECOMMENDATIONS]
                current = metrics[mode]
                current["active_periods"] += 1
                current["tickets"] += len(selected_combos)
                current["cost"] += len(selected_combos) * 2
                for combo in selected_combos:
                    hit_key = (
                        len(set(combo) & actual_red_set),
                        1 if recommended_blue == actual_blue else 0,
                    )
                    prize = PRIZE_RULES.get(hit_key, 0)
                    if prize > 0:
                        current["winnings"] += prize
                        current["prize_counts"][hit_key] += 1
            pbar.update(1)

    print("回测完成。\n")
    return {
        mode: BacktestResult(
            periods=len(backtest_range), active_periods=value["active_periods"],
            tickets=value["tickets"], cost=value["cost"], winnings=value["winnings"],
            prize_counts=value["prize_counts"]
        )
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
    BACKTEST_PERIODS = args.backtest_periods
    REJECTION_LIB_SIZE = args.rejection_size
    RANDOM_SEED = args.seed
    random.seed(RANDOM_SEED)

    print("="*70)
    print("         双色球整合策略分析脚本 (v6.0 - 完全展开注释版)")
    print("="*70)

    # --- [阶段 1/8] 加载与特征工程 ---
    print("\n[阶段 1/8] 正在加载和处理历史数据...")
    full_df = load_and_preprocess_data()
    if full_df is None or len(full_df) < 50:
        exit("错误: 历史数据加载失败或数据量过少（至少需要50期），程序终止。")
    full_df = feature_engineer(full_df)
    FEATURE_COLUMNS = [col for col in full_df.columns if col not in ['期号', '日期', '红球', '蓝球']]
    rule_coverage = audit_historical_rule_coverage(full_df, args.rule_audit_periods)
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
        full_df, params, FEATURE_COLUMNS, BACKTEST_PERIODS, pool_modes=pool_modes
    )
    backtest = backtests[args.pool_mode]
    
    # --- [阶段 3/8] 训练最终预测模型 ---
    print("\n[阶段 3/8] 正在使用全部历史数据，训练用于最终预测的模型...")
    final_ml_models_red = {}
    final_ml_models_blue = {}
    ml_training_df = full_df.iloc[5:].copy()
    
    # 训练33个独立的红球模型
    for i in tqdm(range(1, 34), desc="训练最终红球模型", ncols=80):
        ml_training_df[f'red_{i}_next'] = ml_training_df['红球'].apply(lambda x: 1 if i in x else 0).shift(-1)
        df_temp = ml_training_df.dropna(subset=FEATURE_COLUMNS + [f'red_{i}_next'])
        X = df_temp[FEATURE_COLUMNS]
        y = df_temp[f'red_{i}_next']
        if not X.empty: 
            lgb_clf = lgb.LGBMClassifier(random_state=42, verbose=-1)
            lgb_clf.fit(X, y)
            final_ml_models_red[i] = lgb_clf
            
    # 训练16个独立的蓝球模型
    for i in tqdm(range(1, 17), desc="训练最终蓝球模型", ncols=80):
        ml_training_df[f'blue_{i}_next'] = ml_training_df['蓝球'].apply(lambda x: 1 if x == i else 0).shift(-1)
        df_temp = ml_training_df.dropna(subset=FEATURE_COLUMNS + [f'blue_{i}_next'])
        X = df_temp[FEATURE_COLUMNS]
        y = df_temp[f'blue_{i}_next']
        if not X.empty: 
            lgb_clf = lgb.LGBMClassifier(random_state=42, verbose=-1)
            lgb_clf.fit(X, y)
            final_ml_models_blue[i] = lgb_clf
    
    # --- [阶段 4/8] 执行对下一期的预测 ---
    print(f"\n[阶段 4/8] 正在为下一期号码进行机器学习评分...")
    red_scores, blue_scores = run_strategy_and_get_scores(full_df, params, final_ml_models_red, final_ml_models_blue, FEATURE_COLUMNS)
    red_pool = build_red_pool(red_scores, mode=args.pool_mode)
    recommended_blues = sorted(blue_scores, key=blue_scores.get, reverse=True)[:NUM_BLUE_BALLS]
    print(f"已根据ML评分选出 {POOL_SIZE_RED} 个红球大底: {sorted(red_pool)}")

    # --- [阶段 5/8] 规则过滤 ---
    print(f"\n[阶段 5/8] 正在从大底中生成组合并应用硬规则过滤...")
    
    # --- 生成反向排他库 (展开形式) ---
    rejection_set = make_rejection_set(REJECTION_LIB_SIZE, random.Random(RANDOM_SEED))

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
    print(f"\n[阶段 7/8] 正在从最终组合中，生成高重合度的7红球大底...")
    best_7_reds = find_best_7_red_combinations(passed_combos_tuples, red_pool)

    # --- [阶段 8/8] 最终报告 (完全采纳 ssq2 格式) ---
    print("\n--- 正在生成最终推荐报告 ---")
    report_lines = []
    report_lines.append("="*60); report_lines.append("          双色球策略分析与推荐报告 (高级过滤版)"); report_lines.append("="*60)
    
    latest_issue = str(full_df.iloc[-1]['期号'])
    try: 
        target_issue = int(latest_issue) + 1
    except ValueError: 
        target_issue = f"{latest_issue}_Next"

    report_lines.append("\n--- 0. 报告元数据 ---")
    report_lines.append(f"Data_Basis_Issue: {latest_issue}")
    report_lines.append(f"Prediction_Target_Issue: {target_issue}")
    report_lines.append(f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    report_lines.append("\n--- 1. 策略参数与回测 ---")
    mode_desc = "加载已固化的参数" if params_loaded else "使用内置的默认参数"
    report_lines.append(f"模式: {mode_desc}")
    for key, val in params.items(): 
        report_lines.append(f"  - {key:<20}: {val}")
    
    report_lines.append(f"\n历史滚动回测 ({backtest.periods}期):")
    report_lines.append(f"  - 候选池模式: {args.pool_mode}")
    report_lines.append(f"  - 实际投注期数: {backtest.active_periods}")
    report_lines.append(f"  - 投注注数: {backtest.tickets}")
    report_lines.append(f"  - 总投入: {backtest.cost:.2f} 元")
    report_lines.append(f"  - 总奖金: {backtest.winnings:.2f} 元")
    report_lines.append(f"  - 净收益: {backtest.profit:.2f} 元")
    report_lines.append(f"  - 回报率: {backtest.roi:.2%}")
    if len(backtests) > 1:
        report_lines.append("  - 候选池对照:")
        for name, result in backtests.items():
            report_lines.append(
                f"    {name:<6} 投入 {result.cost:>6.0f} 元，奖金 {result.winnings:>6.0f} 元，"
                f"净收益 {result.profit:>7.0f} 元，回报率 {result.roi:>7.2%}"
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

    report_lines.append("\n--- 2. 推荐组合 ---")
    top_blue = recommended_blues[0] if recommended_blues else "N/A"
    
    report_lines.append("\n【单式推荐 (10组)】")
    if passed_combos_tuples:
        final_selection = sorted(
            passed_combos_tuples,
            key=lambda combo: (
                -score_red_combination(combo, red_scores, last_draw_set, last_2_draw_set), combo
            )
        )[:NUM_RECOMMENDATIONS]
        for i, combo in enumerate(final_selection, 1):
            report_lines.append(f"  组合 {i:>2}: 红球 {str(list(combo)):<24} 蓝球 [{top_blue:02d}]")
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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ssq_analysis_output_{timestamp}.txt"
        filepath = os.path.join(REPORT_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f: 
            f.write(final_report_string)
        print(f"\n\n报告已成功保存到文件: {filepath}")
    except Exception as e:
        print(f"\n\n写入报告文件失败: {e}")

# -*- coding: utf-8 -*-
"""
双色球彩票数据分析与推荐系统
================================

本脚本整合了统计分析、机器学习和策略化组合生成，为双色球彩票提供数据驱动的
号码推荐。脚本支持两种运行模式，由全局变量 `ENABLE_OPTUNA_OPTIMIZATION` 控制：

1.  **分析模式 (默认 `False`)**:
    使用内置的 `DEFAULT_WEIGHTS` 权重，执行一次完整的历史数据分析、策略回测，
    并为下一期生成推荐号码。所有结果会输出到一个带时间戳的详细报告文件中。

2.  **优化模式 (`True`)**:
    在分析前，首先运行 Optuna 框架进行参数搜索，以找到在近期历史数据上
    表现最佳的一组权重。然后，自动使用这组优化后的权重来完成后续的分析、
    回测和推荐。优化过程和结果也会记录在报告中。

版本: 5.1 (Robust & Well-Commented)
"""

# --- 标准库导入 ---
import os
import sys
import json
import time
import datetime
import logging
import io
import random
from collections import Counter
from contextlib import redirect_stdout
from typing import (Union, Optional, List, Dict, Tuple, Any)
from functools import partial

# --- 第三方库导入 ---
import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import concurrent.futures

# ==============================================================================
# --- 全局常量与配置 ---
# ==============================================================================

# --------------------------
# --- 路径与模式配置 ---
# --------------------------
# 脚本文件所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 原始双色球数据CSV文件路径 (由 ssq_data_processor.py 生成)
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'shuangseqiu.csv')
# 预处理后的数据缓存文件路径，避免每次都重新计算特征
PROCESSED_CSV_PATH = os.path.join(SCRIPT_DIR, 'shuangseqiu_processed.csv')

# 运行模式配置:
# True  -> 运行参数优化，耗时较长，但可能找到更优策略。
# False -> 使用默认权重进行快速分析和推荐。
ENABLE_OPTUNA_OPTIMIZATION = True

# --------------------------
# --- 策略开关配置 ---
# --------------------------
# 是否启用最终推荐组合层面的“反向思维”策略 (移除得分最高的几注)
ENABLE_FINAL_COMBO_REVERSE = True
# 在启用反向思维并移除组合后，是否从候选池中补充新的组合以达到目标数量
ENABLE_REVERSE_REFILL = True

# --------------------------
# --- 彩票规则配置 ---
# --------------------------
# 红球的有效号码范围 (1到33)
RED_BALL_RANGE = range(1, 34)
# 蓝球的有效号码范围 (1到16)
BLUE_BALL_RANGE = range(1, 17)
# 红球三分区定义，用于特征工程和模式分析
RED_ZONES = {'Zone1': (1, 11), 'Zone2': (12, 22), 'Zone3': (23, 33)}

# --------------------------
# --- 分析与执行参数配置 ---
# --------------------------
# 机器学习模型使用的滞后特征阶数 (e.g., 使用前1、3、5、10期的数据作为特征)
ML_LAG_FEATURES = [1, 3, 5, 10]
# 用于生成乘积交互特征的特征对 (e.g., 红球和值 * 红球奇数个数)
ML_INTERACTION_PAIRS = [('red_sum', 'red_odd_count')]
# 用于生成自身平方交互特征的特征 (e.g., 红球跨度的平方)
ML_INTERACTION_SELF = ['red_span']
# 计算号码“近期”出现频率时所参考的期数窗口大小
RECENT_FREQ_WINDOW = 20
# 在分析模式下，进行策略回测时所评估的总期数
BACKTEST_PERIODS_COUNT = 100
# 在优化模式下，每次试验用于快速评估性能的回测期数 (数值越小优化越快)
OPTIMIZATION_BACKTEST_PERIODS = 20
# 在优化模式下，Optuna 进行参数搜索的总试验次数
OPTIMIZATION_TRIALS = 100
# 训练机器学习模型时，一个球号在历史数据中至少需要出现的次数 (防止样本过少导致模型不可靠)
MIN_POSITIVE_SAMPLES_FOR_ML = 25

# ==============================================================================
# --- 默认权重配置 (这些参数可被Optuna优化) ---
# ==============================================================================
# 这里的每一项都是一个可调整的策略参数，共同决定了最终的推荐结果。
DEFAULT_WEIGHTS = {
    # --- 反向思维 ---
    # 若启用反向思维，从最终推荐列表中移除得分最高的组合的比例
    'FINAL_COMBO_REVERSE_REMOVE_TOP_PERCENT': 0.3,

    # --- 组合生成 ---
    # 最终向用户推荐的组合（注数）数量
    'NUM_COMBINATIONS_TO_GENERATE': 10,
    # 构建红球候选池时，从所有红球中选取分数最高的N个
    'TOP_N_RED_FOR_CANDIDATE': 25,
    # 构建蓝球候选池时，从所有蓝球中选取分数最高的N个
    'TOP_N_BLUE_FOR_CANDIDATE': 6,

    # --- 红球评分权重 ---
    # 红球历史总频率得分的权重
    'FREQ_SCORE_WEIGHT': 28.19,
    # 红球当前遗漏值（与平均遗漏的偏差）得分的权重
    'OMISSION_SCORE_WEIGHT': 19.92,
    # 红球当前遗漏与其历史最大遗漏比率的得分权重
    'MAX_OMISSION_RATIO_SCORE_WEIGHT_RED': 16.12,
    # 红球近期出现频率的得分权重
    'RECENT_FREQ_SCORE_WEIGHT_RED': 15.71,
    # 红球的机器学习模型预测出现概率的得分权重
    'ML_PROB_SCORE_WEIGHT_RED': 22.43,

    # --- 蓝球评分权重 ---
    # 蓝球历史总频率得分的权重
    'BLUE_FREQ_SCORE_WEIGHT': 27.11,
    # 蓝球当前遗漏值（与平均遗漏的偏差）得分的权重
    'BLUE_OMISSION_SCORE_WEIGHT': 23.26,
    # 蓝球的机器学习模型预测出现概率的得分权重
    'ML_PROB_SCORE_WEIGHT_BLUE': 43.48,

    # --- 组合属性匹配奖励 ---
    # 推荐组合的红球奇数个数若与历史最常见模式匹配，获得的奖励分值
    'COMBINATION_ODD_COUNT_MATCH_BONUS': 13.10,
    # 推荐组合的蓝球奇偶性若与历史最常见模式匹配，获得的奖励分值
    'COMBINATION_BLUE_ODD_MATCH_BONUS': 0.40,
    # 推荐组合的红球区间分布若与历史最常见模式匹配，获得的奖励分值
    'COMBINATION_ZONE_MATCH_BONUS': 13.12,
    # 推荐组合的蓝球大小若与历史最常见模式匹配，获得的奖励分值
    'COMBINATION_BLUE_SIZE_MATCH_BONUS': 0.84,

    # --- 关联规则挖掘(ARM)参数与奖励 ---
    # ARM算法的最小支持度阈值
    'ARM_MIN_SUPPORT': 0.01,
    # ARM算法的最小置信度阈值
    'ARM_MIN_CONFIDENCE': 0.53,
    # ARM算法的最小提升度阈值
    'ARM_MIN_LIFT': 1.53,
    # 推荐组合若命中了某条挖掘出的关联规则，其获得的基础奖励分值
    'ARM_COMBINATION_BONUS_WEIGHT': 18.86,
    # 在计算ARM奖励时，规则的提升度(lift)对此奖励的贡献乘数因子
    'ARM_BONUS_LIFT_FACTOR': 0.48,
    # 在计算ARM奖励时，规则的置信度(confidence)对此奖励的贡献乘数因子
    'ARM_BONUS_CONF_FACTOR': 0.25,

    # --- 组合多样性控制 ---
    # 最终推荐的任意两注组合之间，其红球号码至少要有几个是不同的
    'DIVERSITY_MIN_DIFFERENT_REDS': 3,
}

# ==============================================================================
# --- 机器学习模型参数配置 ---
# ==============================================================================
# 这些是 LightGBM 机器学习模型的核心超参数。
LGBM_PARAMS = {
    'objective': 'binary',              # 目标函数：二分类问题（预测一个球号是否出现）
    'boosting_type': 'gbdt',            # 提升类型：梯度提升决策树
    'learning_rate': 0.04,              # 学习率：控制每次迭代的步长
    'n_estimators': 100,                # 树的数量：总迭代次数
    'num_leaves': 15,                   # 每棵树的最大叶子节点数：控制模型复杂度
    'min_child_samples': 15,            # 一个叶子节点上所需的最小样本数：防止过拟合
    'lambda_l1': 0.15,                  # L1 正则化
    'lambda_l2': 0.15,                  # L2 正则化
    'feature_fraction': 0.7,            # 特征采样比例：每次迭代随机选择70%的特征
    'bagging_fraction': 0.8,            # 数据采样比例：每次迭代随机选择80%的数据
    'bagging_freq': 5,                  # 数据采样的频率：每5次迭代进行一次
    'seed': 42,                         # 随机种子：确保结果可复现
    'n_jobs': 1,                        # 并行线程数：设为1以在多进程环境中避免冲突
    'verbose': -1,                      # 控制台输出级别：-1表示静默
}

# ==============================================================================
# --- 日志系统配置 ---
# ==============================================================================
# 创建两种格式化器
console_formatter = logging.Formatter('%(message)s')  # 用于控制台的简洁格式
detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s') # 用于文件的详细格式

# 主日志记录器
logger = logging.getLogger('ssq_analyzer')
logger.setLevel(logging.DEBUG)
logger.propagate = False # 防止日志向根记录器传递

# 进度日志记录器 (用于回测和Optuna进度条，避免被详细格式污染)
progress_logger = logging.getLogger('progress_logger')
progress_logger.setLevel(logging.INFO)
progress_logger.propagate = False

# 全局控制台处理器
global_console_handler = logging.StreamHandler(sys.stdout)
global_console_handler.setFormatter(console_formatter)

# 进度专用控制台处理器
progress_console_handler = logging.StreamHandler(sys.stdout)
progress_console_handler.setFormatter(logging.Formatter('%(message)s'))

logger.addHandler(global_console_handler)
progress_logger.addHandler(progress_console_handler)

def set_console_verbosity(level=logging.INFO, use_simple_formatter=False):
    """动态设置主日志记录器在控制台的输出级别和格式。"""
    global_console_handler.setLevel(level)
    global_console_handler.setFormatter(console_formatter if use_simple_formatter else detailed_formatter)

# ==============================================================================
# --- 核心工具函数 ---
# ==============================================================================

class SuppressOutput:
    """一个上下文管理器，用于临时抑制标准输出和捕获标准错误。"""
    def __init__(self, suppress_stdout=True, capture_stderr=True):
        self.suppress_stdout, self.capture_stderr = suppress_stdout, capture_stderr
        self.old_stdout, self.old_stderr, self.stdout_io, self.stderr_io = None, None, None, None
    def __enter__(self):
        if self.suppress_stdout: self.old_stdout, self.stdout_io, sys.stdout = sys.stdout, io.StringIO(), self.stdout_io
        if self.capture_stderr: self.old_stderr, self.stderr_io, sys.stderr = sys.stderr, io.StringIO(), self.stderr_io
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.capture_stderr and self.old_stderr:
            sys.stderr = self.old_stderr; captured = self.stderr_io.getvalue(); self.stderr_io.close()
            if captured.strip(): logger.warning(f"在一个被抑制的输出块中捕获到标准错误:\n{captured.strip()}")
        if self.suppress_stdout and self.old_stdout:
            sys.stdout = self.old_stdout; self.stdout_io.close()
        return False # 不抑制异常

def get_prize_level(red_hits: int, blue_hit: bool) -> Optional[str]:
    """根据红球和蓝球的命中个数，确定中奖等级。"""
    if blue_hit:
        if red_hits == 6: return "一等奖"
        if red_hits == 5: return "三等奖"
        if red_hits == 4: return "四等奖"
        if red_hits == 3: return "五等奖"
        if red_hits <= 2: return "六等奖"
    else:
        if red_hits == 6: return "二等奖"
        if red_hits == 5: return "四等奖"
        if red_hits == 4: return "五等奖"
    return None

def format_time(seconds: float) -> str:
    """将秒数格式化为易于阅读的 HH:MM:SS 字符串。"""
    if seconds < 0: return "00:00:00"
    hours, remainder = divmod(seconds, 3600)
    minutes, sec = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(sec):02d}"

# ==============================================================================
# --- 数据处理模块 ---
# ==============================================================================

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    从CSV文件加载数据，并能自动尝试多种常用编码格式。

    Args:
        file_path (str): CSV文件的路径。

    Returns:
        Optional[pd.DataFrame]: 加载成功的DataFrame，如果文件不存在或无法解码则返回None。
    """
    if not os.path.exists(file_path):
        logger.error(f"数据文件未找到: {file_path}")
        return None
    for enc in ['utf-8', 'gbk', 'latin-1']:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"使用编码 {enc} 加载 {file_path} 时出错: {e}")
            return None
    logger.error(f"无法使用任何支持的编码打开文件 {file_path}。"); return None

def clean_and_structure(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    清洗和结构化原始DataFrame，确保数据类型正确，并转换为“一行一期”的格式。

    Args:
        df (pd.DataFrame): 从CSV原始加载的DataFrame。

    Returns:
        Optional[pd.DataFrame]: 清洗和结构化后的DataFrame，如果输入无效或处理失败则返回None。
    """
    if df is None or df.empty: return None
    required_cols = ['期号', '红球', '蓝球']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"输入数据缺少必要列: {required_cols}")
        return None

    df.dropna(subset=required_cols, inplace=True)
    try:
        df['期号'] = pd.to_numeric(df['期号'], errors='coerce')
        df.dropna(subset=['期号'], inplace=True)
        df = df.astype({'期号': int})
    except (ValueError, TypeError) as e:
        logger.error(f"转换'期号'为整数时失败: {e}"); return None

    df.sort_values(by='期号', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    parsed_rows = []
    for _, row in df.iterrows():
        try:
            # 解析红球和蓝球，并进行严格验证
            reds = sorted([int(r) for r in str(row['红球']).split(',')])
            blue = int(row['蓝球'])
            if len(reds) != 6 or not all(r in RED_BALL_RANGE for r in reds) or blue not in BLUE_BALL_RANGE:
                logger.warning(f"期号 {row['期号']} 的数据无效，已跳过: 红球={reds}, 蓝球={blue}")
                continue
            
            # 构建结构化的记录
            record = {'期号': row['期号'], 'blue': blue, **{f'red{i+1}': r for i, r in enumerate(reds)}}
            if '日期' in row and pd.notna(row['日期']):
                record['日期'] = row['日期']
            parsed_rows.append(record)
        except (ValueError, TypeError):
            logger.warning(f"解析期号 {row['期号']} 的号码时失败，已跳过。")
            continue
            
    return pd.DataFrame(parsed_rows) if parsed_rows else None

def feature_engineer(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    为DataFrame计算各种衍生特征，如和值、跨度、奇偶比、区间分布等。

    Args:
        df (pd.DataFrame): 经过清洗和结构化后的DataFrame。

    Returns:
        Optional[pd.DataFrame]: 包含新计算特征的DataFrame。
    """
    if df is None or df.empty: return None
    df_fe = df.copy()
    red_cols = [f'red{i+1}' for i in range(6)]
    
    # 基本统计特征
    df_fe['red_sum'] = df_fe[red_cols].sum(axis=1)
    df_fe['red_span'] = df_fe[red_cols].max(axis=1) - df_fe[red_cols].min(axis=1)
    df_fe['red_odd_count'] = df_fe[red_cols].apply(lambda r: sum(x % 2 != 0 for x in r), axis=1)
    
    # 区间特征
    for zone, (start, end) in RED_ZONES.items():
        df_fe[f'red_{zone}_count'] = df_fe[red_cols].apply(lambda r: sum(start <= x <= end for x in r), axis=1)
        
    # 形态特征
    def count_consecutive(row): return sum(1 for i in range(5) if row.iloc[i+1] - row.iloc[i] == 1)
    df_fe['red_consecutive_count'] = df_fe[red_cols].apply(count_consecutive, axis=1)
    
    # 重号特征 (与上一期的重复个数)
    red_sets = df_fe[red_cols].apply(set, axis=1)
    prev_red_sets = red_sets.shift(1)
    df_fe['red_repeat_count'] = [len(current.intersection(prev)) if isinstance(prev, set) else 0 for current, prev in zip(red_sets, prev_red_sets)]
    
    # 蓝球特征
    df_fe['blue_is_odd'] = (df_fe['blue'] % 2 != 0).astype(int)
    df_fe['blue_is_large'] = (df_fe['blue'] > 8).astype(int)
    
    return df_fe

def create_lagged_features(df: pd.DataFrame, lags: List[int]) -> Optional[pd.DataFrame]:
    """
    为机器学习模型创建滞后特征（将历史期的特征作为当前期的输入）和交互特征。

    Args:
        df (pd.DataFrame): 包含基础特征的DataFrame。
        lags (List[int]): 滞后阶数列表, e.g., [1, 3, 5]。

    Returns:
        Optional[pd.DataFrame]: 一个只包含滞后和交互特征的DataFrame。
    """
    if df is None or df.empty or not lags: return None
    
    feature_cols = [col for col in df.columns if 'red_' in col or 'blue_' in col]
    df_features = df[feature_cols].copy()
    
    # 创建交互特征
    for c1, c2 in ML_INTERACTION_PAIRS:
        if c1 in df_features and c2 in df_features: df_features[f'{c1}_x_{c2}'] = df_features[c1] * df_features[c2]
    for c in ML_INTERACTION_SELF:
        if c in df_features: df_features[f'{c}_sq'] = df_features[c]**2
        
    # 创建滞后特征
    all_feature_cols = df_features.columns.tolist()
    lagged_dfs = [df_features[all_feature_cols].shift(lag).add_suffix(f'_lag{lag}') for lag in lags]
    final_df = pd.concat(lagged_dfs, axis=1)
    final_df.dropna(inplace=True)
    
    return final_df if not final_df.empty else None

# ==============================================================================
# --- 分析与评分模块 ---
# ==============================================================================

def analyze_frequency_omission(df: pd.DataFrame) -> dict:
    """
    分析所有号码的频率、当前遗漏、平均遗漏、最大遗漏和近期频率。

    Args:
        df (pd.DataFrame): 包含历史数据的DataFrame。

    Returns:
        dict: 包含各种频率和遗漏统计信息的字典。
    """
    if df is None or df.empty: return {}
    red_cols, total_periods = [f'red{i+1}' for i in range(6)], len(df)
    most_recent_idx = total_periods - 1
    
    # 频率计算
    all_reds_flat = df[red_cols].values.flatten()
    red_freq, blue_freq = Counter(all_reds_flat), Counter(df['blue'])
    
    # 遗漏和近期频率计算
    current_omission, max_hist_omission, recent_N_freq = {}, {}, Counter()
    
    for num in RED_BALL_RANGE:
        app_indices = df.index[(df[red_cols] == num).any(axis=1)].tolist()
        if app_indices:
            current_omission[num] = most_recent_idx - app_indices[-1]
            gaps = np.diff([0] + app_indices) - 1 # 包含从开始到第一次出现的遗漏
            max_hist_omission[num] = max(gaps.max(), current_omission[num])
        else:
            current_omission[num] = max_hist_omission[num] = total_periods
            
    # 计算近期频率
    if total_periods >= RECENT_FREQ_WINDOW:
        recent_N_freq.update(df.tail(RECENT_FREQ_WINDOW)[red_cols].values.flatten())
        
    for num in BLUE_BALL_RANGE:
        app_indices = df.index[df['blue'] == num].tolist()
        current_omission[f'blue_{num}'] = most_recent_idx - app_indices[-1] if app_indices else total_periods
        
    # 平均间隔（理论遗漏）
    avg_interval = {num: total_periods / (red_freq.get(num, 0) + 1e-9) for num in RED_BALL_RANGE}
    for num in BLUE_BALL_RANGE:
        avg_interval[f'blue_{num}'] = total_periods / (blue_freq.get(num, 0) + 1e-9)
        
    return {
        'red_freq': red_freq, 'blue_freq': blue_freq,
        'current_omission': current_omission, 'average_interval': avg_interval,
        'max_historical_omission_red': max_hist_omission,
        'recent_N_freq_red': recent_N_freq
    }

def analyze_patterns(df: pd.DataFrame) -> dict:
    """
    分析历史数据中的常见模式，如最常见的和值、奇偶比、区间分布等。

    Args:
        df (pd.DataFrame): 包含特征工程后历史数据的DataFrame。

    Returns:
        dict: 包含最常见模式的字典。
    """
    if df is None or df.empty: return {}
    res = {}
    def safe_mode(s): return s.mode().iloc[0] if not s.empty and not s.mode().empty else None
    
    for col, name in [('red_sum', 'sum'), ('red_span', 'span'), ('red_odd_count', 'odd_count')]:
        if col in df.columns: res[f'most_common_{name}'] = safe_mode(df[col])
        
    zone_cols = [f'red_{zone}_count' for zone in RED_ZONES.keys()]
    if all(c in df.columns for c in zone_cols):
        dist_counts = df[zone_cols].apply(tuple, axis=1).value_counts()
        if not dist_counts.empty: res['most_common_zone_distribution'] = dist_counts.index[0]
        
    if 'blue_is_odd' in df.columns: res['most_common_blue_is_odd'] = safe_mode(df['blue_is_odd'])
    if 'blue_is_large' in df.columns: res['most_common_blue_is_large'] = safe_mode(df['blue_is_large'])
    
    return res

def analyze_associations(df: pd.DataFrame, weights_config: Dict) -> pd.DataFrame:
    """
    使用Apriori算法挖掘红球号码之间的关联规则（例如，哪些号码倾向于一起出现）。

    Args:
        df (pd.DataFrame): 包含历史数据的DataFrame。
        weights_config (Dict): 包含ARM算法参数(min_support, min_confidence, min_lift)的字典。

    Returns:
        pd.DataFrame: 一个包含挖掘出的强关联规则的DataFrame。
    """
    min_s = weights_config.get('ARM_MIN_SUPPORT', 0.01)
    min_c = weights_config.get('ARM_MIN_CONFIDENCE', 0.5)
    min_l = weights_config.get('ARM_MIN_LIFT', 1.5)
    red_cols = [f'red{i+1}' for i in range(6)]
    if df is None or df.empty: return pd.DataFrame()
    
    try:
        transactions = df[red_cols].astype(str).values.tolist()
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_oh = pd.DataFrame(te_ary, columns=te.columns_)
        
        frequent_itemsets = apriori(df_oh, min_support=min_s, use_colnames=True)
        if frequent_itemsets.empty: return pd.DataFrame()
        
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_l)
        strong_rules = rules[rules['confidence'] >= min_c].sort_values(by='lift', ascending=False)
        return strong_rules
        
    except Exception as e:
        logger.error(f"关联规则分析失败: {e}"); return pd.DataFrame()

def calculate_scores(freq_data: Dict, probabilities: Dict, weights: Dict) -> Dict[str, Dict[int, float]]:
    """
    根据所有分析结果（频率、遗漏、ML预测），使用加权公式计算每个球的最终推荐分数。

    Args:
        freq_data (Dict): 来自 `analyze_frequency_omission` 的频率和遗漏分析结果。
        probabilities (Dict): 来自机器学习模型的预测概率。
        weights (Dict): 包含所有评分权重的配置字典。

    Returns:
        Dict[str, Dict[int, float]]: 包含红球和蓝球归一化后分数的字典。
    """
    r_scores, b_scores = {}, {}
    r_freq, b_freq = freq_data.get('red_freq', {}), freq_data.get('blue_freq', {})
    omission, avg_int = freq_data.get('current_omission', {}), freq_data.get('average_interval', {})
    max_hist_o, recent_freq = freq_data.get('max_historical_omission_red', {}), freq_data.get('recent_N_freq_red', {})
    r_pred, b_pred = probabilities.get('red', {}), probabilities.get('blue', {})
    
    # 红球评分
    for num in RED_BALL_RANGE:
        # 频率分：出现次数越多，得分越高
        freq_s = (r_freq.get(num, 0)) * weights['FREQ_SCORE_WEIGHT']
        # 遗漏分：当前遗漏接近平均遗漏时得分最高，过冷或过热都会降低分数
        omit_s = np.exp(-0.005 * (omission.get(num, 0) - avg_int.get(num, 0))**2) * weights['OMISSION_SCORE_WEIGHT']
        # 最大遗漏比率分：当前遗漏接近或超过历史最大遗漏时得分高（博冷）
        max_o_ratio = (omission.get(num, 0) / max_hist_o.get(num, 1)) if max_hist_o.get(num, 0) > 0 else 0
        max_o_s = max_o_ratio * weights['MAX_OMISSION_RATIO_SCORE_WEIGHT_RED']
        # 近期频率分：近期出现次数越多，得分越高（追热）
        recent_s = recent_freq.get(num, 0) * weights['RECENT_FREQ_SCORE_WEIGHT_RED']
        # ML预测分
        ml_s = r_pred.get(num, 0.0) * weights['ML_PROB_SCORE_WEIGHT_RED']
        r_scores[num] = sum([freq_s, omit_s, max_o_s, recent_s, ml_s])
        
    # 蓝球评分
    for num in BLUE_BALL_RANGE:
        freq_s = (b_freq.get(num, 0)) * weights['BLUE_FREQ_SCORE_WEIGHT']
        omit_s = np.exp(-0.01 * (omission.get(f'blue_{num}', 0) - avg_int.get(f'blue_{num}', 0))**2) * weights['BLUE_OMISSION_SCORE_WEIGHT']
        ml_s = b_pred.get(num, 0.0) * weights['ML_PROB_SCORE_WEIGHT_BLUE']
        b_scores[num] = sum([freq_s, omit_s, ml_s])

    # 归一化所有分数到0-100范围，便于比较
    def normalize_scores(scores_dict):
        if not scores_dict: return {}
        vals = list(scores_dict.values())
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v: return {k: 50.0 for k in scores_dict}
        return {k: (v - min_v) / (max_v - min_v) * 100 for k, v in scores_dict.items()}

    return {'red_scores': normalize_scores(r_scores), 'blue_scores': normalize_scores(b_scores)}

# ==============================================================================
# --- 机器学习模块 ---
# ==============================================================================

def train_single_lgbm_model(ball_type_str: str, ball_number: int, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Optional[LGBMClassifier], Optional[str]]:
    """为单个球号训练一个LGBM二分类模型（预测它是否出现）。"""
    if y_train.sum() < MIN_POSITIVE_SAMPLES_FOR_ML or y_train.nunique() < 2:
        return None, None # 样本不足或只有一类，无法训练
        
    model_key = f'lgbm_{ball_number}'
    model_params = LGBM_PARAMS.copy()
    
    # 类别不平衡处理：给样本量较少的类别（中奖）更高的权重
    if (pos_count := y_train.sum()) > 0:
        model_params['scale_pos_weight'] = (len(y_train) - pos_count) / pos_count
        
    try:
        model = LGBMClassifier(**model_params)
        model.fit(X_train, y_train)
        return model, model_key
    except Exception as e:
        logger.debug(f"训练LGBM for {ball_type_str} {ball_number} 失败: {e}")
        return None, None

def train_prediction_models(df_train_raw: pd.DataFrame, ml_lags_list: List[int]) -> Optional[Dict[str, Any]]:
    """为所有红球和蓝球并行训练预测模型。"""
    if (X := create_lagged_features(df_train_raw.copy(), ml_lags_list)) is None or X.empty:
        logger.warning("创建滞后特征失败或结果为空，跳过模型训练。")
        return None
        
    if (target_df := df_train_raw.loc[X.index].copy()).empty: return None
    
    red_cols = [f'red{i+1}' for i in range(6)]
    trained_models = {'red': {}, 'blue': {}, 'feature_cols': X.columns.tolist()}
    
    # 使用进程池并行训练，加快速度
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {}
        # 为每个红球提交训练任务
        for ball_num in RED_BALL_RANGE:
            y = target_df[red_cols].eq(ball_num).any(axis=1).astype(int)
            future = executor.submit(train_single_lgbm_model, '红球', ball_num, X, y)
            futures[future] = ('red', ball_num)
        # 为每个蓝球提交训练任务
        for ball_num in BLUE_BALL_RANGE:
            y = target_df['blue'].eq(ball_num).astype(int)
            future = executor.submit(train_single_lgbm_model, '蓝球', ball_num, X, y)
            futures[future] = ('blue', ball_num)
            
        for future in concurrent.futures.as_completed(futures):
            ball_type, ball_num = futures[future]
            try:
                model, model_key = future.result()
                if model and model_key:
                    trained_models[ball_type][model_key] = model
            except Exception as e:
                logger.error(f"训练球号 {ball_num} ({ball_type}) 的模型时出现异常: {e}")

    return trained_models if trained_models['red'] or trained_models['blue'] else None

def predict_next_draw_probabilities(df_historical: pd.DataFrame, trained_models: Optional[Dict], ml_lags_list: List[int]) -> Dict[str, Dict[int, float]]:
    """使用训练好的模型预测下一期每个号码的出现概率。"""
    probs = {'red': {}, 'blue': {}}
    if not trained_models or not (feat_cols := trained_models.get('feature_cols')):
        return probs
        
    max_lag = max(ml_lags_list) if ml_lags_list else 0
    if len(df_historical) < max_lag + 1:
        return probs # 数据不足以创建预测所需的特征
        
    if (predict_X := create_lagged_features(df_historical.tail(max_lag + 1), ml_lags_list)) is None:
        return probs
        
    predict_X = predict_X.reindex(columns=feat_cols, fill_value=0)
    
    for ball_type, ball_range in [('red', RED_BALL_RANGE), ('blue', BLUE_BALL_RANGE)]:
        for ball_num in ball_range:
            if (model := trained_models.get(ball_type, {}).get(f'lgbm_{ball_num}')):
                try:
                    # 预测类别为1（出现）的概率
                    probs[ball_type][ball_num] = model.predict_proba(predict_X)[0, 1]
                except Exception:
                    pass
    return probs

# ==============================================================================
# --- 组合生成与策略应用模块 ---
# ==============================================================================

def generate_combinations(scores_data: Dict, pattern_data: Dict, arm_rules: pd.DataFrame, weights_config: Dict) -> Tuple[List[Dict], List[str]]:
    """根据评分和策略生成最终的推荐组合。"""
    num_to_gen = weights_config['NUM_COMBINATIONS_TO_GENERATE']
    r_scores, b_scores = scores_data.get('red_scores', {}), scores_data.get('blue_scores', {})
    if not r_scores or not b_scores: return [], ["无法生成推荐 (分数数据缺失)。"]

    # 1. 构建候选池
    top_n_red = int(weights_config['TOP_N_RED_FOR_CANDIDATE'])
    top_n_blue = int(weights_config['TOP_N_BLUE_FOR_CANDIDATE'])
    r_cand_pool = [n for n, _ in sorted(r_scores.items(), key=lambda i: i[1], reverse=True)[:top_n_red]]
    b_cand_pool = [n for n, _ in sorted(b_scores.items(), key=lambda i: i[1], reverse=True)[:top_n_blue]]
    if len(r_cand_pool) < 6 or not b_cand_pool: return [], ["候选池号码不足。"]

    # 2. 生成大量初始组合
    large_pool_size = max(num_to_gen * 50, 500)
    gen_pool, unique_combos = [], set()
    r_weights = np.array([r_scores.get(n, 0) + 1 for n in r_cand_pool])
    r_probs = r_weights / r_weights.sum() if r_weights.sum() > 0 else None
    
    for _ in range(large_pool_size * 20): # 尝试多次以生成足量不重复组合
        if len(gen_pool) >= large_pool_size: break
        reds = sorted(np.random.choice(r_cand_pool, size=6, replace=False, p=r_probs).tolist()) if r_probs is not None else sorted(random.sample(r_cand_pool, 6))
        blue = random.choice(b_cand_pool)
        if (combo_tuple := (tuple(reds), blue)) not in unique_combos:
            gen_pool.append({'red': reds, 'blue': blue}); unique_combos.add(combo_tuple)

    # 3. 评分和筛选
    scored_combos = []
    for c in gen_pool:
        # 基础分 = 号码分总和
        base_score = sum(r_scores.get(r, 0) for r in c['red']) + b_scores.get(c['blue'], 0)
        scored_combos.append({'combination': c, 'score': base_score, 'red_tuple': tuple(c['red'])})

    # 4. 多样性筛选和最终选择
    sorted_combos = sorted(scored_combos, key=lambda x: x['score'], reverse=True)
    final_recs = []
    max_common = 6 - int(weights_config.get('DIVERSITY_MIN_DIFFERENT_REDS', 3))
    
    if sorted_combos:
        final_recs.append(sorted_combos.pop(0))
        for cand in sorted_combos:
            if len(final_recs) >= num_to_gen: break
            # 检查与已选组合的多样性
            if all(len(set(cand['red_tuple']) & set(rec['red_tuple'])) <= max_common for rec in final_recs):
                final_recs.append(cand)
    
    # 5. 应用反向思维策略
    applied_msg = ""
    if ENABLE_FINAL_COMBO_REVERSE:
        num_to_remove = int(len(final_recs) * weights_config.get('FINAL_COMBO_REVERSE_REMOVE_TOP_PERCENT', 0))
        if 0 < num_to_remove < len(final_recs):
            removed, final_recs = final_recs[:num_to_remove], final_recs[num_to_remove:]
            applied_msg = f" (反向策略: 移除前{num_to_remove}注"
            if ENABLE_REVERSE_REFILL:
                # 补充被移除的组合
                refill_candidates = [c for c in sorted_combos if c not in final_recs and c not in removed]
                final_recs.extend(refill_candidates[:num_to_remove])
                applied_msg += "并补充)"
            else:
                applied_msg += ")"

    final_recs = sorted(final_recs, key=lambda x: x['score'], reverse=True)[:num_to_gen]

    # 6. 生成输出字符串
    output_strs = [f"推荐组合 (Top {len(final_recs)}{applied_msg}):"]
    for i, c in enumerate(final_recs):
        r_str = ' '.join(f'{n:02d}' for n in c['combination']['red'])
        b_str = f"{c['combination']['blue']:02d}"
        output_strs.append(f"  注 {i+1}: 红球 [{r_str}] 蓝球 [{b_str}] (综合分: {c['score']:.2f})")
        
    return final_recs, output_strs

# ==============================================================================
# --- 核心分析与回测流程 ---
# ==============================================================================

def run_analysis_and_recommendation(df_hist: pd.DataFrame, ml_lags: List[int], weights_config: Dict, arm_rules: pd.DataFrame) -> Tuple:
    """
    执行一次完整的分析和推荐流程，用于特定一期。

    Returns:
        tuple: 包含推荐组合、输出字符串、分析摘要、训练模型和分数的元组。
    """
    freq_data = analyze_frequency_omission(df_hist)
    patt_data = analyze_patterns(df_hist)
    ml_models = train_prediction_models(df_hist, ml_lags)
    probabilities = predict_next_draw_probabilities(df_hist, ml_models, ml_lags) if ml_models else {'red': {}, 'blue': {}}
    scores = calculate_scores(freq_data, probabilities, weights_config)
    recs, rec_strings = generate_combinations(scores, patt_data, arm_rules, weights_config)
    analysis_summary = {'frequency_omission': freq_data, 'patterns': patt_data}
    return recs, rec_strings, analysis_summary, ml_models, scores

def run_backtest(full_df: pd.DataFrame, ml_lags: List[int], weights_config: Dict, arm_rules: pd.DataFrame, num_periods: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    在历史数据上执行策略回测，以评估策略表现。

    Returns:
        tuple: 包含详细回测结果的DataFrame和统计摘要的字典。
    """
    min_data_needed = (max(ml_lags) if ml_lags else 0) + MIN_POSITIVE_SAMPLES_FOR_ML + num_periods
    if len(full_df) < min_data_needed:
        logger.error(f"数据不足以回测{num_periods}期。需要至少{min_data_needed}期，当前有{len(full_df)}期。")
        return pd.DataFrame(), {}

    start_idx = len(full_df) - num_periods
    results, prize_counts, red_cols = [], Counter(), [f'red{i+1}' for i in range(6)]
    best_hits_per_period = []
    
    logger.info("策略回测已启动...")
    start_time = time.time()
    
    for i in range(num_periods):
        current_iter = i + 1
        current_idx = start_idx + i
        
        # 使用SuppressOutput避免在回测循环中打印大量日志
        with SuppressOutput(suppress_stdout=True, capture_stderr=True):
            hist_data = full_df.iloc[:current_idx]
            predicted_combos, _, _, _, _ = run_analysis_and_recommendation(hist_data, ml_lags, weights_config, arm_rules)
            
        actual_outcome = full_df.loc[current_idx]
        actual_red_set, actual_blue = set(actual_outcome[red_cols]), actual_outcome['blue']
        
        period_max_red_hits, period_blue_hit_on_max_red = 0, False
        if not predicted_combos:
            best_hits_per_period.append({'period': actual_outcome['期号'], 'best_red_hits': 0, 'blue_hit': False, 'prize': None})
        else:
            for combo_dict in predicted_combos:
                combo = combo_dict['combination']
                red_hits = len(set(combo['red']) & actual_red_set)
                blue_hit = combo['blue'] == actual_blue
                prize = get_prize_level(red_hits, blue_hit)
                if prize: prize_counts[prize] += 1
                results.append({'period': actual_outcome['期号'], 'red_hits': red_hits, 'blue_hit': blue_hit, 'prize': prize})
                if red_hits > period_max_red_hits: period_max_red_hits, period_blue_hit_on_max_red = red_hits, blue_hit
                elif red_hits == period_max_red_hits and not period_blue_hit_on_max_red and blue_hit: period_blue_hit_on_max_red = True
            
            best_hits_per_period.append({'period': actual_outcome['期号'], 'best_red_hits': period_max_red_hits, 'blue_hit': period_blue_hit_on_max_red, 'prize': get_prize_level(period_max_red_hits, period_blue_hit_on_max_red)})

        # 打印进度
        if current_iter == 1 or current_iter % 10 == 0 or current_iter == num_periods:
            elapsed = time.time() - start_time
            avg_time = elapsed / current_iter
            remaining_time = avg_time * (num_periods - current_iter)
            progress_logger.info(f"回测进度: {current_iter}/{num_periods} | 平均耗时: {avg_time:.2f}s/期 | 预估剩余: {format_time(remaining_time)}")
            
    return pd.DataFrame(results), {'prize_counts': dict(prize_counts), 'best_hits_per_period': pd.DataFrame(best_hits_per_period)}

# ==============================================================================
# --- Optuna 参数优化模块 ---
# ==============================================================================

def objective(trial: optuna.trial.Trial, df_for_opt: pd.DataFrame, ml_lags: List[int], arm_rules: pd.DataFrame) -> float:
    """Optuna 的目标函数，用于评估一组给定的权重参数的好坏。"""
    trial_weights = {}
    
    # 动态地从DEFAULT_WEIGHTS构建搜索空间
    for key, value in DEFAULT_WEIGHTS.items():
        if isinstance(value, int):
            if 'NUM_COMBINATIONS' in key: trial_weights[key] = trial.suggest_int(key, 5, 15)
            elif 'TOP_N' in key: trial_weights[key] = trial.suggest_int(key, 18, 28)
            else: trial_weights[key] = trial.suggest_int(key, max(0, value - 2), value + 2)
        elif isinstance(value, float):
            # 对不同类型的浮点数使用不同的搜索范围
            if any(k in key for k in ['PERCENT', 'FACTOR', 'SUPPORT', 'CONFIDENCE']):
                trial_weights[key] = trial.suggest_float(key, value * 0.5, value * 1.5)
            else: # 对权重参数使用更宽的搜索范围
                trial_weights[key] = trial.suggest_float(key, value * 0.5, value * 2.0)

    full_trial_weights = DEFAULT_WEIGHTS.copy()
    full_trial_weights.update(trial_weights)
    
    # 在快速回测中评估这组权重
    with SuppressOutput():
        _, backtest_stats = run_backtest(df_for_opt, ml_lags, full_trial_weights, arm_rules, OPTIMIZATION_BACKTEST_PERIODS)
        
    # 定义一个分数来衡量表现，高奖金等级的权重更高
    prize_weights = {'一等奖': 1000, '二等奖': 200, '三等奖': 50, '四等奖': 10, '五等奖': 2, '六等奖': 1}
    score = sum(prize_weights.get(p, 0) * c for p, c in backtest_stats.get('prize_counts', {}).items())
    return score

def optuna_progress_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial, total_trials: int):
    """Optuna 的回调函数，用于在控制台报告优化进度。"""
    global OPTUNA_START_TIME
    current_iter = trial.number + 1
    if current_iter == 1 or current_iter % 10 == 0 or current_iter == total_trials:
        elapsed = time.time() - OPTUNA_START_TIME
        avg_time = elapsed / current_iter
        remaining_time = avg_time * (total_trials - current_iter)
        best_value = f"{study.best_value:.2f}" if study.best_trial else "N/A"
        progress_logger.info(f"Optuna进度: {current_iter}/{total_trials} | 当前最佳得分: {best_value} | 预估剩余: {format_time(remaining_time)}")

# ==============================================================================
# --- 主程序入口 ---
# ==============================================================================
if __name__ == "__main__":
    # 1. 初始化日志记录器，同时输出到控制台和文件
    log_filename = os.path.join(SCRIPT_DIR, f"ssq_analysis_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    file_handler = logging.FileHandler(log_filename, 'w', 'utf-8')
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    set_console_verbosity(logging.INFO, use_simple_formatter=True)

    logger.info("--- 双色球数据分析与推荐系统 ---")
    logger.info("启动数据加载和预处理...")

    # 2. 健壮的数据加载逻辑
    main_df = None
    if os.path.exists(PROCESSED_CSV_PATH):
        main_df = load_data(PROCESSED_CSV_PATH)
        if main_df is not None:
             logger.info("从缓存文件加载预处理数据成功。")

    if main_df is None or main_df.empty:
        logger.info("未找到或无法加载缓存数据，正在从原始文件生成...")
        raw_df = load_data(CSV_FILE_PATH)
        if raw_df is not None and not raw_df.empty:
            logger.info("原始数据加载成功，开始清洗...")
            cleaned_df = clean_and_structure(raw_df)
            if cleaned_df is not None and not cleaned_df.empty:
                logger.info("数据清洗成功，开始特征工程...")
                main_df = feature_engineer(cleaned_df)
                if main_df is not None and not main_df.empty:
                    logger.info("特征工程成功，保存预处理数据...")
                    try:
                        main_df.to_csv(PROCESSED_CSV_PATH, index=False)
                        logger.info(f"预处理数据已保存到: {PROCESSED_CSV_PATH}")
                    except IOError as e:
                        logger.error(f"保存预处理数据失败: {e}")
                else:
                    logger.error("特征工程失败，无法生成最终数据集。")
            else:
                logger.error("数据清洗失败。")
        else:
            logger.error("原始数据加载失败。")
    
    if main_df is None or main_df.empty:
        logger.critical("数据准备失败，无法继续。请检查 'ssq_data_processor.py' 是否已成功运行并生成 'shuangseqiu.csv'。程序终止。")
        sys.exit(1)
    
    logger.info(f"数据加载完成，共 {len(main_df)} 期有效数据。")
    last_period = main_df['期号'].iloc[-1]

    # 3. 根据模式执行：优化或直接分析
    active_weights = DEFAULT_WEIGHTS.copy()
    optuna_summary = None

    if ENABLE_OPTUNA_OPTIMIZATION:
        logger.info("\n" + "="*25 + " Optuna 参数优化模式 " + "="*25)
        set_console_verbosity(logging.INFO, use_simple_formatter=False)
        
        # 优化前先进行一次全局关联规则分析
        optuna_arm_rules = analyze_associations(main_df, DEFAULT_WEIGHTS)
        
        study = optuna.create_study(direction="maximize")
        global OPTUNA_START_TIME; OPTUNA_START_TIME = time.time()
        progress_callback_with_total = partial(optuna_progress_callback, total_trials=OPTIMIZATION_TRIALS)
        
        try:
            study.optimize(lambda t: objective(t, main_df, ML_LAG_FEATURES, optuna_arm_rules), n_trials=OPTIMIZATION_TRIALS, callbacks=[progress_callback_with_total])
            logger.info("Optuna 优化完成。")
            active_weights.update(study.best_params)
            optuna_summary = {"status": "完成", "best_value": study.best_value, "best_params": study.best_params}
        except Exception as e:
            logger.error(f"Optuna 优化过程中断: {e}", exc_info=True)
            optuna_summary = {"status": "中断", "error": str(e)}
            logger.warning("优化中断，将使用默认权重继续分析。")
    
    # 4. 切换到报告模式并打印报告头
    report_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(report_formatter)
    global_console_handler.setFormatter(report_formatter)
    
    logger.info("\n\n" + "="*60 + f"\n{' ' * 18}双色球策略分析报告\n" + "="*60)
    logger.info(f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"分析基于数据: 截至 {last_period} 期 (共 {len(main_df)} 期)")
    logger.info(f"本次预测目标: 第 {last_period + 1} 期")
    logger.info(f"日志文件: {os.path.basename(log_filename)}")

    # 5. 打印优化摘要
    if ENABLE_OPTUNA_OPTIMIZATION and optuna_summary:
        logger.info("\n" + "="*25 + " Optuna 优化摘要 " + "="*25)
        logger.info(f"优化状态: {optuna_summary['status']}")
        if optuna_summary['status'] == '完成':
            logger.info(f"最佳性能得分: {optuna_summary['best_value']:.4f}")
            logger.info("--- 本次分析已采用以下优化参数 ---")
            best_params_str = json.dumps(optuna_summary['best_params'], indent=2, ensure_ascii=False)
            logger.info(best_params_str)
        else: logger.info(f"错误信息: {optuna_summary['error']}")
    else:
        logger.info("\n--- 本次分析使用脚本内置的默认权重 ---")

    # 6. 全局分析
    full_history_arm_rules = analyze_associations(main_df, active_weights)
    
    # 7. 回测并打印报告
    logger.info("\n" + "="*25 + " 策 略 回 测 摘 要 " + "="*25)
    backtest_results_df, backtest_stats = run_backtest(main_df, ML_LAG_FEATURES, active_weights, full_history_arm_rules, BACKTEST_PERIODS_COUNT)
    
    if not backtest_results_df.empty:
        num_periods_tested = len(backtest_results_df['period'].unique())
        num_combos_per_period = active_weights.get('NUM_COMBINATIONS_TO_GENERATE', 10)
        total_bets = len(backtest_results_df)
        logger.info(f"回测周期: 最近 {num_periods_tested} 期 | 每期注数: {num_combos_per_period} | 总投入注数: {total_bets}")
        logger.info("\n--- 1. 奖金与回报分析 ---")
        prize_dist, prize_values = backtest_stats.get('prize_counts', {}), {'一等奖': 5e6, '二等奖': 1.5e5, '三等奖': 3e3, '四等奖': 200, '五等奖': 10, '六等奖': 5}
        total_revenue = sum(prize_values.get(p, 0) * c for p, c in prize_dist.items())
        total_cost = total_bets * 2
        roi = (total_revenue - total_cost) * 100 / total_cost if total_cost > 0 else 0
        logger.info(f"  - 估算总回报: {total_revenue:,.2f} 元 (总成本: {total_cost:,.2f} 元)")
        logger.info(f"  - 投资回报率 (ROI): {roi:.2f}%")
        logger.info("  - 中奖等级分布 (总计):")
        if prize_dist:
            for prize in prize_values.keys():
                if prize in prize_dist: logger.info(f"    - {prize:<4s}: {prize_dist[prize]:>4d} 次")
        else: logger.info("    - 未命中任何奖级。")
        logger.info("\n--- 2. 核心性能指标 ---")
        logger.info(f"  - 平均红球命中 (每注): {backtest_results_df['red_hits'].mean():.3f} / 6")
        logger.info(f"  - 蓝球命中率 (每注): {backtest_results_df['blue_hit'].mean() * 100:.2f}%")
        logger.info("\n--- 3. 每期最佳命中表现 ---")
        if (best_hits_df := backtest_stats.get('best_hits_per_period')) is not None and not best_hits_df.empty:
            logger.info("  - 在一期内至少命中:")
            for prize_name, prize_query in [("四等奖(4+1或5+0)", "(`best_red_hits` == 4 and `blue_hit`) or (`best_red_hits` == 5 and not `blue_hit`)"), ("三等奖(5+1)", "`best_red_hits` == 5 and `blue_hit`"), ("二等奖/一等奖", "`best_red_hits` == 6")]:
                count = best_hits_df.query(prize_query).shape[0] if not best_hits_df.empty else 0
                logger.info(f"    - {prize_name:<18s}: {count} / {num_periods_tested} 期")
            any_blue_hit_periods = best_hits_df['blue_hit'].sum()
            logger.info(f"  - 蓝球覆盖率: 在 {any_blue_hit_periods / num_periods_tested:.2%} 的期数中，推荐组合至少有一注命中蓝球")
    else: logger.warning("回测未产生有效结果，可能是数据量不足。")
    
    # 8. 最终推荐
    logger.info("\n" + "="*25 + f" 第 {last_period + 1} 期 号 码 推 荐 " + "="*25)
    final_recs, final_rec_strings, _, _, final_scores = run_analysis_and_recommendation(main_df, ML_LAG_FEATURES, active_weights, full_history_arm_rules)
    
    logger.info("\n--- 单式推荐 ---")
    for line in final_rec_strings: logger.info(line)
    
    logger.info("\n--- 复式参考 ---")
    if final_scores and final_scores.get('red_scores'):
        top_7_red = sorted([n for n, _ in sorted(final_scores['red_scores'].items(), key=lambda x: x[1], reverse=True)[:7]])
        top_7_blue = sorted([n for n, _ in sorted(final_scores['blue_scores'].items(), key=lambda x: x[1], reverse=True)[:7]])
        logger.info(f"  红球 (Top 7): {' '.join(f'{n:02d}' for n in top_7_red)}")
        logger.info(f"  蓝球 (Top 7): {' '.join(f'{n:02d}' for n in top_7_blue)}")
    
    logger.info("\n" + "="*60 + f"\n--- 报告结束 (详情请查阅: {os.path.basename(log_filename)}) ---\n")

import pandas as pd
import numpy as np
from collections import Counter # Counter 用于计数
import itertools
import random
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# from sklearn.feature_selection import RFE # 用于潜在的特征选择
# from sklearn.calibration import CalibratedClassifierCV # 用于概率校准
import concurrent.futures
import time
import os
import json
from lightgbm import LGBMClassifier
import xgboost as xgb
import optuna
from typing import Union, Optional, List, Dict, Tuple, Any
import sys
import datetime
import io
import logging
from contextlib import redirect_stdout, redirect_stderr

# --- 配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # 脚本所在目录
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'shuangseqiu.csv') # 原始数据CSV文件路径
PROCESSED_CSV_PATH = os.path.join(SCRIPT_DIR, 'shuangseqiu_processed.csv') # 处理后数据CSV文件路径
WEIGHTS_CONFIG_FILE = os.path.join(SCRIPT_DIR, 'weights_config.json') # 权重配置文件路径

RED_BALL_RANGE = range(1, 34) # 红球号码范围
BLUE_BALL_RANGE = range(1, 17) # 蓝球号码范围
RED_ZONES = {'Zone1': (1, 11), 'Zone2': (12, 22), 'Zone3': (23, 33)} # 红球分区定义

ML_LAG_FEATURES = [1, 3, 5, 10] # 机器学习使用的滞后特征阶数
# 新增: 需要创建的交互特征
ML_INTERACTION_PAIRS = [('red_sum', 'red_odd_count')] # 示例: ('特征1', '特征2') 用于两者乘积
ML_INTERACTION_SELF = ['red_span'] # 示例: '特征1' 用于其平方

BACKTEST_PERIODS_COUNT = 100 # 完整回测的期数
OPTIMIZATION_BACKTEST_PERIODS = 30 # Optuna优化时用于回测的期数 (减少以加快Optuna示例运行，可调整)
OPTIMIZATION_TRIALS = 100 # Optuna优化试验次数 (减少以加快Optuna示例运行，可调整)
RECENT_FREQ_WINDOW = 20 # 计算近期频率的窗口大小

CANDIDATE_POOL_SCORE_THRESHOLDS = {'High': 70, 'Medium': 40} # 候选池分数阈值
CANDIDATE_POOL_SEGMENT_NAMES = ['High', 'Medium', 'Low'] # 候选池分段名称

DEFAULT_WEIGHTS = { # 默认权重配置
    'NUM_COMBINATIONS_TO_GENERATE': 10,
    'TOP_N_RED_FOR_CANDIDATE': 18,
    'TOP_N_BLUE_FOR_CANDIDATE': 8,
    'FREQ_SCORE_WEIGHT': 18.0,
    'OMISSION_SCORE_WEIGHT': 14.0,
    'MAX_OMISSION_RATIO_SCORE_WEIGHT_RED': 10.0,
    'RECENT_FREQ_SCORE_WEIGHT_RED': 10.0,
    'BLUE_FREQ_SCORE_WEIGHT': 25.0,
    'BLUE_OMISSION_SCORE_WEIGHT': 15.0,
    'ML_PROB_SCORE_WEIGHT_RED': 23.0,
    'ML_PROB_SCORE_WEIGHT_BLUE': 25.0,
    'COMBINATION_ODD_COUNT_MATCH_BONUS': 14.0,
    'COMBINATION_BLUE_ODD_MATCH_BONUS': 7.0,
    'COMBINATION_ZONE_MATCH_BONUS': 11.0,
    'COMBINATION_BLUE_SIZE_MATCH_BONUS': 6.0,
    'ARM_MIN_SUPPORT': 0.008,
    'ARM_MIN_CONFIDENCE': 0.35,
    'ARM_MIN_LIFT': 1.1,
    'ARM_COMBINATION_BONUS_WEIGHT': 10.0,
    'ARM_BONUS_LIFT_FACTOR': 0.2,
    'ARM_BONUS_CONF_FACTOR': 0.1,
    'CANDIDATE_POOL_PROPORTIONS_HIGH': 0.5,
    'CANDIDATE_POOL_PROPORTIONS_MEDIUM': 0.3,
    'CANDIDATE_POOL_MIN_PER_SEGMENT': 2,
    'DIVERSITY_MIN_DIFFERENT_REDS': 3,
    'DIVERSITY_SELECTION_MAX_ATTEMPTS': 20,
    # 新增: 用于其他属性的多样性权重
    'DIVERSITY_SUM_DIFF_THRESHOLD': 15, # 不同组合间红球和值的最小差异
    'DIVERSITY_ODDEVEN_DIFF_MIN_COUNT': 1, # 奇偶数个数的最小差异
    'DIVERSITY_ZONE_DIST_MIN_DIFF_ZONES': 2, # 区间分布中计数必须不同的最小区间数量
    # Optuna 目标函数权重
    'OPTUNA_PRIZE_6_WEIGHT': 0.1,
    'OPTUNA_PRIZE_5_WEIGHT': 0.5,
    'OPTUNA_PRIZE_4_WEIGHT': 1.0,
    'OPTUNA_PRIZE_3_WEIGHT': 2.0,
    'OPTUNA_PRIZE_2_WEIGHT': 5.0,
    'OPTUNA_PRIZE_1_WEIGHT': 10.0,
    'OPTUNA_BLUE_HIT_RATE_WEIGHT': 10.0, # 原名 BLUE_PERF_WEIGHT
    'OPTUNA_RED_HITS_WEIGHT': 1.0, # 原名 RED_PERF_WEIGHT
}
CURRENT_WEIGHTS = DEFAULT_WEIGHTS.copy()

SCORE_SEGMENT_BOUNDARIES = [0, 25, 50, 75, 100] # 分数段边界
SCORE_SEGMENT_LABELS = [f'{SCORE_SEGMENT_BOUNDARIES[i]+1}-{SCORE_SEGMENT_BOUNDARIES[i+1]}'
                        for i in range(len(SCORE_SEGMENT_BOUNDARIES)-1)]
SCORE_SEGMENT_LABELS[0] = f'{SCORE_SEGMENT_BOUNDARIES[0]}-{SCORE_SEGMENT_BOUNDARIES[1]}'
if len(SCORE_SEGMENT_LABELS) != len(SCORE_SEGMENT_BOUNDARIES) - 1:
     raise ValueError("分数段标签数量与边界数量不匹配，请检查配置。")

# 机器学习模型参数 (如果需要，可以通过单独的Optuna研究进行调整)
LGBM_PARAMS = {'objective': 'binary', 'metric': 'binary_logloss', 'n_estimators': 100, 'learning_rate': 0.04, 'feature_fraction': 0.7, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'lambda_l1': 0.15, 'lambda_l2': 0.15, 'num_leaves': 15, 'min_child_samples': 15, 'verbose': -1, 'n_jobs': 1, 'seed': 42, 'boosting_type': 'gbdt'}
LOGISTIC_REG_PARAMS = {'penalty': 'l2', 'C': 0.1, 'solver': 'lbfgs', 'random_state': 42, 'max_iter': 5000, 'tol': 1e-3}
SVC_PARAMS = {'C': 0.1, 'kernel': 'rbf', 'gamma': 'scale', 'probability': True, 'random_state': 42, 'cache_size': 200, 'max_iter': 25000, 'tol': 1e-3}
XGB_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'n_estimators': 100, 'learning_rate': 0.04, 'max_depth': 3, 'subsample': 0.7, 'colsample_bytree': 0.7, 'gamma': 0.1, 'lambda': 0.15, 'alpha': 0.15, 'seed': 42, 'n_jobs': 1}
MIN_POSITIVE_SAMPLES_FOR_ML = 25 # 训练ML模型所需的最小正样本数

# 日志记录器配置
console_formatter = logging.Formatter('%(message)s')
detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ssq_analyzer')
logger.setLevel(logging.DEBUG)
logger.propagate = False

global_console_handler = logging.StreamHandler(sys.stdout)
global_console_handler.setFormatter(console_formatter)
logger.addHandler(global_console_handler)

def set_console_verbosity(level=logging.INFO, use_simple_formatter=False):
    """设置控制台日志的级别和格式。"""
    global_console_handler.setLevel(level)
    if use_simple_formatter:
        global_console_handler.setFormatter(console_formatter)
    else:
        global_console_handler.setFormatter(detailed_formatter)

class SuppressOutput:
    """一个用于抑制标准输出和捕获标准错误的上下文管理器。"""
    def __init__(self, suppress_stdout=True, capture_stderr=True):
        self.suppress_stdout = suppress_stdout
        self.capture_stderr = capture_stderr
        self.old_stdout = None
        self.old_stderr = None
        self.stderr_io = None
    def __enter__(self):
        if self.suppress_stdout:
            self.old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        if self.capture_stderr:
            self.old_stderr = sys.stderr
            self.stderr_io = io.StringIO()
            sys.stderr = self.stderr_io
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.capture_stderr and self.old_stderr:
            sys.stderr = self.old_stderr
            captured_stderr_content = self.stderr_io.getvalue()
            if captured_stderr_content.strip(): # 仅当有实际错误输出时记录
                logger.warning(f"捕获到的标准错误输出:\n{captured_stderr_content.strip()}")
        if self.suppress_stdout and self.old_stdout:
            if sys.stdout and not sys.stdout.closed:
                 sys.stdout.close()
            sys.stdout = self.old_stdout
        return False # 不抑制异常

def load_weights_from_file(filepath: str, defaults: Dict) -> Tuple[Dict, str]:
    """
    从JSON文件加载权重。
    返回一个元组: (weights_dict, status_string)。
    status_string可以是:
    - 'loaded_active_config': 文件存在，有效，并已加载。
    - 'defaults_used_new_config_saved': 文件不存在，使用默认值，并保存了新文件。
    - 'defaults_used_config_error': 文件存在但无效，使用默认值。原始文件未修改。
    """
    try:
        with open(filepath, 'r') as f:
            loaded_weights = json.load(f)
        # 合并加载的权重和默认权重，优先使用加载的权重，但确保类型与默认值一致
        merged_weights = defaults.copy()
        for key in defaults:
            if key in loaded_weights:
                if isinstance(defaults[key], (int, float)) and isinstance(loaded_weights[key], (int, float)):
                    merged_weights[key] = type(defaults[key])(loaded_weights[key]) # 保持默认类型
                elif isinstance(defaults[key], str) and isinstance(loaded_weights[key], str):
                     merged_weights[key] = loaded_weights[key]
                elif type(defaults[key]) == type(loaded_weights[key]): # 其他类型，如列表、布尔值
                    merged_weights[key] = loaded_weights[key]
                # else: logger.warning(f"权重文件中键'{key}'的类型不匹配。将使用默认值。") # 可选的警告

        # 检查是否有默认权重中的键未在加载的权重中（例如，新添加的默认权重）
        for key_default in defaults:
            if key_default not in merged_weights:
                logger.info(f"权重文件 {filepath} 缺少键 '{key_default}'。将使用该键的默认值。")
                merged_weights[key_default] = defaults[key_default]

        # 验证候选池比例的有效性
        prop_h = merged_weights.get('CANDIDATE_POOL_PROPORTIONS_HIGH', 0.5)
        prop_m = merged_weights.get('CANDIDATE_POOL_PROPORTIONS_MEDIUM', 0.3)
        if not (0 <= prop_h <= 1 and 0 <= prop_m <= 1 and (prop_h + prop_m) <= 1):
            logger.warning(f"CANDIDATE_POOL_PROPORTIONS 无效 (H:{prop_h}, M:{prop_m})。恢复为默认值。")
            merged_weights['CANDIDATE_POOL_PROPORTIONS_HIGH'] = defaults['CANDIDATE_POOL_PROPORTIONS_HIGH']
            merged_weights['CANDIDATE_POOL_PROPORTIONS_MEDIUM'] = defaults['CANDIDATE_POOL_PROPORTIONS_MEDIUM']

        logger.info(f"权重已从 {filepath} 成功加载并合并。")
        return merged_weights, 'loaded_active_config'
    except FileNotFoundError:
        logger.info(f"权重配置文件 {filepath} 未找到。将使用默认权重并尝试保存。")
        save_weights_to_file(filepath, defaults) # 保存默认权重到新文件
        return defaults.copy(), 'defaults_used_new_config_saved'
    except json.JSONDecodeError:
        logger.error(f"权重文件 {filepath} 格式错误。将使用默认权重。(原始错误文件未修改)")
        return defaults.copy(), 'defaults_used_config_error'
    except Exception as e:
        logger.error(f"加载权重时发生未知错误: {e}。将使用默认权重。(原始错误文件未修改)")
        return defaults.copy(), 'defaults_used_config_error'

def save_weights_to_file(filepath: str, weights_to_save: Dict):
    """将权重保存到JSON文件。"""
    try:
        with open(filepath, 'w') as f:
            json.dump(weights_to_save, f, indent=4)
        logger.info(f"权重已成功保存到 {filepath}")
    except Exception as e:
        logger.error(f"保存权重时出错: {e}")

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """从CSV文件加载数据，尝试多种编码。"""
    try:
        encodings = ['utf-8', 'gbk', 'latin-1'] # 常用编码列表
        for enc in encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                return df
            except UnicodeDecodeError:
                continue # 尝试下一种编码
        logger.error(f"无法使用任何尝试的编码打开文件 {file_path}。")
        return None
    except FileNotFoundError: logger.error(f"错误: {file_path} 找不到"); return None
    except pd.errors.EmptyDataError: logger.error(f"错误: {file_path} 为空"); return None
    except Exception as e: logger.error(f"加载 {file_path} 出错: {e}"); return None

def clean_and_structure(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """清洗和结构化原始DataFrame。"""
    if df is None or df.empty: return None
    df.dropna(subset=['期号', '红球', '蓝球'], inplace=True) # 删除关键列缺失的行
    if df.empty: return None # 如果删除后为空
    try:
        # 清洗期号，确保为整数并排序
        df['期号'] = pd.to_numeric(df['期号'], errors='coerce').astype('Int64') # 转换为可空整数
        df.dropna(subset=['期号'], inplace=True) # 删除期号转换失败的行
        df['期号'] = df['期号'].astype(int)
        df.sort_values(by='期号', ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
    except Exception: return None # 期号处理失败则返回None
    if df.empty: return None

    parsed_rows = []
    for _, row in df.iterrows():
        try:
            rs, bv, pv = str(row.get('红球','')), row.get('蓝球'), row.get('期号')
            if not rs or pd.isna(bv) or pd.isna(pv): continue # 跳过无效行

            bn = int(bv);
            if not (1 <= bn <= 16): continue # 验证蓝球范围

            reds_str = rs.split(',')
            if len(reds_str) != 6: continue # 验证红球数量
            reds_int = sorted([int(x) for x in reds_str if 1 <= int(x) <= 33]) # 转换并验证红球范围
            if len(reds_int) != 6: continue # 再次验证红球数量

            rd = {'期号': int(pv)};
            if '日期' in row and pd.notna(row['日期']): rd['日期'] = str(row['日期']).strip()
            for i in range(6):
                rd[f'red{i+1}'] = reds_int[i]
                rd[f'red_pos{i+1}'] = reds_int[i] # 位置特征暂时用排序后的红球
            rd['blue'] = bn; parsed_rows.append(rd)
        except Exception: continue # 跳过解析错误的行
    return pd.DataFrame(parsed_rows).sort_values(by='期号').reset_index(drop=True) if parsed_rows else None

def feature_engineer(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """对DataFrame进行特征工程。"""
    if df is None or df.empty: return None
    red_cols = [f'red{i+1}' for i in range(6)]
    if not all(c in df.columns for c in red_cols + ['blue', '期号']): return None

    df_fe = df.copy()
    for r_col in red_cols: df_fe[r_col] = pd.to_numeric(df_fe[r_col], errors='coerce')
    df_fe.dropna(subset=red_cols, inplace=True) # 删除红球转换失败的行

    # 基本红球特征
    df_fe['red_sum'] = df_fe[red_cols].sum(axis=1) # 和值
    df_fe['red_span'] = df_fe[red_cols].max(axis=1) - df_fe[red_cols].min(axis=1) # 跨度

    red_values_numeric = True
    try:
        for col in red_cols: # 确保红球列是数值类型
            if not pd.api.types.is_numeric_dtype(df_fe[col]):
                 df_fe[col] = pd.to_numeric(df_fe[col], errors='coerce')
            if df_fe[col].isnull().any():
                red_values_numeric = False; break
        if not red_values_numeric: df_fe.dropna(subset=red_cols, inplace=True)
    except Exception:
        red_values_numeric = False

    if red_values_numeric and not df_fe.empty:
         df_fe['red_odd_count'] = df_fe[red_cols].apply(lambda r: sum(int(x) % 2 != 0 for x in r), axis=1) # 奇数个数
         for zone, (start, end) in RED_ZONES.items(): # 区间特征
             df_fe[f'red_{zone}_count'] = df_fe[red_cols].apply(lambda r: sum(start <= int(x) <= end for x in r), axis=1)
         
         # 重号个数 (与上一期比较)
         df_fe['current_reds_str'] = df_fe[red_cols].astype(int).astype(str).agg(','.join, axis=1)
         df_fe['prev_reds_str'] = df_fe['current_reds_str'].shift(1)
         df_fe['red_repeat_count'] = df_fe.apply(
             lambda r: len(set(int(x) for x in r['prev_reds_str'].split(',')) & set(int(x) for x in r['current_reds_str'].split(',')))
             if pd.notna(r['prev_reds_str']) and pd.notna(r['current_reds_str']) else 0,
             axis=1
         )
         df_fe.drop(columns=['current_reds_str', 'prev_reds_str'], inplace=True, errors='ignore')
    else: # 如果红球值处理失败
        df_fe['red_odd_count'] = np.nan; df_fe['red_repeat_count'] = np.nan
        for zone in RED_ZONES: df_fe[f'red_{zone}_count'] = np.nan

    # 连号对数 (基于排序后的红球)
    red_pos_cols = [f'red_pos{i+1}' for i in range(6)] # 位置特征暂时用排序后的红球
    if not df_fe.empty and all(c in df_fe.columns for c in red_pos_cols):
        pos_values_numeric = True
        try: # 确保位置列是数值类型
            for col in red_pos_cols:
                 if not pd.api.types.is_numeric_dtype(df_fe[col]):
                    df_fe[col] = pd.to_numeric(df_fe[col], errors='coerce')
                 if df_fe[col].isnull().any():
                    pos_values_numeric = False; break
            if not pos_values_numeric: df_fe.dropna(subset=red_pos_cols, inplace=True)
        except Exception:
            pos_values_numeric = False

        if pos_values_numeric and not df_fe.empty:
            df_fe['red_consecutive_pairs'] = df_fe.apply(
                lambda r: sum(1 for i in range(5) if int(r[red_pos_cols[i]]) + 1 == int(r[red_pos_cols[i+1]])), axis=1
            )
        else:
            df_fe['red_consecutive_pairs'] = np.nan
    else: df_fe['red_consecutive_pairs'] = np.nan


    # 蓝球特征
    if 'blue' in df_fe.columns and pd.api.types.is_numeric_dtype(df_fe['blue']):
        df_fe['blue'] = pd.to_numeric(df_fe['blue'], errors='coerce').dropna().astype(int)
        df_fe['blue_is_odd'] = df_fe['blue'] % 2 != 0 # 奇偶性
        df_fe['blue_is_large'] = df_fe['blue'] > 8 # 大小性 (大于8为大)
        primes = {2, 3, 5, 7, 11, 13}; df_fe['blue_is_prime'] = df_fe['blue'].apply(lambda x: x in primes if pd.notna(x) else False) # 质数性
    else: df_fe['blue_is_odd'] = np.nan; df_fe['blue_is_large'] = np.nan; df_fe['blue_is_prime'] = np.nan
    return df_fe

def analyze_frequency_omission(df: pd.DataFrame, weights_config: Dict) -> dict:
    """分析号码的频率和遗漏。"""
    if df is None or df.empty: return {}
    red_cols = [f'red{i+1}' for i in range(6)]
    most_recent_idx = len(df) - 1 # 最新一期的索引
    if most_recent_idx < 0: return {}

    # 确定有效的数值型红球列和蓝球列
    num_red_cols = [c for c in red_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    num_blue_col = 'blue' if 'blue' in df.columns and pd.api.types.is_numeric_dtype(df['blue']) else None
    if not num_red_cols and not num_blue_col: return {} # 没有可分析的列

    # 计算频率
    all_reds_flat = df[num_red_cols].values.flatten() if num_red_cols else np.array([])
    red_freq = Counter(all_reds_flat[~np.isnan(all_reds_flat)].astype(int))
    blue_freq = Counter(df[num_blue_col].dropna().astype(int)) if num_blue_col else Counter()

    # 计算当前遗漏和最大历史遗漏 (红球)
    current_omission = {}
    max_historical_omission_red = {num: 0 for num in RED_BALL_RANGE}
    recent_N_freq_red = {num: 0 for num in RED_BALL_RANGE} # 近N期频率

    if num_red_cols:
        for num in RED_BALL_RANGE:
            appearances = (df[num_red_cols].astype(float) == float(num)).any(axis=1) # 标记号码出现的行
            app_indices = df.index[appearances] # 号码出现的索引

            if not app_indices.empty:
                current_omission[num] = most_recent_idx - app_indices.max()
                # 计算最大历史遗漏
                max_o = app_indices[0] # 从开始到第一次出现的遗漏
                for i in range(len(app_indices) - 1):
                    max_o = max(max_o, app_indices[i+1] - app_indices[i] - 1) # 两次出现之间的遗漏
                max_o = max(max_o, most_recent_idx - app_indices.max()) # 从最后一次出现到现在的遗漏
                max_historical_omission_red[num] = max_o
            else: # 号码从未出现
                current_omission[num] = len(df)
                max_historical_omission_red[num] = len(df)

        # 计算近N期频率
        recent_df_slice = df.tail(RECENT_FREQ_WINDOW)
        if not recent_df_slice.empty:
            recent_reds_flat = recent_df_slice[num_red_cols].values.flatten()
            recent_freq_counts = Counter(recent_reds_flat[~np.isnan(recent_reds_flat)].astype(int))
            for num in RED_BALL_RANGE:
                recent_N_freq_red[num] = recent_freq_counts.get(num, 0)

    # 计算当前遗漏 (蓝球)
    if num_blue_col:
        for num in BLUE_BALL_RANGE:
            app_indices = df.index[df[num_blue_col].astype(float) == float(num)]
            latest_idx = app_indices.max() if not app_indices.empty else -1
            current_omission[f'blue_{num}'] = len(df) if latest_idx == -1 else most_recent_idx - latest_idx

    # 计算平均出现间隔
    avg_interval = {num: len(df) / (red_freq.get(num, 0) + 1e-9) for num in RED_BALL_RANGE} # 加1e-9避免除零
    for num in BLUE_BALL_RANGE: avg_interval[f'blue_{num}'] = len(df) / (blue_freq.get(num, 0) + 1e-9)

    # 识别冷热号
    red_items = sorted(red_freq.items(), key=lambda item: item[1], reverse=True)
    blue_items = sorted(blue_freq.items(), key=lambda item: item[1], reverse=True)
    hot_reds = [n for n, _ in red_items[:max(0, int(len(RED_BALL_RANGE) * 0.2))]] # 频率最高的20%为热号
    cold_reds = [n for n, _ in red_items[min(len(red_items)-1, int(len(RED_BALL_RANGE) * 0.8)):] if n not in hot_reds] # 频率最低的20%为冷号 (排除已是热号的情况)
    hot_blues = [n for n, _ in blue_items[:max(0, int(len(BLUE_BALL_RANGE) * 0.3))]] # 频率最高的30%为热号
    cold_blues = [n for n, _ in blue_items[min(len(blue_items)-1, int(len(BLUE_BALL_RANGE) * 0.7)):] if n not in hot_blues] # 频率最低的30%为冷号

    return {'red_freq': red_freq, 'blue_freq': blue_freq, 'current_omission': current_omission,
            'average_interval': avg_interval, 'hot_reds': hot_reds, 'cold_reds': cold_reds,
            'hot_blues': hot_blues, 'cold_blues': cold_blues,
            'max_historical_omission_red': max_historical_omission_red,
            'recent_N_freq_red': recent_N_freq_red}

def analyze_patterns(df: pd.DataFrame, weights_config: Dict) -> dict:
    """分析历史数据的模式，如和值、跨度、奇偶比等。"""
    if df is None or df.empty: return {}
    res = {}
    def safe_mode(series): # 安全地获取众数
        return int(series.mode().iloc[0]) if not series.empty and not series.mode().empty else None

    # 和值、跨度统计
    for col, name in [('red_sum', 'sum'), ('red_span', 'span')]:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and not df[col].empty:
            res[f'{name}_stats'] = df[col].describe().to_dict()
            res[f'most_common_{name}'] = safe_mode(df[col])
    # 红球奇偶比
    if 'red_odd_count' in df.columns and pd.api.types.is_numeric_dtype(df['red_odd_count']) and not df['red_odd_count'].empty:
        res['most_common_odd_even_count'] = safe_mode(df['red_odd_count'].dropna())
    # 红球区间分布
    zone_cols = [f'red_{zone}_count' for zone in RED_ZONES.keys()]
    if all(c in df.columns and pd.api.types.is_numeric_dtype(df[c]) for c in zone_cols) and not df.empty:
        zc_df = df[zone_cols].dropna().astype(int)
        if not zc_df.empty:
            dist_counts = zc_df.apply(tuple, axis=1).value_counts()
            res['most_common_zone_distribution'] = dist_counts.index[0] if not dist_counts.empty else None
    # 蓝球奇偶、大小分布
    for col_name, data_key in [('blue_is_odd', 'blue_odd_counts'), ('blue_is_large', 'blue_large_counts')]:
        if col_name in df.columns and not df[col_name].dropna().empty:
            counts = df[col_name].dropna().astype(bool).value_counts()
            res[data_key] = {bool(k): int(v) for k, v in counts.items()}
    return res

def analyze_associations(df: pd.DataFrame, weights_config: Dict) -> pd.DataFrame:
    """使用Apriori算法分析红球之间的关联规则。"""
    min_s = weights_config.get('ARM_MIN_SUPPORT', 0.008)
    min_c = weights_config.get('ARM_MIN_CONFIDENCE', 0.35)
    min_l = weights_config.get('ARM_MIN_LIFT', 1.1)

    if df is None or df.empty or len(df) < 2: return pd.DataFrame() # 数据不足
    red_cols = [f'red{i+1}' for i in range(6)]
    if not all(c in df.columns and pd.api.types.is_numeric_dtype(df[c]) for c in red_cols): return pd.DataFrame() # 红球列无效

    tx_df = df.dropna(subset=red_cols).copy() # 删除包含NaN红球的行
    if tx_df.empty: return pd.DataFrame()
    try:
        tx_df[red_cols] = tx_df[red_cols].astype(int) # 确保红球列为整数
        txs = tx_df[red_cols].astype(str).values.tolist() # 转换为字符串列表的列表作为交易数据
    except ValueError: return pd.DataFrame()
    if not txs: return pd.DataFrame()

    te = TransactionEncoder();
    try:
        te_ary = te.fit_transform(txs)
    except Exception: return pd.DataFrame()

    df_oh = pd.DataFrame(te_ary, columns=te.columns_) # 转换为One-Hot编码的DataFrame
    if df_oh.empty: return pd.DataFrame()
    try:
        # 确保最小支持度对于数据集大小是合理的
        actual_min_support = max(2/len(df_oh) if len(df_oh)>0 else min_s, min_s) # 至少需要出现2次
        f_items = apriori(df_oh, min_support=actual_min_support, use_colnames=True) # 查找频繁项集
        if f_items.empty: return pd.DataFrame()
        rules = association_rules(f_items, metric="lift", min_threshold=min_l) # 生成关联规则
        # 按置信度和提升度筛选规则
        if 'confidence' in rules.columns and isinstance(rules['confidence'], pd.Series):
            return rules[rules['confidence'] >= min_c].sort_values(by='lift', ascending=False)
        else:
            logger.debug("ARM: 关联规则DataFrame中'confidence'列存在问题。")
            return rules.sort_values(by='lift', ascending=False) if 'lift' in rules.columns else pd.DataFrame()

    except Exception as e_apriori:
        logger.debug(f"Apriori/AssociationRules 执行失败: {e_apriori}")
        return pd.DataFrame()

def analyze_associations_with_properties(df: pd.DataFrame, weights_config: Dict) -> pd.DataFrame:
    """占位符：分析包含号码属性的关联规则 (更复杂，未完全实现)。"""
    logger.info("analyze_associations_with_properties 是一个占位符，尚未完全实现基于属性的ARM。")
    # 此函数需要重大更改:
    # 1. 定义如何将属性（如红球和值分段、蓝球奇偶等）添加到交易中。
    #    示例交易: ['red1_val_5', 'red2_val_10', ..., 'red_sum_seg_mid', 'blue_is_odd_True']
    # 2. 调整 TransactionEncoder 和 apriori 调用。
    # 3. 调整在 generate_combinations 中如何解释和应用规则以获得奖励。
    # 目前，它返回与原始ARM相同的结果。
    return analyze_associations(df, weights_config)

def get_score_segment(score: float, boundaries: List[int], labels: List[str]) -> str:
    """根据分数和边界确定分数段标签。"""
    if score is None or pd.isna(score): return "未知" # 处理NaN分数
    tolerance = 1e-9 # 浮点数比较的容差

    if score < boundaries[0] - tolerance: return labels[0] if labels else "未知"
    if score > boundaries[-1] + tolerance: return labels[-1] if labels else "未知"

    for i in range(len(boundaries) - 1):
        if i == 0: # 第一个分段，包含下边界
             if boundaries[i] <= score <= boundaries[i+1] + tolerance: return labels[i]
        elif boundaries[i] < score - tolerance and score <= boundaries[i+1] + tolerance : return labels[i] # 后续分段，下边界不包含
    return "未知" # 如果分数不在任何定义的段内

def analyze_winning_red_ball_score_segments(df: pd.DataFrame, red_ball_scores: dict, score_boundaries: List[int], score_labels: List[str]) -> Tuple[Dict[str, int], Dict[str, float]]:
    """分析历史上中奖红球的分数段分布。"""
    seg_counts = {label: 0 for label in score_labels}; total_win_reds = 0
    red_cols = [f'red{i+1}' for i in range(6)]
    if df is None or df.empty or not red_ball_scores or not all(c in df.columns for c in red_cols):
        return seg_counts, {label: 0.0 for label in score_labels}

    for _, row in df.iterrows():
        win_reds = []
        valid_row = True
        for c in red_cols: # 提取并验证当期中奖红球
            val = row.get(c)
            if pd.isna(val): valid_row = False; break
            try:
                num_val = int(float(val))
                if num_val not in RED_BALL_RANGE: valid_row = False; break
                win_reds.append(num_val)
            except (ValueError, TypeError): valid_row = False; break
        if not valid_row or len(win_reds) != 6: continue

        for ball in win_reds: # 为每个中奖红球确定其分数段
            score = red_ball_scores.get(ball)
            if score is not None and pd.notna(score) and isinstance(score, (int, float)):
                segment = get_score_segment(score, score_boundaries, score_labels)
                if segment in seg_counts and segment != "未知":
                    seg_counts[segment] += 1; total_win_reds += 1

    seg_pcts = {seg: (cnt / total_win_reds) * 100 if total_win_reds > 0 else 0.0 for seg, cnt in seg_counts.items()}
    return seg_counts, seg_pcts

def create_lagged_features(df: pd.DataFrame, lags: List[int]) -> Optional[pd.DataFrame]:
    """为DataFrame创建滞后特征和交互特征。"""
    if df is None or df.empty or not lags: return None
    # 基础特征列候选
    base_cols_candidates = ['red_sum', 'red_span', 'red_odd_count', 'red_consecutive_pairs', 'red_repeat_count'] + \
                           [f'red_{zone}_count' for zone in RED_ZONES.keys()] + \
                           ['blue', 'blue_is_odd', 'blue_is_large', 'blue_is_prime']
    df_temp = df.copy()
    existing_lag_cols = [] # 存储实际用于创建滞后特征的列名
    for col in base_cols_candidates:
        if col in df_temp.columns:
            if pd.api.types.is_bool_dtype(df_temp[col].dtype):
                df_temp[col] = df_temp[col].astype(int) # 布尔转整数
                existing_lag_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df_temp[col].dtype):
                existing_lag_cols.append(col)
            else: # 尝试将非数值、非布尔类型转换为数值
                try:
                    df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
                    if pd.api.types.is_numeric_dtype(df_temp[col].dtype): existing_lag_cols.append(col)
                except Exception: pass
    
    if not existing_lag_cols: return None
    
    df_lagged = df_temp[existing_lag_cols].copy()
    
    # 创建滞后特征
    for lag_val in lags:
        if lag_val > 0:
            for col in existing_lag_cols: df_lagged[f'{col}_lag{lag_val}'] = df_lagged[col].shift(lag_val)

    # 创建交互特征 (基于原始非滞后列)
    df_with_interactions = df_temp.copy()
    for col1, col2 in ML_INTERACTION_PAIRS: # 特征对相乘
        if col1 in df_with_interactions.columns and col2 in df_with_interactions.columns and \
           pd.api.types.is_numeric_dtype(df_with_interactions[col1]) and pd.api.types.is_numeric_dtype(df_with_interactions[col2]):
            interaction_col_name = f'{col1}_x_{col2}'
            df_with_interactions[interaction_col_name] = df_with_interactions[col1] * df_with_interactions[col2]
            if interaction_col_name not in existing_lag_cols:
                 existing_lag_cols.append(interaction_col_name)
                 df_lagged[interaction_col_name] = df_with_interactions[interaction_col_name]
                 for lag_val in lags: # 为新交互特征创建滞后版本
                     if lag_val > 0:
                         df_lagged[f'{interaction_col_name}_lag{lag_val}'] = df_lagged[interaction_col_name].shift(lag_val)

    for col_s in ML_INTERACTION_SELF: # 特征自交互 (平方)
        if col_s in df_with_interactions.columns and pd.api.types.is_numeric_dtype(df_with_interactions[col_s]):
            interaction_col_name = f'{col_s}_sq'
            df_with_interactions[interaction_col_name] = df_with_interactions[col_s] ** 2
            if interaction_col_name not in existing_lag_cols:
                existing_lag_cols.append(interaction_col_name)
                df_lagged[interaction_col_name] = df_with_interactions[interaction_col_name]
                for lag_val in lags: # 为新交互特征创建滞后版本
                    if lag_val > 0:
                        df_lagged[f'{interaction_col_name}_lag{lag_val}'] = df_lagged[interaction_col_name].shift(lag_val)
    
    df_lagged.dropna(inplace=True)
    if df_lagged.empty: return None
    
    # 仅选择滞后列作为最终特征集
    feature_cols = [col for col in df_lagged.columns if any(f'_lag{lag_val}' in col for lag_val in lags)]
    return df_lagged[feature_cols] if feature_cols else None


def train_single_model(model_type, ball_type_str, ball_number, X, y, params, min_pos_samples,
                       lgbm_ref, svc_ref, scaler_ref, pipe_ref, logreg_ref, xgb_ref):
    """训练单个机器学习模型。"""
    if y.sum() < min_pos_samples or len(y.unique()) < 2: # 正样本过少或只有一类样本
        return None, None
    model_key = f'{model_type}_{ball_number}'
    model_params = params.copy()

    # 处理类别不平衡
    positive_count = y.sum()
    negative_count = len(y) - positive_count
    scale_pos_weight_val = negative_count / (positive_count + 1e-9) if positive_count > 0 else 1.0
    class_weight_val = 'balanced' if positive_count > 0 and negative_count > 0 else None

    # --- 特征选择占位符 (例如 RFE) ---
    # if X.shape[1] > 10: # RFE应用示例条件
    #     try:
    #         estimator_for_rfe = lgbm_ref(random_state=42, n_jobs=1)
    #         num_features_rfe = max(1, X.shape[1] // 2) # 根据需要调整n_features_to_select
    #         rfe_selector = RFE(estimator=estimator_for_rfe, n_features_to_select=num_features_rfe, step=0.1)
    #         rfe_selector.fit(X, y)
    #         X_selected = X.loc[:, rfe_selector.support_]
    #         logger.debug(f"RFE for {model_key}: selected {X_selected.shape[1]}/{X.shape[1]} features.")
    #         X_to_use = X_selected
    #     except Exception as e_rfe:
    #         logger.warning(f"RFE for {model_key} failed: {e_rfe}. Using all features.")
    #         X_to_use = X
    # else:
    #     X_to_use = X
    X_to_use = X # 目前默认使用所有特征

    model = None
    try:
        if model_type == 'lgbm':
            model_params['scale_pos_weight'] = scale_pos_weight_val
            model = lgbm_ref(**model_params)
            model.fit(X_to_use, y)
        elif model_type == 'xgb':
            model_params['scale_pos_weight'] = scale_pos_weight_val
            model = xgb_ref(**model_params)
            model.fit(X_to_use, y)
        elif model_type == 'logreg':
            if class_weight_val: model_params['class_weight'] = class_weight_val
            model_params.pop('scale_pos_weight', None)
            model = pipe_ref([('scaler', scaler_ref()), ('logreg', logreg_ref(**model_params))]) # 流水线：标准化 + 逻辑回归
            model.fit(X_to_use, y)
        elif model_type == 'svc':
            if class_weight_val: model_params['class_weight'] = class_weight_val
            model_params.pop('scale_pos_weight', None)
            svc_actual_params = model_params.copy()
            svc_actual_params['probability'] = True # 确保SVC启用概率估计
            model = pipe_ref([('scaler', scaler_ref()), ('svc', svc_ref(**svc_actual_params))]) # 流水线：标准化 + SVC
            model.fit(X_to_use, y)
            svc_estimator = model.named_steps.get('svc')
            if not (svc_estimator and hasattr(svc_estimator, 'probability') and svc_estimator.probability):
                logger.debug(f"SVC for {ball_type_str} {ball_number} did not enable probability correctly.")
                model = None # 如果概率未正确启用，则模型无效
        
        # --- 概率校准占位符 ---
        # if model and model_type in ['logreg', 'svc_without_internal_proba']: # 示例：某些模型可能需要外部校准
        #     try:
        #         # method='isotonic' (保序) 或 'sigmoid' (Sigmoid)
        #         calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit') # 'prefit' 如果模型已训练
        #         # model = calibrated_model # 替换原始模型
        #         logger.debug(f"Probability calibration considered for {model_key}.")
        #     except Exception as e_calib:
        #         logger.warning(f"Calibration failed for {model_key}: {e_calib}")
        
        return model, model_key
    except Exception as e_train:
        logger.debug(f"Training {model_type} for {ball_type_str} {ball_number} failed: {e_train}")
        return None, None

def train_prediction_models(df_train_raw: pd.DataFrame, ml_lags_list: List[int], weights_config: Dict) -> Optional[dict]:
    """为每个球号训练多种类型的预测模型。"""
    # --- 每个模型或每个球号的超参数调整占位符 (使用Optuna) ---
    # 这通常是一个独立的、复杂的Optuna研究，预先运行以确定最佳的LGBM_PARAMS, XGB_PARAMS等，
    # 或者如果计算上可行，则在此处动态调整。目前使用全局定义的参数。

    X = create_lagged_features(df_train_raw.copy(), ml_lags_list) # 创建滞后特征
    if X is None or X.empty: logger.warning("ML训练: 滞后特征为空。"); return None

    target_df = df_train_raw.loc[X.index].copy() # 对齐目标变量与特征 (因滞后产生的NaN已被删除)
    if target_df.empty: logger.warning("ML训练: 目标DataFrame为空。"); return None

    red_cols = [f'red{i+1}' for i in range(6)]
    if not all(c in target_df.columns for c in red_cols + ['blue']): # 检查球号列是否存在
        logger.error("ML训练: 目标DataFrame中缺少球号列。"); return None
    try: # 确保球号列为整数
        for col in red_cols + ['blue']: target_df[col] = pd.to_numeric(target_df[col], errors='coerce').astype(int)
    except (ValueError, TypeError): logger.error("ML训练: 转换球号列为整数失败。"); return None

    trained_models = {'red': {}, 'blue': {}, 'feature_cols': X.columns.tolist()} # 存储训练好的模型
    min_pos = MIN_POSITIVE_SAMPLES_FOR_ML # 最小正样本数

    futures_map = {} # 用于管理并发任务
    num_cpus = os.cpu_count()
    max_workers = num_cpus if num_cpus and num_cpus > 1 else 1 # 并行工作进程数

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 为每个红球训练模型
        for ball_num in RED_BALL_RANGE:
            y_red = target_df[red_cols].apply(lambda row: ball_num in row.values, axis=1).astype(int) # 目标变量：该红球是否出现
            for mt, mp in [('lgbm', LGBM_PARAMS), ('xgb', XGB_PARAMS), ('logreg', LOGISTIC_REG_PARAMS), ('svc', SVC_PARAMS)]:
                f = executor.submit(train_single_model, mt, '红', ball_num, X, y_red, mp, min_pos,
                                    LGBMClassifier, SVC, StandardScaler, Pipeline, LogisticRegression, xgb.XGBClassifier)
                futures_map[f] = ('red', mt, ball_num)
        # 为每个蓝球训练模型
        for ball_num in BLUE_BALL_RANGE:
            y_blue = (target_df['blue'] == ball_num).astype(int) # 目标变量：该蓝球是否出现
            for mt, mp in [('lgbm', LGBM_PARAMS), ('xgb', XGB_PARAMS), ('logreg', LOGISTIC_REG_PARAMS), ('svc', SVC_PARAMS)]:
                f = executor.submit(train_single_model, mt, '蓝', ball_num, X, y_blue, mp, min_pos,
                                    LGBMClassifier, SVC, StandardScaler, Pipeline, LogisticRegression, xgb.XGBClassifier)
                futures_map[f] = ('blue', mt, ball_num)

    models_trained_count = 0
    for future in concurrent.futures.as_completed(futures_map): # 收集训练结果
        ball_type_str, model_type, ball_number = futures_map[future]
        try:
            model, model_key = future.result()
            if model and model_key:
                trained_models[ball_type_str][model_key] = model
                models_trained_count +=1
        except Exception as e_future:
            logger.warning(f"获取 {ball_type_str} {ball_number} {model_type} 的训练结果时发生异常: {e_future}")

    logger.debug(f"ML模型训练完成。成功训练 {models_trained_count} 个模型。")
    return trained_models if models_trained_count > 0 else None

def predict_next_draw_probabilities(df_historical: pd.DataFrame, trained_models: Optional[dict], ml_lags_list: List[int], weights_config: Dict) -> Dict:
    """使用训练好的模型预测下一次开奖的号码出现概率。"""
    probs = {'red': {}, 'blue': {}} # 存储预测概率
    if not trained_models or df_historical is None or df_historical.empty: return probs

    feat_cols = trained_models.get('feature_cols') # 获取训练时使用的特征列
    if not feat_cols: logger.warning("ML预测: trained_models中无feature_cols。"); return probs

    max_hist_lag = max(ml_lags_list) if ml_lags_list else 0
    # 需要足够的数据来创建至少一行滞后特征用于预测
    if len(df_historical) < max_hist_lag + 1 :
        logger.warning(f"ML预测: 历史数据不足 ({len(df_historical)})，无法满足滞后 ({max_hist_lag})。至少需要 {max_hist_lag + 1} 条。")
        return probs

    # 使用最新的可用数据创建下一期开奖的特征
    # df_historical.tail(max_hist_lag + 1) 提供了create_lagged_features所需的确切窗口，
    # 以生成df_historical最后一行 *之后* 的那一期开奖的特征。
    predict_X = create_lagged_features(df_historical.tail(max_hist_lag + 1).copy(), ml_lags_list)
    
    if predict_X is None or predict_X.empty:
        logger.warning("ML预测: create_lagged_features为预测返回了空结果。")
        return probs
    if len(predict_X) != 1: # create_lagged_features和dropna后应始终只有一行
        logger.error(f"ML预测: 预测特征应为1行，实际得到 {len(predict_X)} 行。")
        return probs

    try:
        # 确保predict_X包含模型训练时的所有列，顺序正确，缺失值用0填充
        predict_X = predict_X.reindex(columns=feat_cols, fill_value=0)
        # 转换为数值型并填充任何剩余的NaN (例如，如果新的交互特征全为NaN)
        for col in predict_X.columns: predict_X[col] = pd.to_numeric(predict_X[col], errors='coerce')
        predict_X.fillna(0, inplace=True)
        if predict_X.isnull().values.any(): logger.error("ML预测: 处理后预测特征中仍存在NaN。"); return probs
    except Exception as e_pred_preprocess:
        logger.error(f"ML预测: 预处理预测特征时出错: {e_pred_preprocess}."); return probs

    for ball_type_key, ball_val_range, models_sub_dict in [('red', RED_BALL_RANGE, trained_models.get('red', {})),
                                                           ('blue', BLUE_BALL_RANGE, trained_models.get('blue', {}))]:
        if not models_sub_dict: continue
        for ball_val in ball_val_range:
            ball_preds = [] # 存储来自不同模型的对该球的预测概率
            for model_variant in ['lgbm', 'xgb', 'logreg', 'svc']: # 遍历模型类型
                model_instance = models_sub_dict.get(f'{model_variant}_{ball_val}')
                if model_instance and hasattr(model_instance, 'predict_proba'):
                    try:
                        # 如果训练时使用了RFE，需要选择特征子集 (需要为每个模型存储selected_features)
                        # 目前假设所有模型使用相同的 `feat_cols`
                        # X_for_this_model = predict_X[trained_models[ball_type_key][f'{model_variant}_{ball_val}'].feature_names_in_] # 如果存储了特征名称
                        proba = model_instance.predict_proba(predict_X)[0][1] # 类别1 (球出现) 的概率
                        ball_preds.append(proba)
                    except Exception as e_proba:
                        logger.debug(f"ML预测: {ball_type_key} {ball_val} {model_variant}的predict_proba失败: {e_proba}")
            if ball_preds: probs[ball_type_key][ball_val] = np.mean(ball_preds) # 取不同模型预测概率的平均值
    return probs


def calculate_scores(freq_omission_data: dict, pattern_analysis_data: dict, predicted_probabilities: dict, weights_config: Dict) -> dict:
    """根据频率、遗漏、模式和ML预测概率计算每个球的分数。"""
    r_scores, b_scores = {}, {} # 红球分数，蓝球分数
    # 从分析数据中获取所需信息
    r_freq = freq_omission_data.get('red_freq', {}); b_freq = freq_omission_data.get('blue_freq', {})
    omission = freq_omission_data.get('current_omission', {}); avg_int = freq_omission_data.get('average_interval', {})
    max_hist_omission_r = freq_omission_data.get('max_historical_omission_red', {})
    recent_N_freq_r = freq_omission_data.get('recent_N_freq_red', {})

    # 处理频率数据可能为空或不完整的情况
    r_freq_series = pd.Series(r_freq).reindex(list(RED_BALL_RANGE), fill_value=0)
    r_freq_rank = r_freq_series.rank(method='min', ascending=False) # 高频 = 低排名数字
    b_freq_series = pd.Series(b_freq).reindex(list(BLUE_BALL_RANGE), fill_value=0)
    b_freq_rank = b_freq_series.rank(method='min', ascending=False)

    r_pred_probs = predicted_probabilities.get('red', {}); b_pred_probs = predicted_probabilities.get('blue', {})
    max_r_rank, max_b_rank = len(RED_BALL_RANGE), len(BLUE_BALL_RANGE)

    # 归一化近期频率分数
    recent_freq_values = [v for v in recent_N_freq_r.values() if v is not None]
    min_rec_freq = min(recent_freq_values) if recent_freq_values else 0
    max_rec_freq = max(recent_freq_values) if recent_freq_values else 0

    # 计算红球分数
    for num in RED_BALL_RANGE:
        # 频率分数：排名越高（数字越小）分数越高
        freq_s = max(0, (max_r_rank - (r_freq_rank.get(num, max_r_rank+1)-1))/max_r_rank * weights_config['FREQ_SCORE_WEIGHT'])
        
        # 遗漏分数：当前遗漏与平均间隔的偏差（高斯衰减）
        dev = omission.get(num, max_r_rank*2) - avg_int.get(num, max_r_rank*2) # 偏差越大，惩罚越大
        omit_s = max(0, weights_config['OMISSION_SCORE_WEIGHT'] * np.exp(-0.005 * dev**2)) # 偏差小（接近平均间隔）= 分数高

        # 最大遗漏比率分数
        max_o = max_hist_omission_r.get(num, 0)
        cur_o = omission.get(num, 0)
        max_omit_ratio_s = 0
        if max_o > 0:
            ratio_o = cur_o / max_o # 当前遗漏与历史最大遗漏的比率
            # 当期遗漏接近或超过历史最大时分数增加
            max_omit_ratio_s = max(0, min(1.0, ratio_o)) * weights_config['MAX_OMISSION_RATIO_SCORE_WEIGHT_RED']
            if ratio_o > 1.2: max_omit_ratio_s *= 1.2 # 显著超过最大值则有额外奖励
            if ratio_o < 0.2: max_omit_ratio_s *= 0.5 # 远低于最大值（最近出现）则有惩罚
        else: # 如果从未遗漏（或无历史），若当前遗漏则满分，否则0分
            max_omit_ratio_s = weights_config['MAX_OMISSION_RATIO_SCORE_WEIGHT_RED'] if cur_o > 0 else 0

        # 近期频率分数
        rec_f = recent_N_freq_r.get(num, 0)
        norm_rec_f_score = 0
        if max_rec_freq > min_rec_freq: # 避免近期频率都相同时除零
            norm_rec_f_score = (rec_f - min_rec_freq) / (max_rec_freq - min_rec_freq)
        elif max_rec_freq > 0 : # 如果都相同且非零，则给中间分数
             norm_rec_f_score = 0.5 if rec_f > 0 else 0
        recent_freq_s = max(0, norm_rec_f_score * weights_config['RECENT_FREQ_SCORE_WEIGHT_RED'])

        # ML预测概率分数
        ml_s = max(0, r_pred_probs.get(num, 0.0) * weights_config['ML_PROB_SCORE_WEIGHT_RED'])

        r_scores[num] = freq_s + omit_s + ml_s + max_omit_ratio_s + recent_freq_s

    # 计算蓝球分数
    for num in BLUE_BALL_RANGE:
        freq_s = max(0, (max_b_rank - (b_freq_rank.get(num, max_b_rank+1)-1))/max_b_rank * weights_config['BLUE_FREQ_SCORE_WEIGHT'])
        dev = omission.get(f'blue_{num}', max_b_rank*2) - avg_int.get(f'blue_{num}', max_b_rank*2)
        omit_s = max(0, weights_config['BLUE_OMISSION_SCORE_WEIGHT'] * np.exp(-0.01 * dev**2)) # 蓝球范围小，调整衰减系数
        ml_s = max(0, b_pred_probs.get(num, 0.0) * weights_config['ML_PROB_SCORE_WEIGHT_BLUE'])
        b_scores[num] = freq_s + omit_s + ml_s

    # 将所有分数归一化到0-100范围
    all_s_vals = [s for s in list(r_scores.values()) + list(b_scores.values()) if pd.notna(s) and np.isfinite(s)]
    if all_s_vals:
        min_s_val, max_s_val = min(all_s_vals), max(all_s_vals)
        if (max_s_val - min_s_val) > 1e-9: # 避免所有分数相同时除零
            r_scores = {n: max(0,min(100,(s-min_s_val)/(max_s_val-min_s_val)*100)) if pd.notna(s) and np.isfinite(s) else 0 for n,s in r_scores.items()}
            b_scores = {n: max(0,min(100,(s-min_s_val)/(max_s_val-min_s_val)*100)) if pd.notna(s) and np.isfinite(s) else 0 for n,s in b_scores.items()}
        else: # 所有分数几乎相同，赋中间值
            r_scores = {n:50.0 for n in RED_BALL_RANGE}; b_scores = {n:50.0 for n in BLUE_BALL_RANGE}
    else: # 没有计算出有效分数
        r_scores = {n:0.0 for n in RED_BALL_RANGE}; b_scores = {n:0.0 for n in BLUE_BALL_RANGE}
    return {'red_scores': r_scores, 'blue_scores': b_scores}

def get_combo_properties(red_balls: List[int]) -> Dict:
    """获取红球组合的属性 (和值, 奇数个数, 区间分布)。"""
    props = {}
    props['sum'] = sum(red_balls)
    props['odd_count'] = sum(x % 2 != 0 for x in red_balls)
    zones_count = [0,0,0] # Zone1, Zone2, Zone3 计数
    for ball_num in red_balls:
        if RED_ZONES['Zone1'][0] <= ball_num <= RED_ZONES['Zone1'][1]: zones_count[0]+=1
        elif RED_ZONES['Zone2'][0] <= ball_num <= RED_ZONES['Zone2'][1]: zones_count[1]+=1
        elif RED_ZONES['Zone3'][0] <= ball_num <= RED_ZONES['Zone3'][1]: zones_count[2]+=1
    props['zone_dist'] = tuple(zones_count)
    return props


def generate_combinations(scores_data: dict, pattern_analysis_data: dict, association_rules_df: pd.DataFrame,
                          winning_segment_percentages: Dict[str, float], weights_config: Dict) -> tuple[List[Dict], list[str]]:
    """根据分数、模式和关联规则生成推荐组合。"""
    num_combinations_to_generate = weights_config.get('NUM_COMBINATIONS_TO_GENERATE', 10)
    target_red_pool_size = weights_config.get('TOP_N_RED_FOR_CANDIDATE', 18)
    top_n_blue = weights_config.get('TOP_N_BLUE_FOR_CANDIDATE', 8)

    # 多样性参数
    min_different_reds = weights_config.get('DIVERSITY_MIN_DIFFERENT_REDS', 3)
    max_common_reds_allowed = 6 - min_different_reds # 允许的最大相同红球数
    diversity_max_attempts = weights_config.get('DIVERSITY_SELECTION_MAX_ATTEMPTS', 20)
    
    diversity_sum_diff_thresh = weights_config.get('DIVERSITY_SUM_DIFF_THRESHOLD', 15) # 从权重配置中获取新的多样性阈值
    diversity_oddeven_diff_min = weights_config.get('DIVERSITY_ODDEVEN_DIFF_MIN_COUNT', 1)
    diversity_zone_dist_min_diff_zones = weights_config.get('DIVERSITY_ZONE_DIST_MIN_DIFF_ZONES', 2)

    # 候选池分段比例
    prop_h = weights_config.get('CANDIDATE_POOL_PROPORTIONS_HIGH', 0.5)
    prop_m = weights_config.get('CANDIDATE_POOL_PROPORTIONS_MEDIUM', 0.3)
    prop_l = max(0, 1.0 - prop_h - prop_m) # 确保 prop_l 不为负
    segment_proportions = {'High': prop_h, 'Medium': prop_m, 'Low': prop_l}
    min_per_segment = weights_config.get('CANDIDATE_POOL_MIN_PER_SEGMENT', 2) # 每个分段最少选几个

    r_scores = scores_data.get('red_scores', {})
    b_scores = scores_data.get('blue_scores', {})

    # --- 改进的红球分段候选池选择 ---
    r_cand_pool = []
    if r_scores:
        segmented_balls_dict = {name: [] for name in CANDIDATE_POOL_SEGMENT_NAMES}
        # 根据分数将红球分到高、中、低段
        for ball_num, score_val in r_scores.items():
            if score_val > CANDIDATE_POOL_SCORE_THRESHOLDS['High']:
                segmented_balls_dict['High'].append(ball_num)
            elif score_val > CANDIDATE_POOL_SCORE_THRESHOLDS['Medium']:
                segmented_balls_dict['Medium'].append(ball_num)
            else:
                segmented_balls_dict['Low'].append(ball_num)

        # 对每个分段内的球按分数排序
        for seg_name in CANDIDATE_POOL_SEGMENT_NAMES:
            segment_balls_with_scores = {b: r_scores.get(b, 0) for b in segmented_balls_dict[seg_name]}
            segmented_balls_dict[seg_name] = [b for b, _ in sorted(segment_balls_with_scores.items(), key=lambda x: x[1], reverse=True)]

        temp_pool_set = set() # 临时存储已选入池中的球，避免重复
        
        # 如果有历史中奖号码的分数段百分比数据，用它来调整各分段的选取比例
        win_seg_pcts_norm = {}
        if winning_segment_percentages and sum(winning_segment_percentages.values()) > 1e-6:
            total_pct = sum(winning_segment_percentages.values())
            win_seg_pcts_norm = {seg: pct/total_pct for seg, pct in winning_segment_percentages.items()}

        num_to_pick_segments = {}
        base_proportions_for_pool = segment_proportions # 如果没有获胜统计数据，则使用默认比例
        
        # 使用获胜号码的区间百分比来调整球池选择的比例
        # 将分数段标签 (例如 "0-25") 映射到球池分段名称 ('Low', 'Medium', 'High')
        # 此映射需要稳健。假设 SCORE_SEGMENT_LABELS 是有序的。
        if win_seg_pcts_norm:
            adjusted_proportions = {}
            # 简化映射：低分段 -> 低分区，高分段 -> 高分区，其他 -> 中分区
            # 需要仔细考虑 SCORE_SEGMENT_LABELS如何映射到CANDIDATE_POOL_SEGMENT_NAMES
            try:
                if len(SCORE_SEGMENT_LABELS) >= 3: # 假设至少有低、中、高类似的分数段
                    adjusted_proportions['Low'] = win_seg_pcts_norm.get(SCORE_SEGMENT_LABELS[0], segment_proportions['Low'])
                    adjusted_proportions['High'] = win_seg_pcts_norm.get(SCORE_SEGMENT_LABELS[-1], segment_proportions['High'])
                    # 中分区可以是中间分段的总和或默认值
                    medium_sum_pct = sum(win_seg_pcts_norm.get(lbl, 0) for lbl in SCORE_SEGMENT_LABELS[1:-1])
                    adjusted_proportions['Medium'] = medium_sum_pct if medium_sum_pct > 0 else segment_proportions['Medium']
                    
                    # 归一化调整后的比例，使其总和为1
                    total_adj_prop = sum(adjusted_proportions.values())
                    if total_adj_prop > 1e-6:
                        base_proportions_for_pool = {k: v/total_adj_prop for k,v in adjusted_proportions.items()}
                    else: # 如果win_seg_pcts_norm导致全为零，则回退
                        base_proportions_for_pool = segment_proportions
            except IndexError: # 如果SCORE_SEGMENT_LABELS与预期不符，则回退
                 base_proportions_for_pool = segment_proportions

        # 计算每个分段要选取的球数
        for i, seg_name in enumerate(CANDIDATE_POOL_SEGMENT_NAMES):
            prop = base_proportions_for_pool.get(seg_name, segment_proportions[seg_name]) # 如果可用，则使用调整后的比例，否则使用默认值
            num_to_pick_segments[seg_name] = max(min_per_segment, int(round(prop * target_red_pool_size)))
        
        # 确保总数不会因为min_per_segment而显著超过target_red_pool_size (允许一些灵活性)
        current_total_pick = sum(num_to_pick_segments.values())
        if current_total_pick > target_red_pool_size * 1.2:
            scale_down = (target_red_pool_size / current_total_pick) # 如果总和过高，则进行简单归一化
            for seg_name in num_to_pick_segments:
                num_to_pick_segments[seg_name] = max(min_per_segment, int(round(num_to_pick_segments[seg_name] * scale_down)))

        # 从各分段选取球加入候选池
        for seg_name in CANDIDATE_POOL_SEGMENT_NAMES: # 按高、中、低顺序迭代
            balls_from_segment = segmented_balls_dict[seg_name]
            num_to_add = num_to_pick_segments.get(seg_name, min_per_segment)
            added_count = 0
            for ball in balls_from_segment:
                if len(temp_pool_set) >= target_red_pool_size: break # 池已满
                if ball not in temp_pool_set and added_count < num_to_add:
                    temp_pool_set.add(ball)
                    added_count += 1
            if len(temp_pool_set) >= target_red_pool_size: break
        
        r_cand_pool = list(temp_pool_set)

        # 如果球池仍小于目标大小，则用总体最高分填充
        if len(r_cand_pool) < target_red_pool_size:
            all_sorted_reds_overall = [n for n, _ in sorted(r_scores.items(), key=lambda item: item[1], reverse=True)]
            for ball in all_sorted_reds_overall:
                if len(r_cand_pool) >= target_red_pool_size: break
                if ball not in r_cand_pool: # 如果不存在则添加
                    r_cand_pool.append(ball)
    
    # 确保球池至少有6个球用于抽样
    if len(r_cand_pool) < 6:
        logger.debug(f"红球候选池只有 {len(r_cand_pool)} 个球。扩展到至少6个。")
        current_pool_set = set(r_cand_pool) # 使用集合以提高查找效率
        # 如果r_scores可用，则获取按分数排序的所有红球
        all_sorted_reds_overall = [n for n, _ in sorted(r_scores.items(), key=lambda item: item[1], reverse=True)] if r_scores else list(RED_BALL_RANGE)
        
        for ball in all_sorted_reds_overall:
            if len(r_cand_pool) >= 6: break 
            if ball not in current_pool_set:
                r_cand_pool.append(ball)
                current_pool_set.add(ball)
        # 如果仍然不够（例如r_scores为空），则回退
        if len(r_cand_pool) < 6:
            remaining_needed = 6 - len(r_cand_pool)
            fallback_balls = [b for b in RED_BALL_RANGE if b not in current_pool_set]
            r_cand_pool.extend(random.sample(fallback_balls, min(remaining_needed, len(fallback_balls))))
            # 如果仍然不够（使用RED_BALL_RANGE不应发生），则从范围中取前6个
            if len(r_cand_pool) < 6:
                 r_cand_pool = list(set(r_cand_pool + list(RED_BALL_RANGE)))[:6]

    # 蓝球候选池
    b_cand_pool = [n for n, _ in sorted(b_scores.items(), key=lambda item: item[1], reverse=True)[:top_n_blue]] if b_scores else list(BLUE_BALL_RANGE)[:top_n_blue]
    if len(b_cand_pool) < 1: b_cand_pool = list(BLUE_BALL_RANGE) # 蓝球球池的回退方案
    
    # --- 概率抽样设置 ---
    large_pool_size = max(num_combinations_to_generate * 100, 200) # 生成一个较大的初始组合池
    max_attempts_pool = large_pool_size * 20 # 生成初始池的最大尝试次数

    # 使用历史中奖号码分数段分布调整抽样概率
    win_seg_pcts = winning_segment_percentages
    valid_seg_pcts = win_seg_pcts and all(lbl in win_seg_pcts for lbl in SCORE_SEGMENT_LABELS) and sum(win_seg_pcts.values()) > 1e-6
    seg_factors = {lbl:1.0 for lbl in SCORE_SEGMENT_LABELS} # 默认因子为1
    if valid_seg_pcts:
        seg_factors_temp = {lbl:(pct/100.0)+0.05 for lbl,pct in win_seg_pcts.items()} # 添加一个小的基数以避免因子为零
        tot_factor_sum = sum(seg_factors_temp.values())
        if tot_factor_sum > 1e-9: seg_factors = {lbl:f_val/tot_factor_sum for lbl,f_val in seg_factors_temp.items()} # 归一化

    r_probs_raw = {} # 红球原始抽样概率
    if not r_cand_pool: # 极端情况处理
        logger.error("严重: r_cand_pool在概率计算前为空。将使用默认范围。")
        r_cand_pool = random.sample(list(RED_BALL_RANGE), k=min(target_red_pool_size, len(RED_BALL_RANGE)))

    for n in r_cand_pool: # 使用精炼的r_cand_pool
        seg = get_score_segment(r_scores.get(n,0), SCORE_SEGMENT_BOUNDARIES, SCORE_SEGMENT_LABELS)
        # 按分数和分段因子（分数段的历史表现）加权
        r_probs_raw[n] = (r_scores.get(n,0)+1.0) * seg_factors.get(seg, 1.0/len(seg_factors) if seg_factors else 1.0)

    # 确保r_cand_pool_for_probs包含具有正概率且在r_cand_pool中的球
    r_cand_pool_for_probs = [ball for ball in r_cand_pool if ball in r_probs_raw and r_probs_raw[ball] > 0] 
    
    if not r_cand_pool_for_probs: # 如果没有球有正概率
        logger.debug("没有红球具有正的原始概率。将从r_cand_pool或范围中均匀抽样。")
        # 如果r_cand_pool已填充则回退到它，否则从完整范围随机抽样
        r_cand_pool_for_probs = r_cand_pool if len(r_cand_pool) >= 6 else random.sample(list(RED_BALL_RANGE), k=6)
        r_probs_arr = np.ones(len(r_cand_pool_for_probs)) / len(r_cand_pool_for_probs) if r_cand_pool_for_probs else np.array([])
    else:
        r_probs_arr = np.array([r_probs_raw.get(n,0) for n in r_cand_pool_for_probs])
        tot_r_prob_raw = np.sum(r_probs_arr) 
        if tot_r_prob_raw > 1e-9 :
            r_probs_arr = r_probs_arr / tot_r_prob_raw # 归一化概率
            if len(r_probs_arr) > 1 : r_probs_arr[-1]=max(0,1.0-np.sum(r_probs_arr[:-1])) # 确保归一化后概率总和为1
            elif len(r_probs_arr) == 1: r_probs_arr[0] = 1.0
        else: # 如果概率总和过低则回退
            r_probs_arr = np.ones(len(r_cand_pool_for_probs))/len(r_cand_pool_for_probs)

    # 蓝球抽样概率
    b_weights_arr = np.array([b_scores.get(n,0)+1.0 for n in b_cand_pool]) # 如果分数为0，则加1以避免权重为零
    b_probs_arr = np.zeros(len(b_cand_pool)) 
    if not b_cand_pool: # 应该由上面的回退处理，但作为安全措施
        b_cand_pool = random.sample(list(BLUE_BALL_RANGE), k=min(top_n_blue, len(BLUE_BALL_RANGE)))
        if not b_cand_pool: b_cand_pool = [1] # 绝对回退方案
        b_weights_arr = np.array([b_scores.get(n,0)+1.0 for n in b_cand_pool])

    if np.sum(b_weights_arr) > 1e-9 and len(b_cand_pool) > 0:
        b_probs_arr = b_weights_arr / np.sum(b_weights_arr) # 归一化
        if len(b_probs_arr) > 1: b_probs_arr[-1] = max(0, 1.0 - np.sum(b_probs_arr[:-1])) # 确保总和为1
        elif len(b_probs_arr) == 1: b_probs_arr[0] = 1.0
    elif len(b_cand_pool) > 0: b_probs_arr = np.ones(len(b_cand_pool)) / len(b_cand_pool) # 如果权重为零则均匀分布
    else: # 蓝球概率的绝对回退方案
        b_probs_arr = np.array([1.0]); b_cand_pool = [1]

    sample_size_red = 6 # 每次抽取6个红球
    replace_red_sampling = False # 对于彩票类型的选择通常为False (不放回)
    use_fallback_sampling_flag = False # 如果概率有问题，则切换到随机抽样的标志

    # 检查回退抽样的条件
    if len(r_cand_pool_for_probs) < sample_size_red:
        logger.debug(f"用于概率抽样的红球候选池 ({len(r_cand_pool_for_probs)}) 小于抽样大小 ({sample_size_red})。")
        if len(r_cand_pool_for_probs) == 0: 
             use_fallback_sampling_flag = True # 如果球池为空则无法抽样
        else: # 如果少于6个，则抽样所有可用的
            sample_size_red = len(r_cand_pool_for_probs) 
            replace_red_sampling = False # 仍然不放回
    
    # 进一步检查概率数组的有效性
    if not use_fallback_sampling_flag:
        is_r_probs_invalid = (len(r_probs_arr) != len(r_cand_pool_for_probs) or \
                             not (np.isclose(np.sum(r_probs_arr),1.0) if len(r_probs_arr)>0 else True) or \
                             (len(r_probs_arr) > 0 and np.any(r_probs_arr < 0)))
        is_b_probs_invalid = (len(b_probs_arr) != len(b_cand_pool) or \
                             not (np.isclose(np.sum(b_probs_arr),1.0) if len(b_probs_arr)>0 else True) or \
                             (len(b_probs_arr) > 0 and np.any(b_probs_arr < 0)))
        is_b_pool_invalid = not (len(b_cand_pool)>=1)

        if is_r_probs_invalid or is_b_probs_invalid or is_b_pool_invalid:
            use_fallback_sampling_flag = True
            logger.debug(f"由于概率/球池无效，切换到回退抽样: r_invalid={is_r_probs_invalid}, b_invalid={is_b_probs_invalid}, b_pool_invalid={is_b_pool_invalid}")
            
    if use_fallback_sampling_flag: logger.debug("在generate_combinations中使用回退（随机）抽样。")

    # 生成初始组合池
    gen_pool = []
    attempts = 0
    while len(gen_pool) < large_pool_size and attempts < max_attempts_pool:
        attempts +=1
        try:
            if use_fallback_sampling_flag:
                # 回退：从最初构建的r_cand_pool和b_cand_pool中随机抽样
                safe_r_pool = r_cand_pool if len(r_cand_pool) >= 6 else list(RED_BALL_RANGE)
                r_balls_s = sorted(random.sample(safe_r_pool, 6))
                
                safe_b_pool = b_cand_pool if len(b_cand_pool) >=1 else list(BLUE_BALL_RANGE)
                b_ball_s = random.choice(safe_b_pool)
            else: # 概率抽样
                r_balls_s = sorted(np.random.choice(r_cand_pool_for_probs, size=sample_size_red, replace=replace_red_sampling, p=r_probs_arr).tolist())
                # 如果由于球池小而sample_size_red小于6，则随机填充剩余部分
                if len(r_balls_s) < 6:
                    remaining_needed = 6 - len(r_balls_s)
                    # 填充候选：来自r_cand_pool，排除已选中的
                    fill_pool_candidates = [b for b in r_cand_pool if b not in r_balls_s] 
                    # 如果r_cand_pool中不够，则扩展到完整范围
                    if len(fill_pool_candidates) < remaining_needed: 
                        fill_pool_candidates.extend([b for b in RED_BALL_RANGE if b not in r_balls_s and b not in fill_pool_candidates])
                    
                    if len(fill_pool_candidates) >= remaining_needed:
                         r_balls_s.extend(random.sample(fill_pool_candidates, remaining_needed))
                         r_balls_s = sorted(list(set(r_balls_s))) # 确保唯一且排序
                    else: # 如果r_cand_pool最初大于6或使用RED_BALL_RANGE，则应该很少见
                        logger.debug(f"无法形成6个红球组合，得到 {len(r_balls_s)}。跳过尝试。")
                        continue 
                
                if len(r_balls_s) != 6 : # 潜在填充后的最终检查
                    logger.debug(f"填充后红球数量不匹配，跳过组合: {len(r_balls_s)}")
                    continue

                b_ball_s = np.random.choice(b_cand_pool, size=1, p=b_probs_arr).tolist()[0]

            combo = {'red': r_balls_s, 'blue': b_ball_s}
            # 验证生成的组合
            if len(set(combo['red'])) == 6 and combo['blue'] is not None and 1 <= combo['blue'] <= 16:
                 if combo not in gen_pool: gen_pool.append(combo) # 如果唯一则添加
            else:
                 logger.debug(f"生成并跳过无效组合: {combo}")

        except ValueError as e_val: # 捕获np.random.choice的错误（例如，概率总和不为1）
            use_fallback_sampling_flag = True # 后续尝试切换到回退方案
            if attempts <= 5: logger.debug(f"generate_combinations中的概率抽样失败 ({e_val})，此次运行切换到回退方案。")
        except Exception as e_gen: # 捕获任何其他生成错误
            logger.debug(f"组合生成尝试期间发生异常: {e_gen}")
            continue # 跳过此尝试
            
    if not gen_pool: return [], ["推荐组合:", "无法生成推荐组合。"]
    
    # --- 对生成的组合进行评分和应用ARM奖励 ---
    scored_combos = []
    patt_data = pattern_analysis_data # 历史模式数据
    hist_odd_cnt = patt_data.get('most_common_odd_even_count') # 最常见奇偶比
    hist_zone_dist = patt_data.get('most_common_zone_distribution') # 最常见区间分布
    blue_l_counts = patt_data.get('blue_large_counts',{}) # 蓝球大小分布
    hist_blue_large = blue_l_counts.get(True,0) > blue_l_counts.get(False,0) if blue_l_counts else None
    blue_o_counts = patt_data.get('blue_odd_counts',{}) # 蓝球奇偶分布
    hist_blue_odd = blue_o_counts.get(True,0) > blue_o_counts.get(False,0) if blue_o_counts else None

    arm_rules_processed = pd.DataFrame() # 默认为空DataFrame
    if association_rules_df is not None and not association_rules_df.empty:
        arm_rules_processed = association_rules_df.copy() 
        if not arm_rules_processed.empty:
            try: # 确保'antecedents'和'consequents'是整数的frozenset以进行比较
                arm_rules_processed['antecedents_set'] = arm_rules_processed['antecedents'].apply(lambda x: set(map(int, x)))
                arm_rules_processed['consequents_set'] = arm_rules_processed['consequents'].apply(lambda x: set(map(int, x)))
            except (TypeError, ValueError) as e_arm_conv:
                logger.warning(f"转换ARM规则项为整数集合时出错 ({e_arm_conv})。ARM奖励可能无法正确应用。")
                arm_rules_processed = pd.DataFrame() # 如果转换失败则无效

    for combo_item in gen_pool:
        r_list, b_val = combo_item['red'], combo_item['blue']
        base_s = sum(r_scores.get(ball_num,0) for ball_num in r_list) + b_scores.get(b_val,0) # 基础分
        bonus_s = 0 # 奖励分
        # 模式匹配奖励
        if hist_odd_cnt is not None and sum(x%2!=0 for x in r_list)==hist_odd_cnt: bonus_s += weights_config['COMBINATION_ODD_COUNT_MATCH_BONUS']
        if hist_blue_odd is not None and (b_val%2!=0)==hist_blue_odd: bonus_s += weights_config['COMBINATION_BLUE_ODD_MATCH_BONUS']
        
        combo_props_current = get_combo_properties(r_list) # 计算一次属性
        if hist_zone_dist and combo_props_current['zone_dist'] == hist_zone_dist:
            bonus_s += weights_config['COMBINATION_ZONE_MATCH_BONUS']

        if hist_blue_large is not None and (b_val>8)==hist_blue_large: bonus_s += weights_config['COMBINATION_BLUE_SIZE_MATCH_BONUS']

        # ARM奖励
        arm_specific_bonus = 0
        combo_red_set = set(r_list)
        if not arm_rules_processed.empty and 'antecedents_set' in arm_rules_processed.columns: # 检查集合是否已创建
            for _, rule in arm_rules_processed.iterrows():
                # 确保规则项是集合以进行issubset操作
                if isinstance(rule.get('antecedents_set'), set) and isinstance(rule.get('consequents_set'), set):
                    if rule['antecedents_set'].issubset(combo_red_set) and rule['consequents_set'].issubset(combo_red_set):
                        lift_bonus = (rule.get('lift', 1.0) - 1.0) * weights_config.get('ARM_BONUS_LIFT_FACTOR', 0.2) 
                        conf_bonus = rule.get('confidence', 0.0) * weights_config.get('ARM_BONUS_CONF_FACTOR', 0.1)
                        current_rule_bonus = (lift_bonus + conf_bonus) * weights_config['ARM_COMBINATION_BONUS_WEIGHT']
                        arm_specific_bonus += current_rule_bonus
            # 限制ARM奖励以避免许多小规则的过度影响 (示例上限)
            arm_specific_bonus = min(arm_specific_bonus, weights_config['ARM_COMBINATION_BONUS_WEIGHT'] * 2.0)

        bonus_s += arm_specific_bonus
        scored_combos.append({
            'combination': combo_item, 
            'score': base_s + bonus_s,
            'red_tuple': tuple(sorted(r_list)), # 用于多样性检查和唯一性
            'properties': combo_props_current # 存储计算出的属性
        })
        
    final_recs_data = []
    if not scored_combos: 
        return [], ["推荐组合:", "无法生成推荐组合 (评分后为空)。"]

    sorted_scored_combos = sorted(scored_combos, key=lambda x: x['score'], reverse=True) # 按总分排序

    # --- 增强的多样性选择 ---
    if sorted_scored_combos:
        first_combo = sorted_scored_combos.pop(0) # 第一个总是选最高分的
        final_recs_data.append(first_combo)
    
    attempts_for_diversity = 0
    candidate_idx = 0
    while len(final_recs_data) < num_combinations_to_generate and candidate_idx < len(sorted_scored_combos):
        if attempts_for_diversity > diversity_max_attempts * num_combinations_to_generate : 
            logger.debug(f"多样性选择达到最大尝试次数，当前已选 {len(final_recs_data)} 组合。")
            break 
            
        candidate_combo_dict = sorted_scored_combos[candidate_idx]
        candidate_red_set = set(candidate_combo_dict['red_tuple'])
        candidate_props = candidate_combo_dict['properties']
        is_diverse_enough = True # 假设该组合足够多样
        
        for existing_rec_dict in final_recs_data: # 与已选组合比较
            existing_red_set = set(existing_rec_dict['red_tuple'])
            existing_props = existing_rec_dict['properties']
            
            # 1. 基本的红球差异
            common_reds = len(candidate_red_set.intersection(existing_red_set))
            if common_reds > max_common_reds_allowed:
                is_diverse_enough = False; break
            
            # 2. 和值差异
            if abs(candidate_props['sum'] - existing_props['sum']) < diversity_sum_diff_thresh:
                is_diverse_enough = False; break
                
            # 3. 奇偶数差异
            if abs(candidate_props['odd_count'] - existing_props['odd_count']) < diversity_oddeven_diff_min:
                is_diverse_enough = False; break

            # 4. 区间分布差异
            diff_zones_count = sum(1 for i in range(len(candidate_props['zone_dist'])) if candidate_props['zone_dist'][i] != existing_props['zone_dist'][i])
            if diff_zones_count < diversity_zone_dist_min_diff_zones:
                is_diverse_enough = False; break
        
        if is_diverse_enough:
            final_recs_data.append(candidate_combo_dict) # 如果足够多样，则添加
        
        candidate_idx += 1
        attempts_for_diversity +=1

    # 如果多样性选择未达到num_combinations_to_generate，则填充剩余空位
    if len(final_recs_data) < num_combinations_to_generate:
        logger.debug(f"多样性选择后组合数不足 ({len(final_recs_data)})，将从剩余高分组合中补充。")
        # 为加快查找速度，创建已包含组合字典的集合
        final_recs_tuples_set = {rec['red_tuple'] for rec in final_recs_data} # 用于red_tuple检查
        
        needed_more = num_combinations_to_generate - len(final_recs_data)
        added_count = 0
        # 迭代所有最初排序的带分数组合
        for combo_dict in sorted(scored_combos, key=lambda x: x['score'], reverse=True): 
            if added_count >= needed_more:
                break
            # 检查这个确切的红球组合（元组）是否已存在
            if combo_dict['red_tuple'] not in final_recs_tuples_set:
                 # 如果蓝球在此处对唯一性很重要，还要确保完整组合（红球+蓝球）不是重复的
                 is_already_present_full = any(fc['combination'] == combo_dict['combination'] for fc in final_recs_data)
                 if not is_already_present_full:
                    final_recs_data.append(combo_dict)
                    final_recs_tuples_set.add(combo_dict['red_tuple']) # 将其red_tuple添加到集合中
                    added_count += 1
    
    # 最终排序和修剪
    final_recs_data = sorted(final_recs_data, key=lambda x: x['score'], reverse=True)[:num_combinations_to_generate]

    output_strs = [f"  组合 {i+1}: 红球 {sorted(rec['combination']['red'])} 蓝球 {rec['combination']['blue']} (综合分: {rec['score']:.2f})"
                   for i,rec in enumerate(final_recs_data)] if final_recs_data else ["  无法生成推荐组合。"]
    return final_recs_data, ["推荐组合 (Top {}):".format(len(final_recs_data))] + output_strs


def analyze_and_recommend(
    df_historical: pd.DataFrame, ml_lags_list: List[int], weights_config: Dict,
    association_rules_df_main: pd.DataFrame, # 直接传递ARM规则
    train_ml: bool = True, existing_models: Optional[Dict] = None
) -> tuple[List[Dict], list[str], dict, Optional[Dict], Dict, Dict[str, float]]:
    """执行完整的分析和推荐流程。"""
    recs_data, recs_strs = [], []
    analysis_res = {}; current_models = None; scores_res = {}; win_seg_pcts = {}

    if df_historical is None or df_historical.empty:
        return recs_data, recs_strs, analysis_res, current_models, scores_res, win_seg_pcts # 返回具有正确类型的空结构

    # 基础分析
    freq_om_data = analyze_frequency_omission(df_historical, weights_config)
    patt_an_data = analyze_patterns(df_historical, weights_config)
    # ARM规则现在已传入，无需在此特定调用中重新分析
    analysis_res = {'freq_omission': freq_om_data, 'patterns': patt_an_data, 'associations': association_rules_df_main}

    # 机器学习预测
    pred_probs = {}
    min_ml_periods = (max(ml_lags_list) if ml_lags_list else 0) + 1 + MIN_POSITIVE_SAMPLES_FOR_ML # 训练所需最小数据 + 滞后所需数据
    if len(df_historical) >= min_ml_periods:
        if train_ml: # 是否训练新模型
            logger.debug("为推荐训练ML模型...")
            current_models = train_prediction_models(df_historical, ml_lags_list, weights_config)
            if current_models:
                logger.debug("使用新训练的模型预测下一期开奖概率...")
                pred_probs = predict_next_draw_probabilities(df_historical, current_models, ml_lags_list, weights_config)
        elif existing_models: # 使用预训练模型
            current_models = existing_models
            if current_models:
                logger.debug("使用现有模型预测下一期开奖概率...")
                pred_probs = predict_next_draw_probabilities(df_historical, current_models, ml_lags_list, weights_config)
    else:
        logger.warning(f"历史数据不足 ({len(df_historical)}) 进行ML模型训练/预测 (最少: {min_ml_periods}期)。跳过ML部分。")

    # 评分和组合生成
    scores_res = calculate_scores(freq_om_data, patt_an_data, pred_probs, weights_config)
    _, win_seg_pcts = analyze_winning_red_ball_score_segments(
        df_historical, scores_res.get('red_scores',{}), SCORE_SEGMENT_BOUNDARIES, SCORE_SEGMENT_LABELS
    )
    recs_data, recs_strs = generate_combinations(
        scores_res, patt_an_data, association_rules_df_main, win_seg_pcts, weights_config # 传递ARM规则
    )
    return recs_data, recs_strs, analysis_res, current_models, scores_res, win_seg_pcts

def get_prize_level(red_hits: int, blue_hit: bool) -> Optional[str]:
    """根据红球命中数和蓝球命中情况确定奖级。"""
    if blue_hit:
        if red_hits == 6: return "一等奖" # 6+1
        if red_hits == 5: return "三等奖" # 5+1
        if red_hits == 4: return "四等奖" # 4+1
        if red_hits == 3: return "五等奖" # 3+1
        if red_hits <= 2: return "六等奖" # 2+1, 1+1, 0+1
    else: # 未命中蓝球
        if red_hits == 6: return "二等奖" # 6+0
        if red_hits == 5: return "四等奖" # 5+0
        if red_hits == 4: return "五等奖" # 4+0
    return None # 其他情况无奖

def backtest(df: pd.DataFrame, ml_lags_list: List[int], weights_config: Dict,
             arm_rules_for_backtest: pd.DataFrame, # 为清晰起见更改了名称
             backtest_periods_to_eval: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """执行回测以评估策略性能。"""
    max_hist_lag = max(ml_lags_list) if ml_lags_list else 0
    # 在进行第一次预测之前，初始训练数据所需的最少周期数 (滞后所需数据 + 训练所需数据)
    min_initial_train_periods = max_hist_lag + 1 + MIN_POSITIVE_SAMPLES_FOR_ML 
    
    if len(df) < min_initial_train_periods + 1: # +1 因为有一个周期用于实际结果
        logger.warning(f"回测: 数据不足({len(df)})，需要至少 {min_initial_train_periods + 1} 期 (训练+1预测)。")
        return pd.DataFrame(), {}

    # df中第一个要预测的实际结果所在的索引 (训练数据将是df.iloc[:first_prediction_target_idx])
    first_prediction_target_idx = min_initial_train_periods 
    # df中可以是实际结果的最后一个索引
    last_prediction_target_idx = len(df) - 1

    if first_prediction_target_idx > last_prediction_target_idx:
        logger.warning("回测: 无足够后续数据进行预测评估循环。")
        return pd.DataFrame(), {}

    # 确定用于预测循环的df索引的实际范围 (我们想要评估 backtest_periods_to_eval 个周期)
    loop_start_idx = max(first_prediction_target_idx, last_prediction_target_idx - backtest_periods_to_eval + 1)

    results_list = [] # 存储每期回测结果
    red_cols_list = [f'red{i+1}' for i in range(6)]
    is_opt_run_flag = backtest_periods_to_eval == OPTIMIZATION_BACKTEST_PERIODS # 用于Optuna运行的标志
    
    prize_counts = Counter() # 统计各奖级次数
    best_hit_per_period = [] # 存储每个测试周期的最大红球命中数和蓝球命中状态
    periods_with_any_blue_hit = set() # 存储命中蓝球的期号
    num_combinations_generated_per_run = weights_config.get('NUM_COMBINATIONS_TO_GENERATE', 10)

    # 循环遍历代表我们试图预测的实际结果的df索引
    for df_idx_for_prediction_target in range(loop_start_idx, last_prediction_target_idx + 1):
        
        # 非Optuna运行的进度记录
        if not is_opt_run_flag and (df_idx_for_prediction_target - loop_start_idx + 1) % 10 == 0 :
            current_console_level = global_console_handler.level
            current_console_formatter = global_console_handler.formatter
            set_console_verbosity(logging.INFO, use_simple_formatter=True)
            logger.info(f"  回测进度: {df_idx_for_prediction_target - loop_start_idx + 1} / {last_prediction_target_idx - loop_start_idx + 1}")
            global_console_handler.setLevel(current_console_level)
            global_console_handler.setFormatter(current_console_formatter)

        # 截至（但不包括）当前目标周期的数据用于训练
        current_train_data = df.iloc[:df_idx_for_prediction_target].copy() 
        if len(current_train_data) < min_initial_train_periods: # 使用loop_start_idx逻辑不应发生
            logger.warning(f"跳过周期 {df.loc[df_idx_for_prediction_target, '期号']}，因训练数据不足 ({len(current_train_data)})。")
            continue

        actual_outcome_row = df.loc[df_idx_for_prediction_target] # 当期实际开奖结果
        current_period_actual_id = actual_outcome_row['期号']
        try: # 获取并验证实际开奖号码
            actual_red_set = set(actual_outcome_row[red_cols_list].astype(int).tolist())
            actual_blue_val = int(actual_outcome_row['blue'])
            if not (all(1<=r_val<=33 for r_val in actual_red_set) and 1<=actual_blue_val<=16 and len(actual_red_set)==6):
                raise ValueError("实际球号超出范围或数量不正确")
        except Exception as e_actual:
            logger.debug(f"回测: 获取期号 {current_period_actual_id} 实际结果失败: {e_actual}")
            logger.debug(f"问题行内容: {actual_outcome_row.to_dict()}") # 记录问题行的更多详细信息以进行调试
            # 即使结果解析失败也添加条目，以保持周期计数一致
            best_hit_per_period.append({
                'period': current_period_actual_id, 'max_red_hits': -1, 'blue_hit_in_period': False, 'error': str(e_actual)
            })
            continue # 如果实际结果无效则跳过此周期

        # 在Optuna运行期间抑制详细日志记录以提高速度
        original_logger_level = logger.level
        original_console_level = global_console_handler.level
        
        if is_opt_run_flag: 
            logger.setLevel(logging.CRITICAL) 
            set_console_verbosity(logging.CRITICAL)

        # 对于Optuna运行，arm_rules_for_backtest是特定于试验的。
        # 对于常规回测，它们基于使用当前/最终权重的完整历史。
        # 此处无需重新计算ARM规则，因为它们已传入。
        
        if is_opt_run_flag: # Optuna运行时抑制输出
            with SuppressOutput(suppress_stdout=True, capture_stderr=True):
                 predicted_combos_list, _, _, _, _, _ = analyze_and_recommend(
                     current_train_data, ml_lags_list, weights_config, arm_rules_for_backtest, train_ml=True)
        else: # 常规回测
            predicted_combos_list, _, _, _, _, _ = analyze_and_recommend(
                current_train_data, ml_lags_list, weights_config, arm_rules_for_backtest, train_ml=True)

        if is_opt_run_flag: # 恢复日志级别
            logger.setLevel(original_logger_level)
            set_console_verbosity(original_console_level)

        period_max_red_hits = 0 # 当期推荐组合中的最大红球命中数
        period_blue_hit_achieved_this_draw = False # 在此次抽奖中是否有任何组合命中蓝球？

        if predicted_combos_list: # 如果有推荐组合
            for combo_dict_info in predicted_combos_list: # 评估每个推荐组合
                pred_r_set = set(combo_dict_info['combination']['red'])
                pred_b_val = combo_dict_info['combination']['blue']
                red_h = len(pred_r_set.intersection(actual_red_set)) # 红球命中数
                blue_h = (pred_b_val == actual_blue_val) # 蓝球是否命中

                results_list.append({
                    'period': current_period_actual_id,
                    'predicted_red': sorted(list(pred_r_set)), 'predicted_blue': pred_b_val,
                    'actual_red': sorted(list(actual_red_set)), 'actual_blue': actual_blue_val,
                    'red_hits': red_h,
                    'blue_hit': blue_h,
                    'combination_score': combo_dict_info.get('score', 0.0) # 确保分数存在
                })
                
                prize = get_prize_level(red_h, blue_h) # 获取奖级
                if prize:
                    prize_counts[prize] += 1
                
                if blue_h:
                    periods_with_any_blue_hit.add(current_period_actual_id) # 添加期号ID
                    period_blue_hit_achieved_this_draw = True
                if red_h > period_max_red_hits:
                    period_max_red_hits = red_h
            
            best_hit_per_period.append({ # 记录当期最佳表现
                'period': current_period_actual_id,
                'max_red_hits': period_max_red_hits,
                'blue_hit_in_period': period_blue_hit_achieved_this_draw 
            })
        else: # 未预测任何组合
            best_hit_per_period.append({
                'period': current_period_actual_id, 'max_red_hits': 0, 'blue_hit_in_period': False, 'error': 'No combos predicted'
            })
            logger.debug(f"回测: 期号 {current_period_actual_id} 未预测任何组合。")


    if not results_list: return pd.DataFrame(), {} # 如果没有有效结果则返回空
    
    results_df_final = pd.DataFrame(results_list)
    # 将周期范围存储在DataFrame属性中，以便以后轻松访问
    if '期号' in df.columns and loop_start_idx < len(df) and last_prediction_target_idx < len(df):
        try:
            results_df_final.attrs['start_period_id'] = df.loc[loop_start_idx, '期号']
            results_df_final.attrs['end_period_id'] = df.loc[last_prediction_target_idx, '期号']
            results_df_final.attrs['num_periods_tested'] = last_prediction_target_idx - loop_start_idx + 1
        except KeyError: # 如果索引有效则不应发生
             logger.warning("回测: 由于索引问题，无法设置开始/结束周期属性。")
    
    # 为Optuna和报告添加扩展统计信息
    extended_stats = {
        'prize_counts': dict(prize_counts), # 如果需要，将Counter转换为字典以便于JSON序列化
        'best_hit_per_period_df': pd.DataFrame(best_hit_per_period) if best_hit_per_period else pd.DataFrame(),
        'total_combinations_evaluated': len(results_df_final),
        'num_combinations_per_draw_tested': num_combinations_generated_per_run,
        'periods_with_any_blue_hit_count': len(periods_with_any_blue_hit) # 命中蓝球的独立期数
    }
    return results_df_final, extended_stats


def objective(trial: optuna.trial.Trial, df_for_optimization: pd.DataFrame, fixed_ml_lags: List[int]) -> float:
    """Optuna的目标函数，用于优化权重。"""
    
    # 为各种组件建议权重
    weights_to_eval = {
        'NUM_COMBINATIONS_TO_GENERATE': trial.suggest_int('NUM_COMBINATIONS_TO_GENERATE', 5, 12),
        'TOP_N_RED_FOR_CANDIDATE': trial.suggest_int('TOP_N_RED_FOR_CANDIDATE', 15, 28), 
        'TOP_N_BLUE_FOR_CANDIDATE': trial.suggest_int('TOP_N_BLUE_FOR_CANDIDATE', 6, 12),

        'FREQ_SCORE_WEIGHT': trial.suggest_float('FREQ_SCORE_WEIGHT', 5, 30),
        'OMISSION_SCORE_WEIGHT': trial.suggest_float('OMISSION_SCORE_WEIGHT', 5, 25),
        'MAX_OMISSION_RATIO_SCORE_WEIGHT_RED': trial.suggest_float('MAX_OMISSION_RATIO_SCORE_WEIGHT_RED', 0, 20),
        'RECENT_FREQ_SCORE_WEIGHT_RED': trial.suggest_float('RECENT_FREQ_SCORE_WEIGHT_RED', 0, 20),
        'BLUE_FREQ_SCORE_WEIGHT': trial.suggest_float('BLUE_FREQ_SCORE_WEIGHT', 5, 30),
        'BLUE_OMISSION_SCORE_WEIGHT': trial.suggest_float('BLUE_OMISSION_SCORE_WEIGHT', 5, 25),
        'ML_PROB_SCORE_WEIGHT_RED': trial.suggest_float('ML_PROB_SCORE_WEIGHT_RED', 10, 50),
        'ML_PROB_SCORE_WEIGHT_BLUE': trial.suggest_float('ML_PROB_SCORE_WEIGHT_BLUE', 10, 50),

        'COMBINATION_ODD_COUNT_MATCH_BONUS': trial.suggest_float('COMBINATION_ODD_COUNT_MATCH_BONUS', 0, 20),
        'COMBINATION_BLUE_ODD_MATCH_BONUS': trial.suggest_float('COMBINATION_BLUE_ODD_MATCH_BONUS', 0, 12),
        'COMBINATION_ZONE_MATCH_BONUS': trial.suggest_float('COMBINATION_ZONE_MATCH_BONUS', 0, 18),
        'COMBINATION_BLUE_SIZE_MATCH_BONUS': trial.suggest_float('COMBINATION_BLUE_SIZE_MATCH_BONUS', 0, 12),

        'ARM_MIN_SUPPORT': trial.suggest_float('ARM_MIN_SUPPORT', 0.005, 0.025), 
        'ARM_MIN_CONFIDENCE': trial.suggest_float('ARM_MIN_CONFIDENCE', 0.20, 0.55), 
        'ARM_MIN_LIFT': trial.suggest_float('ARM_MIN_LIFT', 1.0, 1.8), 
        'ARM_COMBINATION_BONUS_WEIGHT': trial.suggest_float('ARM_COMBINATION_BONUS_WEIGHT', 0, 25), 
        'ARM_BONUS_LIFT_FACTOR': trial.suggest_float('ARM_BONUS_LIFT_FACTOR', 0.05, 0.5),
        'ARM_BONUS_CONF_FACTOR': trial.suggest_float('ARM_BONUS_CONF_FACTOR', 0.05, 0.5),

        'CANDIDATE_POOL_MIN_PER_SEGMENT': trial.suggest_int('CANDIDATE_POOL_MIN_PER_SEGMENT', 1, 5), 
        'CANDIDATE_POOL_PROPORTIONS_HIGH': trial.suggest_float('CANDIDATE_POOL_PROPORTIONS_HIGH', 0.1, 0.8), 
        'CANDIDATE_POOL_PROPORTIONS_MEDIUM': trial.suggest_float('CANDIDATE_POOL_PROPORTIONS_MEDIUM', 0.1, 0.8), 
        
        'DIVERSITY_MIN_DIFFERENT_REDS': trial.suggest_int('DIVERSITY_MIN_DIFFERENT_REDS', 2, 5), 
        'DIVERSITY_SELECTION_MAX_ATTEMPTS': trial.suggest_int('DIVERSITY_SELECTION_MAX_ATTEMPTS', 10, 50),
        'DIVERSITY_SUM_DIFF_THRESHOLD': trial.suggest_int('DIVERSITY_SUM_DIFF_THRESHOLD', 5, 25),
        'DIVERSITY_ODDEVEN_DIFF_MIN_COUNT': trial.suggest_int('DIVERSITY_ODDEVEN_DIFF_MIN_COUNT', 0, 2), 
        'DIVERSITY_ZONE_DIST_MIN_DIFF_ZONES': trial.suggest_int('DIVERSITY_ZONE_DIST_MIN_DIFF_ZONES', 1, 3),
        
        # Optuna目标函数本身的权重
        'OPTUNA_PRIZE_6_WEIGHT': trial.suggest_float('OPTUNA_PRIZE_6_WEIGHT', 0.0, 0.5),
        'OPTUNA_PRIZE_5_WEIGHT': trial.suggest_float('OPTUNA_PRIZE_5_WEIGHT', 0.1, 1.0),
        'OPTUNA_PRIZE_4_WEIGHT': trial.suggest_float('OPTUNA_PRIZE_4_WEIGHT', 0.5, 2.0),
        'OPTUNA_PRIZE_3_WEIGHT': trial.suggest_float('OPTUNA_PRIZE_3_WEIGHT', 1.0, 5.0),
        'OPTUNA_PRIZE_2_WEIGHT': trial.suggest_float('OPTUNA_PRIZE_2_WEIGHT', 2.0, 10.0),
        'OPTUNA_PRIZE_1_WEIGHT': trial.suggest_float('OPTUNA_PRIZE_1_WEIGHT', 5.0, 20.0),
        'OPTUNA_BLUE_HIT_RATE_WEIGHT': trial.suggest_float('OPTUNA_BLUE_HIT_RATE_WEIGHT', 5.0, 20.0),
        'OPTUNA_RED_HITS_WEIGHT': trial.suggest_float('OPTUNA_RED_HITS_WEIGHT', 0.5, 5.0),
    }
    
    # 确保候选池比例总和不超过1
    prop_h_trial = weights_to_eval['CANDIDATE_POOL_PROPORTIONS_HIGH']
    prop_m_trial = weights_to_eval['CANDIDATE_POOL_PROPORTIONS_MEDIUM']
    if prop_h_trial + prop_m_trial > 1.0:
        total_prop = prop_h_trial + prop_m_trial
        if total_prop > 1e-9 : 
             weights_to_eval['CANDIDATE_POOL_PROPORTIONS_HIGH'] = prop_h_trial / total_prop
             weights_to_eval['CANDIDATE_POOL_PROPORTIONS_MEDIUM'] = prop_m_trial / total_prop
        else: 
             weights_to_eval['CANDIDATE_POOL_PROPORTIONS_HIGH'] = 0.5 # 默认缩放
             weights_to_eval['CANDIDATE_POOL_PROPORTIONS_MEDIUM'] = 0.5

    # 使用试验的ARM参数为此特定试验重新计算ARM规则
    # 这使用完整的df_for_optimization数据集切片为此试验生成规则。
    logger.debug(f"Optuna试验 {trial.number}: 使用试验ARM参数分析关联规则: S:{weights_to_eval['ARM_MIN_SUPPORT']:.3f} C:{weights_to_eval['ARM_MIN_CONFIDENCE']:.2f} L:{weights_to_eval['ARM_MIN_LIFT']:.2f}")
    with SuppressOutput(suppress_stdout=True, capture_stderr=True): # 抑制mlxtend打印输出
        arm_rules_for_this_trial = analyze_associations(df_for_optimization.copy(), weights_to_eval)
    logger.debug(f"Optuna试验 {trial.number}: 找到 {len(arm_rules_for_this_trial)} 条ARM规则。")

    # 使用当前试验的权重进行回测
    backtest_results_df, extended_bt_stats = backtest(
        df_for_optimization.copy(), fixed_ml_lags, weights_to_eval,
        arm_rules_for_this_trial, 
        OPTIMIZATION_BACKTEST_PERIODS # 使用较短的回测期进行优化
    )

    if backtest_results_df.empty or len(backtest_results_df) == 0: # 如果回测无结果
        return float('inf') # 返回一个极差的值，表示此参数组合无效

    # 计算性能指标
    # 红球命中表现：使用命中数的1.5次方取平均，以更强调高命中数
    avg_weighted_red_hits = (backtest_results_df['red_hits'] ** 1.5).mean() * weights_to_eval['OPTUNA_RED_HITS_WEIGHT']

    # 蓝球命中表现：至少一个组合命中蓝球的期数占比
    num_periods_in_backtest = backtest_results_df['period'].nunique()
    if num_periods_in_backtest == 0:
        blue_hit_rate_score = 0.0
        blue_hit_rate_per_period = 0.0 # 为惩罚检查定义
    else:
        blue_hit_periods_count = extended_bt_stats.get('periods_with_any_blue_hit_count', 0)
        blue_hit_rate_per_period = blue_hit_periods_count / num_periods_in_backtest
        blue_hit_rate_score = blue_hit_rate_per_period * weights_to_eval['OPTUNA_BLUE_HIT_RATE_WEIGHT']
    
    # 奖金表现：根据不同奖级的权重计算总奖金分数
    prize_score = 0
    prizes_achieved = extended_bt_stats.get('prize_counts', {})
    prize_map_weights = {
        "一等奖": weights_to_eval['OPTUNA_PRIZE_1_WEIGHT'],
        "二等奖": weights_to_eval['OPTUNA_PRIZE_2_WEIGHT'],
        "三等奖": weights_to_eval['OPTUNA_PRIZE_3_WEIGHT'],
        "四等奖": weights_to_eval['OPTUNA_PRIZE_4_WEIGHT'],
        "五等奖": weights_to_eval['OPTUNA_PRIZE_5_WEIGHT'],
        "六等奖": weights_to_eval['OPTUNA_PRIZE_6_WEIGHT'],
    }
    for prize_level, count in prizes_achieved.items():
        prize_score += prize_map_weights.get(prize_level, 0) * count
    
    # 将奖金分数归一化到每个组合的平均奖金贡献
    total_combinations_in_backtest = extended_bt_stats.get('total_combinations_evaluated', 1)
    prize_score_rate = prize_score / total_combinations_in_backtest if total_combinations_in_backtest > 0 else 0

    # 综合性能分
    performance_score = avg_weighted_red_hits + blue_hit_rate_score + prize_score_rate

    # 对极低蓝球命中率和红球命中表现的惩罚
    if blue_hit_rate_per_period < 0.01 and (backtest_results_df['red_hits'] ** 1.5).mean() < 0.1: 
        performance_score -= 5.0 

    return -performance_score # Optuna默认最小化目标，故取负


if __name__ == "__main__":
    log_filename = os.path.join(SCRIPT_DIR, f"ssq_analysis_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    set_console_verbosity(logging.INFO, use_simple_formatter=True) # 初始控制台日志设为简洁模式

    logger.info(f"--- 双色球分析报告 ---")
    logger.info(f"运行日期: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"日志文件: {log_filename}")
    logger.info(f"红球候选池分数阈值: High > {CANDIDATE_POOL_SCORE_THRESHOLDS['High']}, Medium > {CANDIDATE_POOL_SCORE_THRESHOLDS['Medium']}")
    logger.info(f"ML 特征滞后阶数: {ML_LAG_FEATURES}")
    logger.info(f"ML 交互特征对: {ML_INTERACTION_PAIRS}, 自交互: {ML_INTERACTION_SELF}")
    logger.info("-" * 30)

    # 加载权重，如果文件不存在或无效，将触发后续的Optuna优化
    CURRENT_WEIGHTS, load_status = load_weights_from_file(WEIGHTS_CONFIG_FILE, DEFAULT_WEIGHTS)
    
    # 数据加载和预处理
    original_console_level_main = global_console_handler.level # 保存主程序开始时的控制台级别
    set_console_verbosity(logging.INFO, use_simple_formatter=False) # 加载数据时使用详细日志
    main_df = None
    if os.path.exists(PROCESSED_CSV_PATH): # 尝试加载已处理数据
        df_proc = load_data(PROCESSED_CSV_PATH)
        required_cols = [f'red{i+1}' for i in range(6)] + ['blue', '期号', 'red_sum'] # 基础检查
        if df_proc is not None and not df_proc.empty and all(c in df_proc.columns for c in required_cols):
            main_df = df_proc
            logger.info(f"成功加载已处理数据: {PROCESSED_CSV_PATH}")
    
    if main_df is None: # 如果未加载到已处理数据，则处理原始数据
        logger.info(f"处理原始数据: {CSV_FILE_PATH}")
        df_raw_main = load_data(CSV_FILE_PATH)
        if df_raw_main is not None and not df_raw_main.empty:
            df_clean_main = clean_and_structure(df_raw_main)
            if df_clean_main is not None and not df_clean_main.empty:
                main_df = feature_engineer(df_clean_main)
                if main_df is not None and not main_df.empty:
                    try: main_df.to_csv(PROCESSED_CSV_PATH, index=False); logger.info(f"已处理数据已保存到: {PROCESSED_CSV_PATH}")
                    except Exception as e_csv_save: logger.warning(f"保存已处理数据失败: {e_csv_save}")
                else: logger.error("特征工程失败。")
            else: logger.error("数据清洗失败。")
        else: logger.error("原始数据加载失败。")

    if main_df is None or main_df.empty:
        logger.error("数据准备失败，无法继续。"); sys.exit(1)
    
    # 对主DataFrame进行最终的数据类型转换和清洗
    for r_col_m in [f'red{i+1}' for i in range(6)]:
        if r_col_m in main_df.columns: main_df[r_col_m] = pd.to_numeric(main_df[r_col_m], errors='coerce')
    if 'blue' in main_df.columns: main_df['blue'] = pd.to_numeric(main_df['blue'], errors='coerce')
    main_df.dropna(subset=([f'red{i+1}' for i in range(6)] + ['blue']), inplace=True) # 删除包含NaN球号的行
    for r_col_m in [f'red{i+1}' for i in range(6)]:
        if r_col_m in main_df.columns: main_df[r_col_m] = main_df[r_col_m].astype(int)
    if 'blue' in main_df.columns: main_df['blue'] = main_df['blue'].astype(int)

    set_console_verbosity(original_console_level_main, use_simple_formatter=True) # 恢复主程序开始时的控制台级别

    full_history_arm_rules = pd.DataFrame() # 初始化用于存储完整历史关联规则的DataFrame

    # 判断是否运行Optuna优化
    run_optuna_trigger = (load_status == 'defaults_used_new_config_saved' or \
                          load_status == 'defaults_used_config_error')

    optuna_timeout_setting = None # 设置为None表示不限时，或设置为秒数以启用超时

    # 将这两个变量移到 run_optuna_trigger 块内部，但在回调函数定义之前
    # 这样它们在回调函数的作用域内是已定义的，并且不需要 nonlocal
    # 为了在回调函数中修改它们，我们将它们声明为全局的。
    start_time_optuna_global = 0 
    estimated_time_logged_global = False


    if run_optuna_trigger:
        logger.info(f"\n>>> 权重配置文件缺失或无效 (状态: {load_status})。开始权重优化过程...")
        # Optuna优化所需的最少数据量
        min_data_for_opt = (max(ML_LAG_FEATURES) if ML_LAG_FEATURES else 0) + 1 + MIN_POSITIVE_SAMPLES_FOR_ML + OPTIMIZATION_BACKTEST_PERIODS
        if len(main_df) >= min_data_for_opt:
            df_for_opt_objective = main_df.copy() # 使用完整数据集进行优化目标评估
            optuna.logging.set_verbosity(optuna.logging.WARNING) # 设置Optuna日志级别
            
            optuna_study = optuna.create_study(direction='minimize') # Optuna研究：目标是最小化（因为我们返回负的性能分）
            
            # 根据optuna_timeout_setting设置日志信息
            timeout_message = "不超时" if optuna_timeout_setting is None else f"超时: {optuna_timeout_setting}s"
            logger.info(f"Optuna优化试验次数: {OPTIMIZATION_TRIALS}, {timeout_message}")
            
            # --- Optuna 运行时间和预估回调 ---
            # 在 run_optuna_trigger 块内，但在回调定义之前初始化
            start_time_optuna_global = time.time() 
            estimated_time_logged_global = False

            def optuna_time_estimation_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
                global estimated_time_logged_global, start_time_optuna_global

                if estimated_time_logged_global:
                    return

                estimation_trigger_trial_count = 1
                
                # 使用 len(study.trials) 获取当前已创建（包括正在运行和已完成）的试验数量
                # 或者 trial.number + 1 (因为trial.number是0-indexed)
                current_trials_count = trial.number + 1 # trial.number 是当前试验的索引 (从0开始)

                if current_trials_count >= estimation_trigger_trial_count:
                    current_duration = time.time() - start_time_optuna_global
                    # 使用 current_trials_count 作为已完成（或至少已开始）的试验数
                    avg_time_per_trial = current_duration / current_trials_count 
                    remaining_trials = OPTIMIZATION_TRIALS - current_trials_count
                    
                    # 仅当有剩余试验时才计算预估时间
                    if remaining_trials > 0:
                        estimated_remaining_time = avg_time_per_trial * remaining_trials
                        hours, rem = divmod(estimated_remaining_time, 3600)
                        minutes, seconds = divmod(rem, 60)
                        
                        original_console_level_callback = global_console_handler.level
                        original_console_formatter_callback = global_console_handler.formatter
                        set_console_verbosity(logging.INFO, use_simple_formatter=False)
                        
                        logger.info(f"Optuna 优化进度: 已完成 {current_trials_count}/{OPTIMIZATION_TRIALS} 次试验。")
                        logger.info(f"  平均每次试验耗时: {avg_time_per_trial:.2f} 秒。")
                        logger.info(f"  预估剩余完成时间: {int(hours):02d}小时 {int(minutes):02d}分钟 {int(seconds):02d}秒。")
                        
                        global_console_handler.setLevel(original_console_level_callback)
                        global_console_handler.setFormatter(original_console_formatter_callback)
                    else:
                        # 如果没有剩余试验（例如，这是最后一次试验），可以不打印预估时间或打印一个完成消息
                        pass # 或者 logger.info("Optuna 优化即将完成...")
                        
                    estimated_time_logged_global = True
            # --- 回调结束 ---

            try:
                # 运行Optuna优化
                optuna_study.optimize(
                    lambda trial_obj: objective(trial_obj, df_for_opt_objective, ML_LAG_FEATURES),
                    n_trials=OPTIMIZATION_TRIALS,
                    timeout=optuna_timeout_setting, # 使用变量控制超时
                    n_jobs=1,
                    callbacks=[optuna_time_estimation_callback] # 添加回调函数
                )
            except optuna.exceptions.OptunaError as e_optuna: # 捕获Optuna可能的错误，例如超时
                 logger.error(f"Optuna优化过程中发生错误或被中断: {e_optuna}")
            except KeyboardInterrupt:
                 logger.warning("Optuna优化被用户中断。")


            best_params_from_optuna = optuna_study.best_params # 获取最佳参数
            updated_weights = DEFAULT_WEIGHTS.copy() # 从默认权重开始更新

            # 将Optuna找到的最佳参数更新到权重字典
            for key, value in best_params_from_optuna.items():
                 if key in updated_weights:
                     if isinstance(updated_weights[key], int):
                         updated_weights[key] = int(round(value)) # 整数参数四舍五入
                     elif isinstance(updated_weights[key], float):
                         updated_weights[key] = float(value) # 浮点数参数
            
            # 再次确保候选池比例有效
            prop_h_opt = updated_weights.get('CANDIDATE_POOL_PROPORTIONS_HIGH', DEFAULT_WEIGHTS['CANDIDATE_POOL_PROPORTIONS_HIGH'])
            prop_m_opt = updated_weights.get('CANDIDATE_POOL_PROPORTIONS_MEDIUM', DEFAULT_WEIGHTS['CANDIDATE_POOL_PROPORTIONS_MEDIUM'])
            if prop_h_opt + prop_m_opt > 1.0:
                total_prop_opt = prop_h_opt + prop_m_opt
                if total_prop_opt > 1e-9:
                    updated_weights['CANDIDATE_POOL_PROPORTIONS_HIGH'] = prop_h_opt / total_prop_opt
                    updated_weights['CANDIDATE_POOL_PROPORTIONS_MEDIUM'] = prop_m_opt / total_prop_opt
                else: 
                    updated_weights['CANDIDATE_POOL_PROPORTIONS_HIGH'] = 0.5 
                    updated_weights['CANDIDATE_POOL_PROPORTIONS_MEDIUM'] = 0.5

            logger.info(f"\n>>> Optuna优化完成。")
            if optuna_study.best_trial: # 检查是否有最佳试验
                logger.info(f"  最佳目标函数值 (负的性能分): {optuna_study.best_value:.4f}")
                logger.info(f"  最佳参数 (部分):")
                for k, v in list(optuna_study.best_params.items())[:5]: # 显示部分最佳参数
                    logger.info(f"    {k}: {v}")
            else:
                logger.info("  Optuna未找到最佳试验 (可能由于提前中断或无有效试验)。")


            CURRENT_WEIGHTS = updated_weights # 更新当前使用的权重
            save_weights_to_file(WEIGHTS_CONFIG_FILE, CURRENT_WEIGHTS) # 保存优化后的权重
            logger.info("使用优化后的权重重新分析关联规则...")
            full_history_arm_rules = analyze_associations(main_df, CURRENT_WEIGHTS) # 使用新权重分析ARM
        else:
            logger.warning(f"数据不足 ({len(main_df)}期) 进行权重优化 (需要至少 {min_data_for_opt}期)。将使用默认权重。")
            # 回退：如果由于数据不足而跳过Optuna，则使用默认权重分析ARM规则
            logger.info("使用默认权重分析关联规则...")
            full_history_arm_rules = analyze_associations(main_df, CURRENT_WEIGHTS) # 此处的CURRENT_WEIGHTS是默认值
    else: # load_status == 'loaded_active_config' (成功加载现有配置)
        logger.info(f"\n>>> 成功加载现有权重配置文件 (状态: {load_status})。跳过优化。")
        logger.info("使用加载的权重分析关联规则...")
        full_history_arm_rules = analyze_associations(main_df, CURRENT_WEIGHTS)

    # 显示当前使用的部分权重
    logger.info(f"\n>>> 当前使用权重 (部分展示):")
    for key_to_show in ['NUM_COMBINATIONS_TO_GENERATE', 'DIVERSITY_MIN_DIFFERENT_REDS', 'ML_PROB_SCORE_WEIGHT_RED', 'ARM_COMBINATION_BONUS_WEIGHT', 'DIVERSITY_SUM_DIFF_THRESHOLD', 'ARM_MIN_SUPPORT']:
        logger.info(f"  {key_to_show}: {CURRENT_WEIGHTS.get(key_to_show, 'N/A')}")

    # 数据概况
    min_p_val, max_p_val, total_p_val = main_df['期号'].min(), main_df['期号'].max(), len(main_df)
    last_draw_dt = main_df['日期'].iloc[-1] if '日期' in main_df.columns and not main_df.empty else "未知"
    last_draw_period = main_df['期号'].iloc[-1] if not main_df.empty else "未知"
    
    set_console_verbosity(logging.INFO, use_simple_formatter=False) # 报告时使用详细日志
    logger.info(f"\n{'='*15} 数据概况 {'='*15}")
    logger.info(f"  数据范围: {min_p_val} - {max_p_val} (共 {total_p_val} 期)")
    logger.info(f"  最后开奖: {last_draw_dt} (期号: {last_draw_period})")

    # 检查是否有足够数据进行完整分析和回测
    min_periods_for_full_run = (max(ML_LAG_FEATURES) if ML_LAG_FEATURES else 0) + 1 + MIN_POSITIVE_SAMPLES_FOR_ML + BACKTEST_PERIODS_COUNT
    if total_p_val < min_periods_for_full_run:
        logger.error(f"数据不足 ({total_p_val}期) 进行完整分析和回测报告 (需 {min_periods_for_full_run}期)。")
    else:
        # 完整历史统计分析
        logger.info(f"\n{'='*10} 完整历史统计分析 {'='*10}")
        original_console_level_stats = global_console_handler.level
        set_console_verbosity(logging.INFO, use_simple_formatter=False) # 确保详细输出
        full_freq_d = analyze_frequency_omission(main_df, CURRENT_WEIGHTS)
        full_patt_d = analyze_patterns(main_df, CURRENT_WEIGHTS)
        
        logger.info(f"  热门红球 (Top 5): {[int(x) for x in full_freq_d.get('hot_reds', [])[:5]]}")
        logger.info(f"  冷门红球 (Bottom 5): {[int(x) for x in full_freq_d.get('cold_reds', [])[-5:]]}")
        logger.info(f"  最近 {RECENT_FREQ_WINDOW} 期热门红球: " + str(sorted([(int(k),v) for k,v in full_freq_d.get('recent_N_freq_red', {}).items() if v > 0], key=lambda x: x[1], reverse=True)[:5]))
        logger.info(f"  最常见红球奇偶比: {full_patt_d.get('most_common_odd_even_count')}")
        if full_history_arm_rules is not None and not full_history_arm_rules.empty: 
            logger.info(f"  发现 {len(full_history_arm_rules)} 条关联规则 (Top 3 LIFT): \n{full_history_arm_rules.head(3).to_string(index=False)}")
        else: logger.info("  未找到显著关联规则.")
        global_console_handler.setLevel(original_console_level_stats) # 恢复之前的控制台日志级别

        # 回测摘要
        logger.info(f"\n{'='*15} 回测摘要 {'='*15}")
        set_console_verbosity(logging.INFO, use_simple_formatter=True) # 回测时用简洁日志，避免过多输出
        backtest_res_df, extended_bt_stats = backtest(main_df, ML_LAG_FEATURES, CURRENT_WEIGHTS, full_history_arm_rules, BACKTEST_PERIODS_COUNT)
        set_console_verbosity(logging.INFO, use_simple_formatter=False) # 回测结果报告用详细日志

        if not backtest_res_df.empty:
            s_p_f = backtest_res_df.attrs.get('start_period_id', 'N/A'); e_p_f = backtest_res_df.attrs.get('end_period_id', 'N/A')
            num_tested_periods = backtest_res_df.attrs.get('num_periods_tested', 'N/A')
            logger.info(f"  回测期范围: {s_p_f} 至 {e_p_f} (共测试 {num_tested_periods} 期)")
            logger.info(f"  每期生成组合数: {extended_bt_stats.get('num_combinations_per_draw_tested', 'N/A')}")
            logger.info(f"  总评估组合数: {extended_bt_stats.get('total_combinations_evaluated', 'N/A')}")
            
            logger.info(f"  --- 整体命中表现 ---")
            logger.info(f"    每个组合平均红球命中: {backtest_res_df['red_hits'].mean():.3f}")
            logger.info(f"    每个组合加权(x^1.5)平均红球命中: {(backtest_res_df['red_hits']**1.5).mean():.3f}")
            
            blue_hit_overall_rate = backtest_res_df['blue_hit'].mean() * 100 
            logger.info(f"    蓝球命中率 (每个组合): {blue_hit_overall_rate:.2f}%")
            
            periods_any_blue_hit_count = extended_bt_stats.get('periods_with_any_blue_hit_count', 0)
            if isinstance(num_tested_periods, int) and num_tested_periods > 0:
                logger.info(f"    至少一个组合命中蓝球的期数占比: {periods_any_blue_hit_count / num_tested_periods:.2%}")

            logger.info(f"  --- 红球命中数分布 (按组合) ---")
            hit_counts_dist = backtest_res_df['red_hits'].value_counts(normalize=True).sort_index() * 100
            for hit_num, pct in hit_counts_dist.items():
                logger.info(f"    命中 {hit_num} 红球: {pct:.2f}%")

            logger.info(f"  --- 中奖等级统计 (按组合) ---")
            prize_dist = extended_bt_stats.get('prize_counts', {})
            if prize_dist:
                prize_order = {"一等奖": 1, "二等奖": 2, "三等奖": 3, "四等奖": 4, "五等奖": 5, "六等奖": 6}
                sorted_prize_dist = sorted(prize_dist.items(), key=lambda item: prize_order.get(item[0], 99)) # 按奖级排序
                for prize_level, count in sorted_prize_dist:
                    logger.info(f"    {prize_level}: {count} 次")
            else:
                logger.info("    未命中任何奖级。")

            best_hits_df = extended_bt_stats.get('best_hit_per_period_df') # 每期最佳命中统计
            if best_hits_df is not None and not best_hits_df.empty:
                logger.info(f"  --- 每期最佳红球命中数分布 ---") 
                best_red_dist = best_hits_df['max_red_hits'].value_counts(normalize=True).sort_index() * 100
                for hit_num, pct in best_red_dist.items():
                     if hit_num >=0 : logger.info(f"    最佳命中 {hit_num} 红球的期数占比: {pct:.2f}%") 
                
                if 'blue_hit_in_period' in best_hits_df.columns and isinstance(num_tested_periods, int) and num_tested_periods > 0:
                     periods_with_best_blue_hit = best_hits_df['blue_hit_in_period'].sum()
                     logger.info(f"    至少一个组合命中蓝球的期数占比 (来自best_hit_per_period): {periods_with_best_blue_hit / num_tested_periods:.2%}")
        else: logger.info("  最终回测未产生结果。")

        # 最终推荐号码
        logger.info(f"\n{'='*12} 最终推荐号码 {'='*12}")
        set_console_verbosity(logging.INFO, use_simple_formatter=True) # 推荐时用简洁日志
        final_recs_list, final_rec_strs_list, _, _, final_scores_dict, final_win_seg_pcts = analyze_and_recommend(
            main_df, ML_LAG_FEATURES, CURRENT_WEIGHTS, full_history_arm_rules, train_ml=True # 确保为最终推荐训练模型
        )
        for line_str in final_rec_strs_list: logger.info(line_str)

        set_console_verbosity(logging.INFO, use_simple_formatter=False) # 恢复详细日志
        logger.info(f"\n{'='*8} 中奖红球分数段历史分析 (基于最终分数) {'='*8}")
        if final_scores_dict.get('red_scores'):
            disp_cts, disp_pcts_vals = analyze_winning_red_ball_score_segments(main_df, final_scores_dict['red_scores'], SCORE_SEGMENT_BOUNDARIES, SCORE_SEGMENT_LABELS)

            tot_win_reds_d = sum(disp_cts.values())
            logger.info(f"  历史中奖红球分数段分布 (总计 {tot_win_reds_d} 个红球实例):")
            # 按分数段顺序显示
            for seg_name in sorted(disp_cts.keys(), key=lambda x: (int(x.split('-')[0]) if '-' in x else 999)):
                logger.info(f"    分数段 {seg_name}: {disp_cts.get(seg_name,0)} 个 ({disp_pcts_vals.get(seg_name,0.0):.2f}%)")
        else: logger.info("  无法进行分数段分析（无红球得分）。")

        logger.info(f"\n{'='*14} 7+7 复式参考 {'='*14}")
        r_s_77_f = final_scores_dict.get('red_scores', {}); b_s_77_f = final_scores_dict.get('blue_scores', {})
        if r_s_77_f and len(r_s_77_f)>=7 and b_s_77_f and len(b_s_77_f)>=7:
            top_7r_f = sorted([n_val for n_val,_ in sorted(r_s_77_f.items(), key=lambda i_item:i_item[1], reverse=True)[:7]])
            top_7b_f = sorted([n_val for n_val,_ in sorted(b_s_77_f.items(), key=lambda i_item:i_item[1], reverse=True)[:7]])
            logger.info(f"  推荐7红球: {[int(x) for x in top_7r_f]}")
            logger.info(f"  推荐7蓝球: {[int(x) for x in top_7b_f]}")
        else: logger.info("  评分号码不足以选择7+7。")

    logger.info(f"\n--- 分析报告结束 (详情请查阅日志文件: {log_filename}) ---")

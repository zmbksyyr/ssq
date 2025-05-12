import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import itertools
import random
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
# Removed RandomForestClassifier as we are adding others for probability prediction
# from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC # Added SVC
from sklearn.preprocessing import StandardScaler # Added StandardScaler
from sklearn.pipeline import Pipeline # Added Pipeline


# Try importing LightGBM, provide fallback if not installed
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None # Use None as a flag if import fails

from sklearn.metrics import accuracy_score, mean_squared_error
from typing import Union, Optional, List, Dict, Tuple, Any

import sys
import datetime
import os
import io
import logging
from contextlib import redirect_stdout, redirect_stderr

# --- 配置 ---
# 获取脚本目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 构建CSV文件的完整路径
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'shuangseqiu.csv')  # 假设shuangseqiu.csv与脚本在同一目录
PROCESSED_CSV_PATH = os.path.join(SCRIPT_DIR, 'shuangseqiu_processed.csv')  # 处理后的数据文件路径

RED_BALL_RANGE = range(1, 34)
BLUE_BALL_RANGE = range(1, 17)
RED_ZONES = {
    'Zone1': (1, 11),
    'Zone2': (12, 22),
    'Zone3': (23, 33)
}
NUM_COMBINATIONS_TO_GENERATE = 5  # 最终推荐的号码组合数量 (单式或小复式)
TOP_N_RED_FOR_CANDIDATE = 25  # 用于生成组合的红球候选池大小（按分数从高到低选择，预测概率后可以扩大池）
TOP_N_BLUE_FOR_CANDIDATE = 12  # 用于生成组合的蓝球候选池大小（按分数从高到低选择，预测概率后可以扩大池）
ML_LAG_FEATURES = [1, 3, 5, 10]  # ML模型使用的滞后特征期数，例如 [1, 3, 5] 表示使用前1期、前3期、前5期的数据作为特征
BACKTEST_PERIODS_COUNT = 100  # 回测使用的最近历史期数 (This will be maximum periods if data is sufficient)
SHOW_PLOTS = False  # 是否显示分析过程中生成的图表 (True 显示, False 屏蔽)

# 关联规则挖掘配置 (可按需调整)
ARM_MIN_SUPPORT = 0.008
ARM_MIN_CONFIDENCE = 0.35
ARM_MIN_LIFT = 1.0

# 评分权重 (启发式 - 可调整)
FREQ_SCORE_WEIGHT = 15 # 降低频率权重，增加ML概率权重
OMISSION_SCORE_WEIGHT = 10 # 降低遗漏权重
# Removed ODD_EVEN_TENDENCY_BONUS, ZONE_TENDENCY_BONUS_MULTIPLIER as ML predicts individual probabilities
BLUE_FREQ_SCORE_WEIGHT = 20 # 降低蓝球频率权重
BLUE_OMISSION_SCORE_WEIGHT = 8 # 降低蓝球遗漏权重
# Removed BLUE_ODD_TENDENCY_BONUS, BLUE_SIZE_TENDENCY_BONUS

# New weights for ML predicted probability score
ML_PROB_SCORE_WEIGHT_RED = 70 # ML预测红球概率的权重
ML_PROB_SCORE_WEIGHT_BLUE = 70 # ML预测蓝球概率的权重

# Combination bonus based on matching HISTORICAL patterns (still useful for structure)
COMBINATION_ODD_COUNT_MATCH_BONUS = 15  # 组合匹配历史最常见奇数数量的奖励
COMBINATION_BLUE_ODD_MATCH_BONUS = 10  # 组合匹配历史最常见蓝球奇偶的奖励
COMBINATION_ZONE_MATCH_BONUS = 10  # 组合匹配历史最常见区域模式的奖励
COMBINATION_BLUE_SIZE_MATCH_BONUS = 8  # 组合匹配历史最常见蓝球大小的奖励


# ML模型参数 (新增或修改)
# Parameters for LightGBM (adjust as needed, these are examples focusing on regularization)
# Adjust parameters here to control overfitting
LGBM_PARAMS = {
    'objective': 'binary', # Binary classification for ball presence
    'metric': 'binary_logloss',
    'n_estimators': 120,   # Number of boosting rounds, increase slightly
    'learning_rate': 0.05, # Reduce learning rate
    'feature_fraction': 0.7, # Fraction of features considered per iteration (column sampling)
    'bagging_fraction': 0.8, # Fraction of data sampled per iteration (row sampling)
    'bagging_freq': 5,
    'lambda_l1': 0.1, # L1 regularization - increase
    'lambda_l2': 0.1, # L2 regularization - increase
    'num_leaves': 16, # Maximum number of leaves in one tree (controls tree complexity) - decrease
    'verbose': -1, # Suppress verbose output
    'n_jobs': -1, # Use all available cores
    'seed': 42,
    'boosting_type': 'gbdt',
    # 'max_depth': -1, # No limit by default in LGBM, num_leaves is main control
}

# Parameters for Logistic Regression (adjust as needed)
LOGISTIC_REG_PARAMS = {
    'penalty': 'l2', # L2 regularization
    'C': 1.0, # Inverse of regularization strength; smaller values specify stronger regularization - decrease C for stronger regularization
    'solver': 'saga', # Good for small datasets, supports L1/L2
    'random_state': 42,
    'max_iter': 1000 # Increased max iterations for convergence
    # Removed 'n_jobs' as it has no effect with 'liblinear'
}

# Parameters for SVC (adjust as needed, probability=True enables predict_proba but is expensive)
# Using a linear kernel as a starting point for potentially better interpretability and less complexity than RBF
SVC_PARAMS = {
    'C': 0.8, # Regularization parameter. Smaller C means stronger regularization.
    'kernel': 'linear', # 'rbf', 'poly', 'sigmoid' are other options. Linear is simpler.
    'probability': True, # Enable probability estimates - makes SVC slower due to cross-validation
    'random_state': 42,
    'cache_size': 200, # Specify size of the kernel cache (in MB)
    'max_iter': 10000 # Increased max iterations significantly
    'tol': 1e-3  # 添加容差参数，允许更早收敛
    # SVC 对特征缩放敏感，在管道中使用 StandardScaler
}

# Minimum number of positive samples required to train a classifier for a specific ball
MIN_POSITIVE_SAMPLES_FOR_ML = 8 # Increased threshold slightly for robustness

# --- 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # 输出到控制台stdout
    ]
)
logger = logging.getLogger('ssq_analyzer')

# 添加进度条显示函数
def show_progress(current, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    显示进度条
    @param current: 当前进度
    @param total: 总进度
    @param prefix: 前缀字符串
    @param suffix: 后缀字符串
    @param decimals: 小数位数
    @param length: 进度条长度
    @param fill: 进度条填充字符
    """
    if total <= 0:
        print(f'\r{prefix} |{fill * length}| 100.0% {suffix}', end='')
        if current >= total:
            print()
        return

    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', file=sys.stdout, flush=True)  # 确保输出到stdout并刷新
    if current >= total:
        print(file=sys.stdout, flush=True)


# 创建上下文管理器来暂时重定向输出，并捕获stderr
class SuppressOutput:
    """
    上下文管理器：暂时抑制标准输出，捕获标准错误输出到StringIO对象。
    退出时可以将捕获的stderr写入日志。
    """
    def __init__(self, suppress_stdout=True, capture_stderr=True):
        self.suppress_stdout = suppress_stdout
        self.capture_stderr = capture_stderr
        self.stdout_redirect = None
        self.stderr_redirect = None
        self.old_stdout = None
        self.old_stderr = None
        self.stderr_io = None

    def __enter__(self):
        if self.suppress_stdout:
            self.old_stdout = sys.stdout
            # 将stdout重定向到/dev/null或等效位置
            sys.stdout = open(os.devnull, 'w')

        if self.capture_stderr:
            self.old_stderr = sys.stderr
            self.stderr_io = io.StringIO()
            sys.stderr = self.stderr_io

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 先恢复stderr
        if self.capture_stderr and self.old_stderr:
            sys.stderr = self.old_stderr
            captured_stderr_content = self.stderr_io.getvalue()
            if captured_stderr_content.strip():  # 如果捕获的stderr不为空，则记录日志
                 logger.warning(f"Captured stderr:\n{captured_stderr_content.strip()}")

        # 恢复stdout
        if self.suppress_stdout and self.old_stdout:
            if sys.stdout and not sys.stdout.closed:  # 在关闭前检查重定向对象是否有效
                 sys.stdout.close()
            sys.stdout = self.old_stdout

        # 不抑制异常
        return False


# --- 数据准备与基本处理 ---

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """从CSV加载数据。"""
    try:
        # 假设CSV有如'期号', '日期', '红球', '蓝球'等列
        # 如果默认utf-8失败，尝试不同的编码
        try:
             df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
             try:
                 df = pd.read_csv(file_path, encoding='gbk')
             except UnicodeDecodeError:
                 df = pd.read_csv(file_path, encoding='latin-1')  # 回退到latin-1或其他编码
        logger.info("数据加载成功。")
        logger.info(f"读取总期数: {len(df)}")
        return df
    except FileNotFoundError:
        logger.error(f"错误: 在 {file_path} 找不到文件")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"错误: {file_path} 文件为空")
        return None
    except Exception as e:
        logger.error(f"从 {file_path} 加载数据时发生错误: {e}")
        return None


def clean_and_structure(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """清理数据并结构化红球/蓝球。"""
    if df is None or df.empty:
        logger.warning("没有数据可清理和结构化。")
        return None

    initial_rows = len(df)
    # 删除缺少'期号', '红球'或'蓝球'的行
    df.dropna(subset=['期号', '红球', '蓝球'], inplace=True)
    if len(df) < initial_rows:
        logger.warning(f"删除了 {initial_rows - len(df)} 行缺少必要值的数据。")
        initial_rows = len(df)  # 更新用于后续检查

    if df.empty:
        logger.warning("删除缺少必要值的行后没有剩余数据。")
        return None

    # 确保'期号'为整数并按其排序
    try:
        df['期号'] = pd.to_numeric(df['期号'], errors='coerce').astype('Int64')  # 使用可空整数类型
        df.dropna(subset=['期号'], inplace=True)  # 删除转换失败的行
        df['期号'] = df['期号'].astype(int)  # 转换为标准int
        df.sort_values(by='期号', ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)  # 排序后重置索引
    except Exception as e:
        logger.error(f"错误: '期号'列无法清理或转换为整数。 {e}")
        return None

    if df.empty:
        logger.warning("清理'期号'后没有剩余数据。")
        return None

    parsed_rows_data = []
    rows_skipped_parsing = 0

    # 重新从解析的数据构建DataFrame以增强健壮性
    for index, row in df.iterrows():
         try:
             red_str = row.get('红球')  # 使用.get以保安全
             blue_val = row.get('蓝球')
             period_val = row.get('期号')

             if not isinstance(red_str, str) or pd.isna(blue_val) or pd.isna(period_val):
                  rows_skipped_parsing += 1
                  continue  # 如果基本数据缺失或类型错误则跳过

             # 提前处理潜在的非整数蓝球数据
             blue_num = int(blue_val)
             if not (1 <= blue_num <= 16):
                 rows_skipped_parsing += 1
                 continue

             # 在蓝球验证后处理红球
             reds = sorted([int(x) for x in red_str.split(',')])  # 排序的红球
             if len(reds) != 6 or not all(1 <= r <= 33 for r in reds):
                 rows_skipped_parsing += 1
                 continue

             # 如果到达这里，该行有效。添加到parsed_rows_data。
             row_data = {'期号': int(period_val)}  # 确保期号为int
             if '日期' in row:  # 如果存在，包含'日期'
                 row_data['日期'] = row['日期']
             # 添加排序后的红球和蓝球
             for i in range(6):
                 row_data[f'red{i+1}'] = reds[i]
                 # red_pos columns are identical to sorted red columns
                 row_data[f'red_pos{i+1}'] = reds[i]
             row_data['blue'] = blue_num

             parsed_rows_data.append(row_data)

         except (ValueError, AttributeError) as e:
             rows_skipped_parsing += 1
             period_val_safe = row.get('期号', 'N/A')
             # logger.warning(f"Parsing error for period {period_val_safe}: {e}. Skipping row.") # Too noisy, log count later
             continue
         except Exception as e:
             rows_skipped_parsing += 1
             period_val_safe = row.get('期号', 'N/A')
             logger.warning(f"General error processing period {period_val_safe}: {e}. Skipping row.")
             continue

    if rows_skipped_parsing > 0:
         logger.warning(f"由于解析错误，跳过了 {rows_skipped_parsing} 行。")

    if not parsed_rows_data:
         logger.error("全面解析后没有有效的数据行。")
         return None

    processed_df = pd.DataFrame(parsed_rows_data)

    # 确保按期号排序并重置索引
    processed_df.sort_values(by='期号', ascending=True, inplace=True)
    processed_df.reset_index(drop=True, inplace=True)

    logger.info(f"数据已清理和结构化。剩余有效期数: {len(processed_df)}")
    return processed_df


def feature_engineer(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """从结构化数据中提取特征。"""
    if df is None or df.empty:
        logger.warning("没有数据可进行特征工程。")
        return None

    # 确保清理/结构化后存在必要的列
    red_cols = [f'red{i+1}' for i in range(6)]
    red_pos_cols = [f'red_pos{i+1}' for i in range(6)]
    essential_cols = red_cols + red_pos_cols + ['blue', '期号'] # Add 期号 as essential
    if not all(col in df.columns for col in essential_cols):
        missing = [col for col in essential_cols if col not in df.columns]
        logger.error(f"清理后缺少特征工程所需的必要列: {missing}。")
        return None

    df_fe = df.copy()  # 使用副本

    # 红球和
    df_fe['red_sum'] = df_fe[red_cols].sum(axis=1)

    # 红球跨度
    df_fe['red_span'] = df_fe[red_cols].max(axis=1) - df_fe[red_cols].min(axis=1)

    # 红球奇偶计数
    df_fe['red_odd_count'] = df_fe[red_cols].apply(lambda row: sum(x % 2 != 0 for x in row), axis=1)
    df_fe['red_even_count'] = 6 - df_fe['red_odd_count']

    # 红球区域计数
    for zone, (start, end) in RED_ZONES.items():
        df_fe[f'red_{zone}_count'] = df_fe[red_cols].apply(lambda row: sum(start <= x <= end for x in row), axis=1)

    # 连续数字计数（使用排序球的简单对计数）
    def count_consecutive_pairs(row):
        count = 0
        # 使用表示排序位置的red_pos_cols
        # Check for potential NaN or empty values in the row slice before accessing
        if row[red_pos_cols].isnull().any() or row[red_pos_cols].empty:
             return 0
        # Ensure values are integers before comparison
        pos_balls = row[red_pos_cols].astype(int).tolist()
        for i in range(5):
            if pos_balls[i] + 1 == pos_balls[i+1]:
                count += 1
        return count

    # 仅当df_fe不为空时应用
    if not df_fe.empty:
        # Ensure red_pos_cols are numeric before applying function
        try:
             df_fe[red_pos_cols] = df_fe[red_pos_cols].astype(int)
             df_fe['red_consecutive_pairs'] = df_fe.apply(count_consecutive_pairs, axis=1)
        except ValueError as e:
             logger.error(f"Error converting red_pos_cols to int during feature engineering: {e}. red_consecutive_pairs will be NaN.")
             df_fe['red_consecutive_pairs'] = np.nan # Assign NaN if conversion fails
    else:
         df_fe['red_consecutive_pairs'] = pd.Series(dtype=int)


    # 与上期重复（红球）
    # 需要处理第一行。先添加shift，然后计算。
    # Create a string representation of the sorted red balls for easy comparison
    df_fe['current_reds_str'] = df_fe[red_cols].astype(str).agg(','.join, axis=1)
    df_fe['prev_reds_str'] = df_fe['current_reds_str'].shift(1)

    df_fe['red_repeat_count'] = 0  # Initialize
    if len(df_fe) > 1:
        for i in range(1, len(df_fe)):
            prev_reds_str = df_fe.loc[i, 'prev_reds_str']
            current_reds_str = df_fe.loc[i, 'current_reds_str']

            if pd.notna(prev_reds_str) and pd.notna(current_reds_str):
                 try:
                     prev_reds = set(int(x) for x in prev_reds_str.split(','))
                     current_reds = set(int(x) for x in current_reds_str.split(','))
                     df_fe.loc[i, 'red_repeat_count'] = len(prev_reds.intersection(current_reds))
                 except ValueError:
                     # Should not happen after robust cleaning, but as a safeguard
                     logger.warning(f"Error parsing red ball strings for repeat count at index {i}. Setting repeat count to 0.")
                     df_fe.loc[i, 'red_repeat_count'] = 0
            else:
                 # First row or missing previous data
                 df_fe.loc[i, 'red_repeat_count'] = 0


    df_fe.drop(columns=['current_reds_str', 'prev_reds_str'], errors='ignore', inplace=True)


    # 蓝球特征
    df_fe['blue_is_odd'] = df_fe['blue'] % 2 != 0
    df_fe['blue_is_large'] = df_fe['blue'] > 8  # 1-8 小，9-16 大
    # 添加蓝球质数状态（简化）
    primes = {2, 3, 5, 7, 11, 13}
    df_fe['blue_is_prime'] = df_fe['blue'].apply(lambda x: x in primes)

    logger.info("特征工程完成。")
    return df_fe

# --- 历史统计与模式分析 ---

def analyze_frequency_omission(df: pd.DataFrame) -> dict:
    """分析每个号码和位置的频率和当前遗漏。"""
    if df is None or df.empty:
        logger.warning("没有数据可分析频率和遗漏。")
        return {}  # 返回空字典

    red_cols = [f'red{i+1}' for i in range(6)]
    red_pos_cols = [f'red_pos{i+1}' for i in range(6)] # Assuming red_pos are sorted red balls
    # Use *current* index to calculate omission relative to the latest period in this df slice
    most_recent_period_index_in_df = len(df) - 1

    if most_recent_period_index_in_df < 0:
        logger.warning("DataFrame is empty in analyze_frequency_omission after checks.")
        return {}

    # Ensure ball columns are numeric
    try:
        df[red_cols + ['blue']] = df[red_cols + ['blue']].astype(int)
        # df[red_pos_cols] = df[red_pos_cols].astype(int) # red_pos should be same as red_cols after cleaning
    except ValueError as e:
        logger.error(f"Error converting ball columns to int for frequency/omission analysis: {e}. Skipping analysis.")
        return {}


    all_reds = df[red_cols].values.flatten()
    all_blues = df['blue'].values

    # 频率（总体）
    red_freq = Counter(all_reds)
    blue_freq = Counter(all_blues)

    # 位置红球频率 - Based on red_pos_cols which are sorted red balls
    red_pos_freq = {}
    for col in red_pos_cols:
         if col in df.columns:
              red_pos_freq[col] = Counter(df[col])
         else:
              logger.warning(f"Column {col} not found for position frequency analysis.")


    # 遗漏（当前）- 上次出现至今的期数
    # Omission is calculated relative to the *end* of the provided df.
    # An omission of 0 means it appeared in the latest period (index most_recent_period_index_in_df).
    # An omission of k means it last appeared at index most_recent_period_index_in_df - k.
    # If never seen in this df, omission is most_recent_period_index_in_df + 1 (or len(df))

    current_omission = {}
    # Red balls (any position)
    for number in RED_BALL_RANGE:
        latest_appearance_index = df.index[
            (df[red_cols] == number).any(axis=1)
        ].max() # This gives the index *within the current df*

        if pd.isna(latest_appearance_index):
             current_omission[number] = len(df) # Never seen in these data
        else:
             current_omission[number] = most_recent_period_index_in_df - latest_appearance_index

    # Red balls (by position)
    red_pos_current_omission = {}
    for col in red_pos_cols:
        red_pos_current_omission[col] = {}
        if col in df.columns:
            for number in RED_BALL_RANGE:
                latest_appearance_index = df.index[
                     (df[col] == number)
                ].max()
                if pd.isna(latest_appearance_index):
                    red_pos_current_omission[col][number] = len(df)
                else:
                    red_pos_current_omission[col][number] = most_recent_period_index_in_df - latest_appearance_index
        else:
             logger.warning(f"Column {col} not found for position omission analysis.")


    # Blue balls
    for number in BLUE_BALL_RANGE:
         latest_appearance_index = df.index[
             (df['blue'] == number)
         ].max()
         if pd.isna(latest_appearance_index):
             current_omission[number] = len(df)
         else:
            current_omission[number] = most_recent_period_index_in_df - latest_appearance_index

    # 平均间隔（平均遗漏的代理）- calculated on the provided data
    average_interval = {}
    total_periods = len(df)
    # Add 1 to frequency in denominator to handle balls that never appeared, avoiding division by zero
    for number in RED_BALL_RANGE:
        average_interval[number] = total_periods / (red_freq.get(number, 0) + 1e-9) # Use small epsilon
    for number in BLUE_BALL_RANGE:
        average_interval[number] = total_periods / (blue_freq.get(number, 0) + 1e-9) # Use small epsilon

    # 位置平均间隔
    red_pos_average_interval = {}
    for col in red_pos_cols:
        red_pos_average_interval[col] = {}
        if col in red_pos_freq: # Use red_pos_freq which might be empty if col was missing
            col_freq = red_pos_freq.get(col, {})
            for number in RED_BALL_RANGE:
                 red_pos_average_interval[col][number] = total_periods / (col_freq.get(number, 0) + 1e-9) # Use small epsilon
        else:
             logger.warning(f"Position frequency data missing for {col}, cannot calculate average interval.")


    # 识别热/冷号（基于总体频率）
    # Handle empty frequency data gracefully
    red_freq_items = sorted(red_freq.items(), key=lambda item: item[1], reverse=True) if red_freq else []
    blue_freq_items = sorted(blue_freq.items(), key=lambda item: item[1], reverse=True) if blue_freq else []

    # Define hot/cold based on top/bottom percentage (ensure thresholds are valid indices)
    num_red_balls_possible = len(RED_BALL_RANGE)
    num_blue_balls_possible = len(BLUE_BALL_RANGE)

    # Calculate thresholds ensuring they are within the bounds of the actual data points
    red_hot_threshold = max(0, min(len(red_freq_items), int(num_red_balls_possible * 0.2)))
    red_cold_threshold = max(0, min(len(red_freq_items), int(num_red_balls_possible * 0.8)))
    blue_hot_threshold = max(0, min(len(blue_freq_items), int(num_blue_balls_possible * 0.3)))
    blue_cold_threshold = max(0, min(len(blue_freq_items), int(num_blue_balls_possible * 0.7)))


    hot_reds = [num for num, freq in red_freq_items[:red_hot_threshold]]
    # Ensure cold reds are distinct from hot reds if thresholds overlap or data is sparse
    cold_reds_candidates = red_freq_items[red_cold_threshold:]
    cold_reds = [num for num, freq in cold_reds_candidates if num not in hot_reds]


    hot_blues = [num for num, freq in blue_freq_items[:blue_hot_threshold]]
    # Ensure cold blues are distinct
    cold_blues_candidates = blue_freq_items[blue_cold_threshold:]
    cold_blues = [num for num, freq in cold_blues_candidates if num not in hot_blues]


    analysis_results = {
        'red_freq': red_freq,
        'blue_freq': blue_freq,
        'red_pos_freq': red_pos_freq,
        'current_omission': current_omission,
        'red_pos_current_omission': red_pos_current_omission,
        'average_interval': average_interval,
        'red_pos_average_interval': red_pos_average_interval,
        'hot_reds': hot_reds,
        'cold_reds': cold_reds,
        'hot_blues': hot_blues,
        'cold_blues': cold_blues
    }

    return analysis_results


def analyze_patterns(df: pd.DataFrame) -> dict:
    """分析计算特征的分布。"""
    if df is None or df.empty:
        logger.warning("没有数据可分析模式。")
        return {}  # 返回空字典

    # Pattern analysis results
    pattern_results = {}

    # Helper to get mode safely and convert to int
    def safe_mode_int(series):
        if series is None or series.empty or series.mode().empty:
            return None
        return int(series.mode()[0])

    # Red ball sum distribution
    if 'red_sum' in df.columns and not df['red_sum'].empty:
        pattern_results['sum_stats'] = df['red_sum'].describe().to_dict()
        pattern_results['most_common_sum'] = safe_mode_int(df['red_sum'])
    else:
         pattern_results['sum_stats'] = {}
         pattern_results['most_common_sum'] = None

    # Red ball span distribution
    if 'red_span' in df.columns and not df['red_span'].empty:
        pattern_results['span_stats'] = df['red_span'].describe().to_dict()
        pattern_results['most_common_span'] = safe_mode_int(df['red_span'])
    else:
        pattern_results['span_stats'] = {}
        pattern_results['most_common_span'] = None

    # Odd/even count distribution
    if 'red_odd_count' in df.columns and not df['red_odd_count'].empty:
        # Ensure counts are integers before value_counts
        if pd.api.types.is_numeric_dtype(df['red_odd_count']):
             odd_even_counts = df['red_odd_count'].astype(int).value_counts().sort_index()
             pattern_results['odd_even_ratios'] = {f'{odd}:{6-odd}': int(count) for odd, count in odd_even_counts.items()}
             pattern_results['most_common_odd_even_count'] = safe_mode_int(df['red_odd_count'])
        else:
             logger.warning("red_odd_count column is not numeric, skipping odd/even distribution analysis.")
             pattern_results['odd_even_ratios'] = {}
             pattern_results['most_common_odd_even_count'] = None
    else:
        pattern_results['odd_even_ratios'] = {}
        pattern_results['most_common_odd_even_count'] = None

    # Zone distribution
    zone_cols = [f'red_{zone}_count' for zone in RED_ZONES.keys()]
    if all(col in df.columns for col in zone_cols) and not df.empty:
        zone_counts_df = df[zone_cols]
        if not zone_counts_df.empty and pd.api.types.is_numeric_dtype(zone_counts_df.values):
            # Ensure counts are integers, then form tuples
            zone_counts_df = zone_counts_df.astype(int)
            zone_distribution_counts = zone_counts_df.apply(lambda row: tuple(row), axis=1).value_counts()
            # Convert keys (tuples) and values (counts) to standard Python types
            pattern_results['zone_distribution_counts'] = {
                tuple(int(c) for c in dist): int(count) for dist, count in zone_distribution_counts.items()
            }
            pattern_results['most_common_zone_distribution'] = zone_distribution_counts.index[0] if not zone_distribution_counts.empty else (0, 0, 0)
        else:
            logger.warning("Zone count columns not found or not numeric, skipping zone distribution analysis.")
            pattern_results['zone_distribution_counts'] = {}
            pattern_results['most_common_zone_distribution'] = (0, 0, 0) # Default
    else:
         pattern_results['zone_distribution_counts'] = {}
         pattern_results['most_common_zone_distribution'] = (0, 0, 0) # Default

    # Consecutive pairs distribution
    if 'red_consecutive_pairs' in df.columns and not df['red_consecutive_pairs'].empty:
        if pd.api.types.is_numeric_dtype(df['red_consecutive_pairs']):
            consecutive_counts = df['red_consecutive_pairs'].astype(int).value_counts().sort_index()
            pattern_results['consecutive_counts'] = {int(count): int(freq) for count, freq in consecutive_counts.items()}
        else:
             logger.warning("red_consecutive_pairs column is not numeric, skipping consecutive counts analysis.")
             pattern_results['consecutive_counts'] = {}
    else:
        pattern_results['consecutive_counts'] = {}

    # Repeat counts frequency
    if 'red_repeat_count' in df.columns and not df['red_repeat_count'].empty:
        if pd.api.types.is_numeric_dtype(df['red_repeat_count']):
            repeat_counts = df['red_repeat_count'].astype(int).value_counts().sort_index()
            pattern_results['repeat_counts'] = {int(count): int(freq) for count, freq in repeat_counts.items()}
        else:
             logger.warning("red_repeat_count column is not numeric, skipping repeat counts analysis.")
             pattern_results['repeat_counts'] = {}
    else:
        pattern_results['repeat_counts'] = {}

    # Blue ball pattern analysis
    # Ensure blue_is_odd is boolean or convertible to boolean
    if 'blue_is_odd' in df.columns and not df['blue_is_odd'].empty:
         try:
             blue_odd_counts = df['blue_is_odd'].astype(bool).value_counts()
             pattern_results['blue_odd_counts'] = {bool(is_odd): int(count) for is_odd, count in blue_odd_counts.items()} # Convert bool key, int value
         except ValueError:
             logger.warning("blue_is_odd column cannot be converted to boolean, skipping blue oddness analysis.")
             pattern_results['blue_odd_counts'] = {}
    else:
        pattern_results['blue_odd_counts'] = {}

    # Ensure blue_is_large is boolean or convertible to boolean
    if 'blue_is_large' in df.columns and not df['blue_is_large'].empty:
        try:
            blue_large_counts = df['blue_is_large'].astype(bool).value_counts()
            pattern_results['blue_large_counts'] = {bool(is_large): int(count) for is_large, count in blue_large_counts.items()}
        except ValueError:
            logger.warning("blue_is_large column cannot be converted to boolean, skipping blue size analysis.")
            pattern_results['blue_large_counts'] = {}
    else:
        pattern_results['blue_large_counts'] = {}

    # Ensure blue_is_prime is boolean or convertible to boolean
    if 'blue_is_prime' in df.columns and not df['blue_is_prime'].empty:
        try:
            blue_prime_counts = df['blue_is_prime'].astype(bool).value_counts()
            pattern_results['blue_prime_counts'] = {bool(is_prime): int(count) for is_prime, count in blue_prime_counts.items()}
        except ValueError:
            logger.warning("blue_is_prime column cannot be converted to boolean, skipping blue prime analysis.")
            pattern_results['blue_prime_counts'] = {}
    else:
        pattern_results['blue_prime_counts'] = {}


    return pattern_results


def analyze_associations(df: pd.DataFrame, min_support: float = ARM_MIN_SUPPORT, min_confidence: float = ARM_MIN_CONFIDENCE, min_lift: float = ARM_MIN_LIFT) -> pd.DataFrame:
    """查找红球的频繁项集和关联规则。"""
    # 需要至少2期以查找关联
    if df is None or df.empty or len(df) < 2:
        logger.warning("Not enough data for association rule mining (need at least 2 periods).")
        return pd.DataFrame()  # 返回空DataFrame

    red_cols = [f'red{i+1}' for i in range(6)]
    # 检查红球列是否存在且在切片中不全为NaN/空
    if not all(col in df.columns for col in red_cols) or df[red_cols].isnull().all().all():
         logger.warning("Red ball columns missing or all NaN for association rule mining.")
         return pd.DataFrame()

    # 将球号转换为字符串以用于TransactionEncoder（通常更安全）
    # Filter out any rows where red_cols might contain NaN after subsetting
    transactions_df = df.dropna(subset=red_cols)
    if transactions_df.empty:
        logger.warning("No complete red ball rows available after dropping NaNs for ARM.")
        return pd.DataFrame()

    transactions = transactions_df[red_cols].astype(str).values.tolist()


    # Filter out empty transactions (shouldn't happen after dropna but safety)
    transactions = [t for t in transactions if all(item and item != 'nan' for item in t)]
    if not transactions:
         logger.warning("No valid transactions found for association rule mining after filtering.")
         return pd.DataFrame()


    te = TransactionEncoder()
    try:
        te_ary = te.fit(transactions).transform(transactions)
        df_onehot = pd.DataFrame(te_ary, columns=te.columns_)
    except Exception as e:
        logger.warning(f"Error during association rule TransactionEncoder transformation: {e}")
        return pd.DataFrame()

    if df_onehot.empty:
        logger.warning("One-hot encoded DataFrame is empty after ARM transformation.")
        return pd.DataFrame()

    try:
        # Adjust min_support based on data size to require a minimum absolute frequency
        # Use a minimum number of occurrences instead of a strict percentage if data is small
        min_support_abs = max(2, int(min_support * len(df_onehot))) # Need at least 2 occurrences
        min_support_adj = min_support_abs / len(df_onehot) if len(df_onehot) > 0 else 0

        frequent_itemsets = apriori(df_onehot, min_support=min_support_adj, use_colnames=True)

        if frequent_itemsets.empty:
             logger.info(f"No frequent itemsets found with adjusted min_support={min_support_adj:.4f}.")
             return pd.DataFrame()

        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    except Exception as e:
        logger.warning(f"Error during Apriori algorithm execution: {e}")
        return pd.DataFrame()

    try:
        # Generate rules with minimum confidence and lift
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
        if min_confidence is not None: # Apply confidence threshold if specified
             rules = rules[rules['confidence'] >= min_confidence]

        rules.sort_values(by='lift', ascending=False, inplace=True)
    except Exception as e:
        logger.warning(f"Error during association rules generation: {e}")
        return pd.DataFrame()

    return rules

# --- 用于预测号码概率的ML ---

def create_lagged_features(df: pd.DataFrame, lags: List[int]) -> Optional[pd.DataFrame]:
    """为ML模型创建滞后特征。"""
    if df is None or df.empty or not lags:
         logger.warning("Input DataFrame empty or lags list is empty for feature creation.")
         return None

    # Select base features to lag - ensure these columns exist and are numeric
    lag_base_cols = ['red_sum', 'red_span', 'red_odd_count', 'red_consecutive_pairs', 'red_repeat_count']
    # Add zone counts as base features
    lag_base_cols.extend([f'red_{zone}_count' for zone in RED_ZONES.keys()])
    # Add blue features
    # Explicitly convert boolean/object blue features to int before lagging
    df_temp = df.copy()
    for col in ['blue', 'blue_is_odd', 'blue_is_large', 'blue_is_prime']:
         if col in df_temp.columns:
             if pd.api.types.is_bool_dtype(df_temp[col]):
                 df_temp[col] = df_temp[col].astype(int) # Convert boolean to 0 or 1
             elif not pd.api.types.is_numeric_dtype(df_temp[col]):
                  # Try converting other non-numeric columns if they somehow appear here
                  try:
                       df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
                       logger.warning(f"Coerced blue column '{col}' to numeric for lagging.")
                  except Exception as e:
                       logger.error(f"Could not convert blue column '{col}' to numeric: {e}. It might cause issues.")
                       # Keep original dtype if coercion fails, but warn


    lag_base_cols.extend(['blue', 'blue_is_odd', 'blue_is_large', 'blue_is_prime'])


    # Filter out columns that don't exist in the dataframe (using df_temp now)
    existing_lag_cols = [col for col in lag_base_cols if col in df_temp.columns]


    if not existing_lag_cols:
         logger.warning("No base columns found for creating lagged features.")
         return None

    # Ensure remaining base columns are numeric before lagging (should be handled by blue conversion above, but safety check)
    for col in existing_lag_cols:
        if not pd.api.types.is_numeric_dtype(df_temp[col]):
             logger.warning(f"Column '{col}' is not numeric after preparation and will be skipped for lagging base.")
             existing_lag_cols.remove(col) # Remove from list if not numeric


    if not existing_lag_cols:
         logger.warning("No numeric base columns remaining after filtering/coercion for lagging.")
         return None

    df_lagged = df_temp[existing_lag_cols].copy()


    for lag in lags:
        if lag > 0:
            for col in existing_lag_cols:
                 # Use .name to get original column name
                 df_lagged[f'{col}_lag{lag}'] = df_lagged[col].shift(lag)


    # Drop rows with NaN values introduced by lagging
    initial_rows = len(df_lagged)
    df_lagged.dropna(inplace=True)
    if len(df_lagged) < initial_rows:
         # logger.info(f"Dropped {initial_rows - len(df_lagged)} rows due to lagging NaNs.") # Avoid noise in backtest
         pass


    if df_lagged.empty:
        logger.warning("Lagged DataFrame is empty after dropping NaNs.")
        return None

    # The features are the lagged columns
    feature_cols = [col for col in df_lagged.columns if any(f'_lag{lag}' in col for lag in lags)]


    # Ensure feature columns actually exist in the dataframe after potential drops
    feature_cols = [col for col in feature_cols if col in df_lagged.columns]


    if not feature_cols:
         logger.warning("No feature columns created after lagging and dropping NaNs.")
         return None

    # Return DataFrame containing lagged features (X) and original (non-lagged) base columns (which are targets for the current period)
    # The non-lagged base columns are only needed if we were predicting features directly.
    # For predicting individual ball presence, we need the actual balls drawn in the target period.
    # So just return the features DataFrame X.
    return df_lagged[feature_cols]


def train_prediction_models(df_train_raw: pd.DataFrame, lags: List[int]) -> Optional[dict]:
    """训练ML模型以预测下一期单个号码出现的概率。"""

    # 从训练数据创建滞后特征。create_lagged_features 返回的DF只包含特征X
    X = create_lagged_features(df_train_raw.copy(), lags)

    if X is None or X.empty:
        logger.warning("无法创建滞后特征，ML模型无法训练。")
        return None  # 如果无法训练则返回None

    # 目标(y) 是当前期（与滞后特征对齐的期）的开奖号码。
    # create_lagged_features 丢弃了由滞后引入的开始的行。
    # 我们需要从 df_train_raw 中获取与 X 对齐的实际开奖数据
    # X 的索引对应于 df_train_raw.loc[X.index]
    target_df = df_train_raw.loc[X.index].copy()


    if target_df.empty:
         logger.warning("目标DataFrame为空，ML模型无法训练。")
         return None

    # 检查是否存在红蓝球列并且是数值类型
    red_cols = [f'red{i+1}' for i in range(6)]
    if not all(col in target_df.columns for col in red_cols) or 'blue' not in target_df.columns:
        logger.error("目标DataFrame中缺少红球或蓝球列，无法训练号码概率模型。")
        return None

    try:
        target_df[red_cols + ['blue']] = target_df[red_cols + ['blue']].astype(int)
    except ValueError as e:
        logger.error(f"无法将目标红球或蓝球列转换为整数: {e}。无法训练号码概率模型。")
        return None


    trained_models = {
        'red': {},
        'blue': {}
    }

    logger.info("开始训练ML模型预测单个号码概率...")

    # 红球模型
    for ball in RED_BALL_RANGE:
        # 创建目标变量 y_red: 1 如果红球出现在本期，0 否则
        y_red = target_df[red_cols].apply(lambda row: ball in row.values, axis=1).astype(int)

        if y_red.empty or len(y_red) != len(X):
            logger.warning(f"红球 {ball} 的目标变量Y ({len(y_red)}) 与特征X ({len(X)}) 长度不匹配或为空，跳过训练。")
            continue

        # 检查目标变量的类别分布，确保至少有两个类别且正样本足够多才能训练分类器
        positive_count = y_red.sum()
        if len(y_red.unique()) < 2 or positive_count < MIN_POSITIVE_SAMPLES_FOR_ML:
             logger.warning(f"红球 {ball} 在训练数据中正样本({positive_count})不足或类别不平衡，无法训练可靠的分类器进行概率预测。跳过此球模型训练。")
             continue

        # Train LightGBM
        if LGBMClassifier is not None:
            try:
                # Create a pipeline for LGBM (optional but good practice, though less sensitive to scaling)
                # lgbm_pipeline = Pipeline([('scaler', StandardScaler()), ('lgbm', LGBMClassifier(**LGBM_PARAMS))])
                # Silence potential LightGBM stdout/stderr during training
                with SuppressOutput(suppress_stdout=True, capture_stderr=True):
                     lgbm_model = LGBMClassifier(**LGBM_PARAMS)
                     lgbm_model.fit(X, y_red) # Fit directly, scaling handled if needed before create_lagged_features
                # 这里引用了未定义的svc_pipeline变量，应该改为lgbm_model
                trained_models['red'][f'lgbm_{ball}'] = lgbm_model
            except Exception as e:
                logger.warning(f"警告: 训练红球 {ball} 的 LightGBM 模型失败: {e}")
        # else:
             # logger.info("LightGBM 未安装，跳过 LightGBM 模型训练。")
             # Logged once globally is enough


        # Train Logistic Regression
        try:
            # Create a pipeline for Logistic Regression including scaling
            logreg_pipeline = Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(**LOGISTIC_REG_PARAMS))])
            with SuppressOutput(suppress_stdout=True, capture_stderr=True):
                 logreg_pipeline.fit(X, y_red)
            trained_models['red'][f'logreg_{ball}'] = logreg_pipeline # Store the pipeline
        except Exception as e:
             logger.warning(f"警告: 训练红球 {ball} 的 Logistic Regression 模型失败: {e}")

        # Train SVC
        try:
            # Create a pipeline for SVC including scaling
            svc_pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC(**SVC_PARAMS))])
            
            # SVC training can be slow, potentially suppress output
            with SuppressOutput(suppress_stdout=True, capture_stderr=True):
                svc_pipeline.fit(X, y_red)
            
            # 首先确保SVC模型存在且有predict_proba方法
            if hasattr(svc_pipeline, 'predict_proba'):
                # 检查SVC模型是否启用了概率输出
                svc_estimator = svc_pipeline.named_steps.get('svc')
                if svc_estimator is not None and hasattr(svc_estimator, 'probability') and svc_estimator.probability:
                    trained_models['red'][f'svc_{ball}'] = svc_pipeline  # Store the pipeline
                else:
                    logger.warning(f"警告: 红球 {ball} 的 SVC 模型未启用概率预测。")
            else:
                logger.warning(f"警告: 红球 {ball} 的 SVC pipeline没有predict_proba方法。")
                
        except Exception as e:
            logger.warning(f"警告: 训练红球 {ball} 的 SVC 模型失败: {e}")


    # 蓝球模型
    for ball in BLUE_BALL_RANGE:
        # Create target variable y_blue: 1 if blue ball is the one for this period, 0 otherwise
        y_blue = (target_df['blue'] == ball).astype(int)

        if y_blue.empty or len(y_blue) != len(X):
            logger.warning(f"蓝球 {ball} 的目标变量Y ({len(y_blue)}) 与特征X ({len(X)}) 长度不匹配或为空，跳过训练。")
            continue

        # Check target variable balance
        positive_count = y_blue.sum()
        if len(y_blue.unique()) < 2 or positive_count < MIN_POSITIVE_SAMPLES_FOR_ML:
             logger.warning(f"蓝球 {ball} 在训练数据中正样本({positive_count})不足或类别不平衡，无法训练可靠的分类器进行概率预测。跳过此球模型训练。")
             continue


         # 训练 LightGBM
        if LGBMClassifier is not None:
            try:
                 with SuppressOutput(suppress_stdout=True, capture_stderr=True):
                     lgbm_model = LGBMClassifier(**LGBM_PARAMS)
                     lgbm_model.fit(X, y_blue) # Fit directly
                 trained_models['blue'][f'lgbm_{ball}'] = lgbm_model
            except Exception as e:
                 logger.warning(f"警告: 训练蓝球 {ball} 的 LightGBM 模型失败: {e}")

        # 训练 Logistic Regression
        try:
             logreg_pipeline = Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(**LOGISTIC_REG_PARAMS))])
             with SuppressOutput(suppress_stdout=True, capture_stderr=True):
                 logreg_pipeline.fit(X, y_blue)
             trained_models['blue'][f'logreg_{ball}'] = logreg_pipeline # Store the pipeline
        except Exception as e:
             logger.warning(f"警告: 训练蓝球 {ball} 的 Logistic Regression 模型失败: {e}")

        # Train SVC
        try:
            # Create a pipeline for SVC including scaling
            svc_pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC(**SVC_PARAMS))])
            
            # Suppress output during training
            with SuppressOutput(suppress_stdout=True, capture_stderr=True):
                svc_pipeline.fit(X, y_blue)
            
            # 首先确保SVC模型存在且有predict_proba方法
            if hasattr(svc_pipeline, 'predict_proba'):
                # 检查SVC模型是否启用了概率输出
                svc_estimator = svc_pipeline.named_steps.get('svc')
                if svc_estimator is not None and hasattr(svc_estimator, 'probability') and svc_estimator.probability:
                    trained_models['blue'][f'svc_{ball}'] = svc_pipeline  # Store the pipeline
                else:
                    logger.warning(f"警告: 蓝球 {ball} 的 SVC 模型未启用概率预测。")
            else:
                logger.warning(f"警告: 蓝球 {ball} 的 SVC pipeline没有predict_proba方法。")
                
        except Exception as e:
            logger.warning(f"警告: 训练蓝球 {ball} 的 SVC 模型失败: {e}")


    if not trained_models['red'] and not trained_models['blue']:
        logger.warning("没有模型成功训练。")
        return None

    trained_models['feature_cols'] = X.columns.tolist() # Store feature columns from X

    logger.info(f"ML模型训练完成。成功训练红球模型 {len(trained_models['red'])} 个，蓝球模型 {len(trained_models['blue'])} 个。")

    return trained_models


def predict_next_draw_probabilities(df_historical: pd.DataFrame, trained_models: Optional[dict], lags: List[int]) -> Dict:
    """使用训练好的ML模型和最新数据预测下一期单个号码出现的概率。"""
    predicted_probabilities = {
        'red': {},
        'blue': {}
    }

    if trained_models is None or df_historical is None or df_historical.empty:
        # logger.warning("ML models or historical data missing for probability prediction.")
        return predicted_probabilities # Return empty dict

    feature_cols = trained_models.get('feature_cols')
    if feature_cols is None or not feature_cols:
        logger.warning("Trained models missing feature_cols, cannot predict probabilities.")
        return predicted_probabilities

    max_lag = max(lags) if lags else 0
    if len(df_historical) < max_lag + 1:
        # logger.warning(f"Not enough history ({len(df_historical)} periods) to create prediction features (need at least {max_lag + 1}). Skipping ML prediction.")
        return predicted_probabilities

    # Create features for the *next* draw using the latest history
    # Need max_lag + 1 rows to compute the lagged features for the very last row
    df_latest_history_for_lagging = df_historical.tail(max_lag + 1).copy()
    # create_lagged_features now returns only the feature DataFrame X
    predict_X = create_lagged_features(df_latest_history_for_lagging, lags)

    if predict_X is None or predict_X.empty:
        logger.warning("Failed to prepare prediction features, cannot predict probabilities.")
        return predicted_probabilities

    # Ensure prediction features match training features and handle potential NaNs
    # The number of rows in predict_X should be exactly 1
    if len(predict_X) != 1:
         logger.error(f"Prediction feature DataFrame has {len(predict_X)} rows instead of 1. Cannot predict.")
         return predicted_probabilities

    try:
        predict_X = predict_X.reindex(columns=feature_cols, fill_value=0)
        if predict_X.isnull().values.any():
             logger.warning("NaNs found in prediction features after reindexing/filling. Attempting final fillna(0).")
             predict_X.fillna(0, inplace=True)
             if predict_X.isnull().values.any(): # Final check
                  logger.error("Prediction features still contain NaNs after final fillna. Cannot predict.")
                  return predicted_probabilities
    except Exception as e:
        logger.error(f"Error preparing prediction features (reindex/fillna): {e}. Cannot predict.")
        return predicted_probabilities

    # Predict probabilities for each red ball
    red_models = trained_models.get('red', {})
    for ball in RED_BALL_RANGE:
        lgbm_model = red_models.get(f'lgbm_{ball}')
        logreg_pipeline = red_models.get(f'logreg_{ball}') # Get LogReg pipeline
        svc_pipeline = red_models.get(f'svc_{ball}') # Get SVC pipeline

        predictions = [] # Store probability predictions from available models
        if lgbm_model:
            try:
                # predict_proba returns [[prob_class_0, prob_class_1]]
                predictions.append(lgbm_model.predict_proba(predict_X)[0][1])
            except Exception as e:
                # logger.warning(f"Warning: LGBM prediction for red ball {ball} failed: {e}") # Too noisy in backtest
                pass
        if logreg_pipeline: # Predict using LogReg pipeline if available
            try:
                 # Pipeline's predict_proba applies scaler internally
                 predictions.append(logreg_pipeline.predict_proba(predict_X)[0][1])
            except Exception as e:
                 # logger.warning(f"Warning: LogReg prediction for red ball {ball} failed: {e}") # Too noisy
                 pass
        if svc_pipeline: # Predict using SVC pipeline if available
            try:
                # Pipeline's predict_proba applies scaler internally
                 predictions.append(svc_pipeline.predict_proba(predict_X)[0][1])
            except Exception as e:
                 logger.warning(f"Warning: SVC prediction for red ball {ball} failed: {e}") # Log SVC prediction failures


        # Combine predictions (e.g., average)
        if predictions:
            predicted_probabilities['red'][ball] = np.mean(predictions)
        # If no models were available or succeeded, probability defaults to 0.0 (already initialized)


    # Predict probabilities for each blue ball
    blue_models = trained_models.get('blue', {})
    for ball in BLUE_BALL_RANGE:
        lgbm_model = blue_models.get(f'lgbm_{ball}')
        logreg_pipeline = blue_models.get(f'logreg_{ball}') # Get LogReg pipeline
        svc_pipeline = blue_models.get(f'svc_{ball}') # Get SVC pipeline


        predictions = []
        if lgbm_model:
            try:
                 predictions.append(lgbm_model.predict_proba(predict_X)[0][1])
            except Exception as e:
                 # logger.warning(f"Warning: LGBM prediction for blue ball {ball} failed: {e}") # Too noisy
                 pass
        if logreg_pipeline: # Predict using LogReg pipeline if available
            try:
                 predictions.append(logreg_pipeline.predict_proba(predict_X)[0][1])
            except Exception as e:
                 # logger.warning(f"Warning: LogReg prediction for blue ball {ball} failed: {e}") # Too noisy
                 pass
        if svc_pipeline: # Predict using SVC pipeline if available
            try:
                 predictions.append(svc_pipeline.predict_proba(predict_X)[0][1])
            except Exception as e:
                 logger.warning(f"Warning: SVC prediction for blue ball {ball} failed: {e}")


        # Combine predictions (e.g., average)
        if predictions:
            predicted_probabilities['blue'][ball] = np.mean(predictions)
        # If no models were available or succeeded, probability defaults to 0.0


    # Optional: Log top N probabilities for inspection
    # if predicted_probabilities.get('red') or predicted_probabilities.get('blue'):
    #     sorted_red_probs = sorted(predicted_probabilities.get('red', {}).items(), key=lambda item: item[1], reverse=True)
    #     sorted_blue_probs = sorted(predicted_probabilities.get('blue', {}).items(), key=lambda item: item[1], reverse=True)
    #     logger.info(f"Predicted Top 10 Red Probabilities: {sorted_red_probs[:10]}")
    #     logger.info(f"Predicted Top 5 Blue Probabilities: {sorted_blue_probs[:5]}")


    return predicted_probabilities


def calculate_scores(freq_omission_data: dict, pattern_analysis_data: dict, predicted_probabilities: dict) -> dict:
    """计算每个号码的综合得分，结合频率、遗漏、历史模式和ML预测概率。"""
    red_scores = {}
    blue_scores = {}

    # --- 评分因素 ---
    red_freq = freq_omission_data.get('red_freq', {})
    blue_freq = freq_omission_data.get('blue_freq', {})
    current_omission = freq_omission_data.get('current_omission', {})  # 任意位置
    average_interval = freq_omission_data.get('average_interval', {})  # 任意位置

    # 将频率转换为排名（处理freq数据为空的情况）
    # 使用所有可能的范围内数字创建一致排名的序列
    red_freq_series = pd.Series(red_freq).reindex(RED_BALL_RANGE, fill_value=0)  # 确保包含所有数字
    red_freq_rank = red_freq_series.rank(method='min', ascending=False)  # 排名1是最高频率

    blue_freq_series = pd.Series(blue_freq).reindex(BLUE_BALL_RANGE, fill_value=0)  # 确保包含所有数字
    blue_freq_rank = blue_freq_series.rank(method='min', ascending=False)

    # ML Predicted Probabilities
    red_pred_probs = predicted_probabilities.get('red', {})
    blue_pred_probs = predicted_probabilities.get('blue', {})


    # --- 评分公式 ---
    max_red_rank = len(RED_BALL_RANGE)
    max_blue_rank = len(BLUE_BALL_RANGE)

    # Weights are defined in Configuration section

    for num in RED_BALL_RANGE:
        # Factor 1: Frequency rank (inverted) - Higher frequency gets higher base score
        # Ensure num is in red_freq_rank index before getting rank, default to max_red_rank + 1 if not seen
        freq_rank_val = red_freq_rank.get(num, max_red_rank + 1)
        freq_score = (max_red_rank - (freq_rank_val - 1)) / max_red_rank * FREQ_SCORE_WEIGHT


        # Factor 2: Omission deviation (reward for being close to average)
        # Use .get with default values if num is not in current_omission or average_interval
        current_omit = current_omission.get(num, len(RED_BALL_RANGE) * 2) # Use a large default if never seen
        avg_int = average_interval.get(num, len(RED_BALL_RANGE) * 2)      # Use a large default if never seen
        dev = current_omit - avg_int
        omission_score = OMISSION_SCORE_WEIGHT * np.exp(-0.005 * dev**2) # Adjust decay rate if needed

        # Factor 3: ML Predicted Probability Score
        # Map probability (0 to 1) to a score component
        ml_prob = red_pred_probs.get(num, 0.0) # Default to 0 if no prediction
        ml_prob_score = ml_prob * ML_PROB_SCORE_WEIGHT_RED # Simple linear scaling


        # Combine factors
        red_scores[num] = freq_score + omission_score + ml_prob_score

    for num in BLUE_BALL_RANGE:
        # Factor 1: Frequency rank (inverted)
        # Ensure num is in blue_freq_rank index before getting rank
        freq_rank_val = blue_freq_rank.get(num, max_blue_rank + 1)
        freq_score = (max_blue_rank - (freq_rank_val - 1)) / max_blue_rank * BLUE_FREQ_SCORE_WEIGHT

        # Factor 2: Omission deviation
        current_omit = current_omission.get(num, len(BLUE_BALL_RANGE) * 2)
        avg_int = average_interval.get(num, len(BLUE_BALL_RANGE) * 2)
        dev = current_omit - avg_int
        omission_score = BLUE_OMISSION_SCORE_WEIGHT * np.exp(-0.01 * dev**2) # Adjust decay rate

        # Factor 3: ML Predicted Probability Score
        ml_prob = blue_pred_probs.get(num, 0.0)
        ml_prob_score = ml_prob * ML_PROB_SCORE_WEIGHT_BLUE

        # Combine factors
        blue_scores[num] = freq_score + omission_score + ml_prob_score

    # Normalize scores to a fixed range (e.g., 0-100)
    # Ensure there's at least one score to avoid division by zero
    all_scores = list(red_scores.values()) + list(blue_scores.values())
    if all_scores:
        min_score, max_score = min(all_scores), max(all_scores)
        # Add a small floating point comparison tolerance to handle cases where all scores are identical
        if (max_score - min_score) > 1e-9:
            red_scores = {num: (score - min_score) / (max_score - min_score) * 100 for num, score in red_scores.items()}
            blue_scores = {num: (score - min_score) / (max_score - min_score) * 100 for num, score in blue_scores.items()}
        else: # Handle case where all scores are very close or identical
             red_scores = {num: 50.0 for num in RED_BALL_RANGE}
             blue_scores = {num: 50.0 for num in BLUE_BALL_RANGE}
    else: # If all_scores is empty (e.g., no data), return default scores
        red_scores = {num: 50.0 for num in RED_BALL_RANGE}
        blue_scores = {num: 50.0 for num in BLUE_BALL_RANGE}


    return {'red_scores': red_scores, 'blue_scores': blue_scores}


def generate_combinations(scores_data: dict, pattern_analysis_data: dict, num_combinations: int = NUM_COMBINATIONS_TO_GENERATE) -> tuple[List[Dict], list[str]]:
    """基于分数和历史模式生成潜在组合。
       返回组合字典列表和用于输出的格式化字符串列表。
    """
    red_scores = scores_data.get('red_scores', {})
    blue_scores = scores_data.get('blue_scores', {})

    # Based on scores, select candidate pools
    # Ensure scores exist before sorting
    sorted_red_scores = sorted(red_scores.items(), key=lambda item: item[1], reverse=True) if red_scores else []
    # Select top N red candidates, fall back to full range if not enough scored balls
    red_candidate_pool = [num for num, score in sorted_red_scores[:TOP_N_RED_FOR_CANDIDATE]]
    if len(red_candidate_pool) < 6:
         red_candidate_pool = list(RED_BALL_RANGE)
         logger.warning(f"Not enough high-scoring red balls ({len(sorted_red_scores)}) for candidate pool. Using full range.")


    sorted_blue_scores = sorted(blue_scores.items(), key=lambda item: item[1], reverse=True) if blue_scores else []
    # Select top N blue candidates, fall back to full range if not enough scored balls
    blue_candidate_pool = [num for num, score in sorted_blue_scores[:TOP_N_BLUE_FOR_CANDIDATE]]
    if len(blue_candidate_pool) < 1:
         blue_candidate_pool = list(BLUE_BALL_RANGE)
         logger.warning(f"Not enough high-scoring blue balls ({len(sorted_blue_scores)}) for candidate pool. Using full range.")


    # Revised generation strategy: Generate a large pool from the candidates, score/rank, pick top N
    large_pool_size = num_combinations * 1000 # Increased pool size multiplier for more diversity
    if large_pool_size < 500: large_pool_size = 500 # Ensure minimum pool size

    generated_pool = []
    attempts = 0
    max_attempts_multiplier = 50 # Increased attempts multiplier relative to pool size
    max_attempts_pool = large_pool_size * max_attempts_multiplier

    # Calculate probabilities based on scores from the candidate pool
    # Ensure scores are non-negative and handle potential zero total weight
    red_weights = np.array([red_scores.get(num, 0) for num in red_candidate_pool])
    red_weights[red_weights < 0] = 0  # 确保权重非负
    total_red_weight = np.sum(red_weights)
    # Add a small epsilon to avoid division by zero
    red_probabilities = red_weights / (total_red_weight + 1e-9)
    # Re-normalize to sum to 1 in case epsilon caused deviation
    red_probabilities /= np.sum(red_probabilities) if np.sum(red_probabilities) > 1e-9 else 1.0


    blue_weights = np.array([blue_scores.get(num, 0) for num in blue_candidate_pool])
    blue_weights[blue_weights < 0] = 0
    total_blue_weight = np.sum(blue_weights)
    blue_probabilities = blue_weights / (total_blue_weight + 1e-9)
    blue_probabilities /= np.sum(blue_probabilities) if np.sum(blue_probabilities) > 1e-9 else 1.0


    # Ensure probabilities sum to exactly 1 (handle floating point inaccuracies)
    if len(red_probabilities) > 0: # Avoid index error on empty array
        red_probabilities[-1] = 1.0 - np.sum(red_probabilities[:-1])
    if len(blue_probabilities) > 0: # Avoid index error on empty array
        blue_probabilities[-1] = 1.0 - np.sum(blue_probabilities[:-1])

    # Check if probabilities are valid after adjustment (can happen with very few candidates or zero weights)
    if np.any(red_probabilities < 0) or not np.isclose(np.sum(red_probabilities), 1.0) or np.isnan(red_probabilities).any():
         logger.warning(f"Adjusted red probabilities are invalid. Using uniform probability. Sum: {np.sum(red_probabilities):.4f}, Negatives: {np.any(red_probabilities < 0)}.")
         red_probabilities = np.ones(len(red_candidate_pool)) / len(red_candidate_pool) if len(red_candidate_pool) > 0 else np.array([]) # Fallback to uniform
    if np.any(blue_probabilities < 0) or not np.isclose(np.sum(blue_probabilities), 1.0) or np.isnan(blue_probabilities).any():
         logger.warning(f"Adjusted blue probabilities are invalid. Using uniform probability. Sum: {np.sum(blue_probabilities):.4f}, Negatives: {np.any(blue_probabilities < 0)}.")
         blue_probabilities = np.ones(len(blue_candidate_pool)) / len(blue_candidate_pool) if len(blue_candidate_pool) > 0 else np.array([]) # Fallback to uniform


    while len(generated_pool) < large_pool_size and attempts < max_attempts_pool:
         attempts += 1
         try:
              # Check if candidate pools are large enough for sampling before attempting
              if len(red_candidate_pool) < 6:
                   logger.warning(f"Red candidate pool size {len(red_candidate_pool)} < 6. Cannot sample 6 distinct balls.")
                   break # Cannot generate combinations
              if len(blue_candidate_pool) < 1:
                   logger.warning(f"Blue candidate pool size {len(blue_candidate_pool)} < 1. Cannot sample 1 ball.")
                   break # Cannot generate combinations


              # Use calculated probabilities from candidate pool
              # Check probability validity before sampling
              if np.any(red_probabilities < 0) or not np.isclose(np.sum(red_probabilities), 1.0) or np.isnan(red_probabilities).any():
                   # Fallback to simple random if probabilities are bad
                   sampled_red_balls = sorted(random.sample(red_candidate_pool, 6))
                   # logger.warning(f"Invalid red probabilities detected during sampling attempt {attempts}. Using random.sample.") # Too noisy
              else:
                  sampled_red_balls = sorted(np.random.choice(
                      red_candidate_pool, size=6, replace=False, p=red_probabilities
                  ).tolist())

              if np.any(blue_probabilities < 0) or not np.isclose(np.sum(blue_probabilities), 1.0) or np.isnan(blue_probabilities).any():
                   # Fallback to simple random if probabilities are bad
                   sampled_blue_ball = random.choice(blue_candidate_pool)
                   # logger.warning(f"Invalid blue probabilities detected during sampling attempt {attempts}. Using random.choice.") # Too noisy
              else:
                  sampled_blue_ball = np.random.choice(
                       blue_candidate_pool, size=1, replace=False, p=blue_probabilities
                  ).tolist()[0]

              generated_pool.append({'red': sampled_red_balls, 'blue': sampled_blue_ball})

         except ValueError as e:
             # This might happen if probabilities are zero for all options in the pool or other sampling issues
             logger.warning(f"Probability sampling (np.random.choice) failed on attempt {attempts}: {e}. Red pool size: {len(red_candidate_pool)}, Blue pool size: {len(blue_candidate_pool)}. Red probs sum: {np.sum(red_probabilities):.4f}, Blue probs sum: {np.sum(blue_probabilities):.4f}. Falling back to random sample for this attempt.")
             try:
                  # Fallback to simple random sampling if np.random.choice fails
                  if len(red_candidate_pool) >= 6:
                     sampled_red_balls = sorted(random.sample(red_candidate_pool, 6))
                  else:
                      # Should be caught by pool size check before, but safety
                      sampled_red_balls = sorted(random.sample(list(RED_BALL_RANGE), 6))

                  if blue_candidate_pool:
                       sampled_blue_ball = random.choice(blue_candidate_pool)
                  else:
                       sampled_blue_ball = random.choice(list(BLUE_BALL_RANGE))

                  generated_pool.append({'red': sampled_red_balls, 'blue': sampled_blue_ball})
             except ValueError as e_fallback:
                 logger.error(f"Fallback random sampling failed on attempt {attempts}: {e_fallback}. Stopping combination generation attempts.")
                 break # If even fallback fails, give up
         except Exception as e:
             logger.warning(f"Unexpected error during combination sampling attempt {attempts}: {e}. Skipping attempt.")
             continue # Continue to next attempt

    if not generated_pool:
         logger.warning("没有生成组合。")
         return [], []  # 返回空列表

    # Now, score the generated combinations based on their ball scores AND how they fit HISTORICAL patterns
    scored_combinations = []

    # Get historical pattern tendencies for combination scoring bonus
    hist_most_common_odd_count = pattern_analysis_data.get('most_common_odd_even_count')
    hist_most_common_zone_dist = pattern_analysis_data.get('most_common_zone_distribution')
    # Check if counts exist before comparing
    blue_large_counts = pattern_analysis_data.get('blue_large_counts', {})
    hist_most_common_blue_large = blue_large_counts.get(True, 0) > blue_large_counts.get(False, 0) if blue_large_counts else None

    blue_odd_counts = pattern_analysis_data.get('blue_odd_counts', {})
    hist_most_common_blue_odd_val = blue_odd_counts.get(True, 0) > blue_odd_counts.get(False, 0) if blue_odd_counts else None


    # Determine if historical tendencies are available for bonus
    use_odd_count_tendency = hist_most_common_odd_count is not None
    use_blue_odd_tendency = hist_most_common_blue_odd_val is not None
    use_zone_dist_tendency = hist_most_common_zone_dist is not None
    use_blue_size_tendency = hist_most_common_blue_large is not None


    for combo in generated_pool:
        red_balls = combo['red']
        blue_ball = combo['blue']

        # Calculate combination's base score (sum of individual ball scores)
        # Use .get with default 0 if number not in scores data
        combo_score = sum(scores_data.get('red_scores', {}).get(r, 0) for r in red_balls) + scores_data.get('blue_scores', {}).get(blue_ball, 0)

        # Add bonus based on fitting HISTORICAL patterns
        feature_match_score = 0

        # Red ball odd count match
        if use_odd_count_tendency:
             actual_odd_count = sum(x % 2 != 0 for x in red_balls)
             if actual_odd_count == hist_most_common_odd_count:
                  feature_match_score += COMBINATION_ODD_COUNT_MATCH_BONUS

        # Blue ball odd/even match
        if use_blue_odd_tendency:
            actual_blue_is_odd = blue_ball % 2 != 0
            if actual_blue_is_odd == hist_most_common_blue_odd_val:
                feature_match_score += COMBINATION_BLUE_ODD_MATCH_BONUS

        # Zone distribution match
        if use_zone_dist_tendency:
             actual_zone_counts = [0, 0, 0]
             for ball in red_balls:
                 if RED_ZONES['Zone1'][0] <= ball <= RED_ZONES['Zone1'][1]: actual_zone_counts[0] += 1
                 elif RED_ZONES['Zone2'][0] <= ball <= RED_ZONES['Zone2'][1]: actual_zone_counts[1] += 1
                 elif RED_ZONES['Zone3'][0] <= ball <= RED_ZONES['Zone3'][1]: actual_zone_counts[2] += 1
             if tuple(actual_zone_counts) == hist_most_common_zone_dist:
                 feature_match_score += COMBINATION_ZONE_MATCH_BONUS

        # Blue ball size match
        if use_blue_size_tendency:
             is_large = blue_ball > 8
             if is_large == hist_most_common_blue_large:
                  feature_match_score += COMBINATION_BLUE_SIZE_MATCH_BONUS

        # Combine base score and feature match score
        total_combo_score = combo_score + feature_match_score

        scored_combinations.append({'combination': combo, 'score': total_combo_score})

    # Sort combinations by score and select the top N
    scored_combinations.sort(key=lambda x: x['score'], reverse=True)
    final_recommendations_data = scored_combinations[:num_combinations]

    # --- Format output strings ---
    output_strings = []
    output_strings.append("推荐组合:")
    if final_recommendations_data:
         for i, rec in enumerate(final_recommendations_data):
             output_strings.append(f"组合 {i+1}: 红球 {sorted(rec['combination']['red'])} 蓝球 {rec['combination']['blue']} (分数: {rec['score']:.2f})")
    else:
         output_strings.append("无法生成推荐组合。请检查数据和配置。")

    # Return list of combination dicts AND formatted output strings
    return final_recommendations_data, output_strings

# --- 核心分析和推荐函数 (新) ---
# 此函数封装了给定数据集切片的主要逻辑流
def analyze_and_recommend(
    df_historical: pd.DataFrame,
    lags: List[int],
    num_combinations: int,
    train_ml: bool = True,  # Whether to train ML models in this run
    existing_models: Optional[Dict] = None  # Pass existing models if train_ml is False
) -> tuple[List[Dict], list[str], dict, Optional[Dict]]:
    """
    基于提供的历史数据执行分析，训练/预测ML概率，计算分数，并为下一期生成组合。
    可选地训练ML模型或使用现有的模型。

    返回: (recommendations_data, recommendations_strings, analysis_data, trained_models)
    """
    if df_historical is None or df_historical.empty:
        logger.error("没有提供用于分析和推荐的历史数据。")
        return [], [], {}, None

    # 1. 执行历史分析（频率、遗漏、模式）
    # 此分析基于提供的df_historical切片
    freq_omission_data = analyze_frequency_omission(df_historical)
    pattern_analysis_data = analyze_patterns(df_historical)
    # 关联规则分析暂未在评分中使用
    # association_rules_data = analyze_associations(df_historical, ARM_MIN_SUPPORT, ARM_MIN_CONFIDENCE, ARM_MIN_LIFT)

    analysis_data = {
        'freq_omission': freq_omission_data,
        'patterns': pattern_analysis_data,
        # 'association_rules': association_rules_data # 如果以后需要，可以包含
    }

    # 2. 训练/预测 ML 模型 (预测单个号码概率)
    current_trained_models = None
    predicted_probabilities = {} # Output for ML predicted probabilities

    # Need enough historical data to create lagged features and train ML models
    max_lag = max(lags) if lags else 1
    min_periods_for_ml = max_lag + 1 # Need this many periods to get the first row of X and its corresponding Y

    if len(df_historical) >= min_periods_for_ml:
         if train_ml:
             # Train ML models on the provided historical data
             current_trained_models = train_prediction_models(df_historical, lags)
             # Check if any models were successfully trained
             if current_trained_models and (current_trained_models.get('red', {}) or current_trained_models.get('blue', {})):
                 # Use the newly trained models and the latest history to predict probabilities
                 predicted_probabilities = predict_next_draw_probabilities(df_historical, current_trained_models, lags)
                 if not predicted_probabilities.get('red') and not predicted_probabilities.get('blue'):
                     logger.warning("ML prediction using newly trained models failed. Will not use ML probabilities for scoring.")
             else:
                  logger.warning("ML model training failed or no models were successfully trained. Will not use ML probabilities for scoring.")

         elif existing_models:
             # Use provided trained models to predict probabilities on the latest data
             current_trained_models = existing_models # Use the provided models
             # Check if existing_models actually contains models
             if current_trained_models and (current_trained_models.get('red', {}) or current_trained_models.get('blue', {})):
                  predicted_probabilities = predict_next_draw_probabilities(df_historical, current_trained_models, lags)
                  if not predicted_probabilities.get('red') and not predicted_probabilities.get('blue'):
                      logger.warning("ML prediction using existing models failed. Will not use ML probabilities for scoring.")
             else:
                  logger.warning("No valid existing ML models provided. Will not use ML probabilities for scoring.")
    else:
        logger.warning(f"Not enough historical data ({len(df_historical)} periods) for ML training/prediction (need at least {min_periods_for_ml}). Will not use ML probabilities for scoring.")


    # 3. 计算号码分数（结合历史分析和ML预测概率）
    scores_data = calculate_scores(
        freq_omission_data,
        pattern_analysis_data, # Still pass for combination bonus based on historical patterns
        predicted_probabilities # Pass ML probabilities (might be empty if ML failed)
    )

    # 4. 基于分数和历史模式生成组合
    # Note: generate_combinations now uses scores based on ML probabilities.
    # The pattern_analysis_data is still passed for the combination-level bonus based on historical patterns.
    recommendations_data, recommendations_strings = generate_combinations(
        scores_data,
        pattern_analysis_data,  # Pass historical patterns for combination bonus
        num_combinations=num_combinations
    )

    # Return combinations, output strings, analysis data, and the trained models
    return recommendations_data, recommendations_strings, analysis_data, current_trained_models


# --- 验证、回测与持续优化 ---

def backtest(df: pd.DataFrame, lags: List[int], num_combinations_per_period: int, backtest_periods_count: int) -> pd.DataFrame:
    """
    在历史数据上执行回测，包括重新训练ML模型并预测概率。
    """
    logger.info("\n" + "="*50)
    logger.info(" 开始回测 ")
    logger.info("="*50)

    # Determine the minimum number of periods needed for the initial training data
    # Need enough periods to create lagged features for the *first* backtest period's target
    max_lag = max(lags) if lags else 0
    min_periods_for_initial_training = max_lag + 1 # Need this many periods to get the first row of X and its corresponding Y

    if len(df) < min_periods_for_initial_training + 1: # Need min_periods for training + 1 period to predict
         logger.warning(f"数据不足({len(df)})，无法进行回测(至少需要{min_periods_for_initial_training + 1}期)。跳过回测。")
         logger.info("="*50)
         logger.info(" 回测已跳过 ")
         logger.info("="*50)
         return pd.DataFrame()

    # Determine the range of periods to backtest
    # The *first* period we make a prediction FOR will be at index min_periods_for_initial_training
    # The last period we make a prediction FOR will be the last period in the df (index len(df)-1)
    start_prediction_index = min_periods_for_initial_training
    end_prediction_index = len(df) - 1

    if start_prediction_index > end_prediction_index:
         logger.warning(f"没有足够的后续数据进行回测预测。开始预测索引: {start_prediction_index}, 结束索引: {end_prediction_index}。跳过回测。")
         logger.info("="*50)
         logger.info(" 回测已跳过 ")
         logger.info("="*50)
         return pd.DataFrame()

    # Adjust the actual number of periods to backtest based on available data and requested count
    available_backtest_periods = end_prediction_index - start_prediction_index + 1
    actual_backtest_periods_count = min(backtest_periods_count, available_backtest_periods)

    # Calculate the index of the *first* period whose prediction results we will evaluate
    # We predict FOR periods from start_prediction_index up to end_prediction_index
    # So the first period we evaluate is end_prediction_index - actual_backtest_periods_count + 1
    backtest_evaluation_start_index = end_prediction_index - actual_backtest_periods_count + 1

    # Ensure the evaluation starts at or after the earliest possible prediction index
    if backtest_evaluation_start_index < start_prediction_index:
         backtest_evaluation_start_index = start_prediction_index


    # Get the actual period numbers for the backtest evaluation range
    start_period_number = df.loc[backtest_evaluation_start_index, '期号'] if backtest_evaluation_start_index < len(df) else "N/A"
    end_period_number = df.loc[end_prediction_index, '期号'] if end_prediction_index < len(df) else "N/A"


    logger.info(f"回测 {actual_backtest_periods_count} 期的预测结果，期号范围: {start_period_number} 至 {end_period_number}")
    logger.info(f"预测对象期索引范围: {backtest_evaluation_start_index} 至 {end_prediction_index}。")
    logger.info(f"使用直到索引 {backtest_evaluation_start_index - 1} 的数据进行第一次预测的分析/训练。")


    results = []
    red_cols = [f'red{i+1}' for i in range(6)]

    # Display initial progress bar - only to console, not report file
    total_steps = end_prediction_index - backtest_evaluation_start_index + 1
    
    # Save the current stdout before redirecting
    original_stdout = sys.stdout

    # Iterate through periods to predict FOR
    for i in range(backtest_evaluation_start_index, end_prediction_index + 1):
        # Update progress bar - restore original stdout to show progress, then switch back
        current_progress = i - backtest_evaluation_start_index + 1

        # Temporarily switch stdout to the original console stdout to display the progress bar
        sys.stdout = sys.__stdout__
        show_progress(current_progress, total_steps, prefix='回测进度:', suffix='完成', length=50)
        # Restore stdout to its previous state (which might be redirected to a file in the main program)
        sys.stdout = original_stdout

        # Data available for training/analysis for predicting period i
        train_data = df.iloc[:i].copy()

        if train_data.empty or len(train_data) < (max(lags) if lags else 0) + 1: # Ensure enough data to create features
             logger.warning(f"训练数据不足({len(train_data)}行)用于预测期索引{i}。跳过此期预测。")
             continue

        # The period we are predicting FOR is at index i
        actual_row_index = i
        # Ensure the actual row exists in the dataframe
        if actual_row_index not in df.index:
             logger.error(f"DataFrame中找不到期索引{actual_row_index}(期号: {actual_period})的实际结果。跳过此期预测。")
             continue

        actual_period = df.loc[actual_row_index, '期号']

        try:
            actual_red = set(df.loc[actual_row_index, red_cols].tolist())
            actual_blue = df.loc[actual_row_index, 'blue']
        except KeyError as e:
             logger.error(f"期索引{actual_row_index}(期号: {actual_period})的实际结果中缺少红球或蓝球数据: {e}。跳过此期预测。")
             continue
        except ValueError as e:
             logger.error(f"期索引{actual_row_index}(期号: {actual_period})的实际结果中红球或蓝球数据格式错误: {e}。跳过此期预测。")
             continue


        # Perform analysis, train ML models, and generate combinations
        # Use SuppressOutput to hide analyze_and_recommend's internal printing/logging during backtest
        with SuppressOutput(suppress_stdout=True, capture_stderr=True):
             # Pass train_ml=True in backtest to force retraining models for each period
             predicted_combinations_data, predicted_combinations_strings, analysis_data, trained_models_this_period = analyze_and_recommend(
                 train_data, # Train on data BEFORE the target period
                 lags,
                 num_combinations=num_combinations_per_period,
                 train_ml=True # Retrain ML models for each backtest period
             )


        if predicted_combinations_data:
            for combo_info in predicted_combinations_data:
                predicted_red = set(combo_info['combination']['red'])
                predicted_blue = combo_info['combination']['blue']

                red_hits = len(predicted_red.intersection(actual_red))
                blue_hit = (predicted_blue == actual_blue)

                results.append({
                    'period': actual_period,
                    'predicted_red': sorted(list(predicted_red)),
                    'predicted_blue': predicted_blue,
                    'actual_red': sorted(list(actual_red)),
                    'actual_blue': actual_blue,
                    'red_hits': red_hits,
                    'blue_hit': blue_hit,
                    'combination_score': combo_info['score']
                })
        # else:
            # logger.warning(f"未为期 {actual_period} 生成组合。") # Avoid noise in backtest log

    # Display the final completed progress bar on the console
    sys.stdout = sys.__stdout__ # Ensure final progress is shown on the console
    show_progress(total_steps, total_steps, prefix='回测进度:', suffix='完成', length=50)
    # Restore redirection
    sys.stdout = original_stdout


    logger.info("\n" + "="*50)
    logger.info(" 回测完成 ")
    logger.info("="*50)

    if not results:
        logger.warning("未记录回测结果。")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Add backtest period number range to the results DataFrame attributes for use in the report
    results_df.attrs['start_period'] = start_period_number
    results_df.attrs['end_period'] = end_period_number

    return results_df


# --- 绘图函数（移到分析函数之外）---
def plot_analysis_results(freq_omission_data: dict, pattern_analysis_data: dict):
     """从分析结果生成图表。"""
     if not SHOW_PLOTS:
          # logger.info("Plotting is disabled.") # Avoid noise if running multiple times
          plt.close('all')  # Close any lingering figures
          return

     logger.info("生成图表...")

     # Check if data is available before plotting
     if not freq_omission_data or not pattern_analysis_data:
          logger.warning("Analysis data not available for plotting.")
          return

     # Frequency plots
     red_freq = freq_omission_data.get('red_freq', {})
     blue_freq = freq_omission_data.get('blue_freq', {})
     red_pos_freq = freq_omission_data.get('red_pos_freq', {})
     # Assuming red_pos_cols keys exist in red_pos_freq structure if data is present
     red_pos_cols = [f'red_pos{i+1}' for i in range(6)]


     # Check if there is any data in frequencies before plotting
     has_freq_data = bool(red_freq) or bool(blue_freq) or any(red_pos_freq.values())
     if has_freq_data:
         plt.figure(figsize=(14, 6))
         subplot_count = 0
         if red_freq:
             subplot_count += 1
             plt.subplot(1, 2, subplot_count)
             # Sort by number for consistent plotting
             sorted_red_items = sorted(red_freq.items())
             sns.barplot(x=[item[0] for item in sorted_red_items], y=[item[1] for item in sorted_red_items])
             plt.title('红球总体频率')
             plt.xlabel('数字'); plt.ylabel('频率')
         if blue_freq:
             subplot_count += 1
             plt.subplot(1, 2, subplot_count)
             # Sort by number for consistent plotting
             sorted_blue_items = sorted(blue_freq.items())
             sns.barplot(x=[item[0] for item in sorted_blue_items], y=[item[1] for item in sorted_blue_items])
             plt.title('蓝球频率')
             plt.xlabel('数字'); plt.ylabel('频率')
         if subplot_count > 0:
             plt.tight_layout()
             plt.show()
         else:
             plt.close() # Close empty figure if no data was plotted


     # Positional red ball frequency plot
     if red_pos_freq and any(red_pos_freq.values()): # Check red_pos_freq is not empty and has data
          # Determine how many subplots are actually needed (based on available columns)
          valid_pos_cols = [col for col in red_pos_cols if col in red_pos_freq and red_pos_freq[col]]
          if valid_pos_cols:
              n_cols = 3 # Number of columns in subplot grid
              n_rows = (len(valid_pos_cols) + n_cols - 1) // n_cols # Calculate rows needed
              fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
              axes = axes.flatten() # Flatten for easy iteration

              for i, col in enumerate(valid_pos_cols):
                   # Sort keys to maintain consistent plotting order
                   sorted_freq_items = sorted(red_pos_freq[col].items())
                   sns.barplot(x=[item[0] for item in sorted_freq_items], y=[item[1] for item in sorted_freq_items], ax=axes[i])
                   axes[i].set_title(f'红球位置 {col.replace("red_pos", "")} 频率')
                   axes[i].set_xlabel('数字')
                   axes[i].set_ylabel('频率')

              # Hide any unused subplots
              for j in range(len(valid_pos_cols), len(axes)):
                  fig.delaxes(axes[j])

              plt.tight_layout()
              plt.show()
          else:
              logger.warning("No valid position frequency data found for plotting.")


     # Pattern distribution plots (sum, span, odd/even, consecutive, repeat)
     # Example: Plotting Odd/Even ratio distribution if data is available
     odd_even_ratios = pattern_analysis_data.get('odd_even_ratios', {})
     if odd_even_ratios:
          plt.figure(figsize=(8, 5)) # Adjusted size
          # Sort x-axis labels numerically if possible (e.g., "0:6", "1:5", ...)
          sorted_odd_ratios = sorted(odd_even_ratios.items(), key=lambda item: int(item[0].split(':')[0]))
          sns.barplot(x=[item[0] for item in sorted_odd_ratios], y=[item[1] for item in sorted_odd_ratios])
          plt.title('红球奇:偶比分布')
          plt.xlabel('奇:偶比'); plt.ylabel('频率'); plt.show()

     # Example: Plotting Consecutive pairs distribution if data is available
     consecutive_counts = pattern_analysis_data.get('consecutive_counts', {})
     if consecutive_counts:
          plt.figure(figsize=(8, 5)) # Adjusted size
          # Sort by number of consecutive pairs
          sorted_consecutive = sorted(consecutive_counts.items())
          sns.barplot(x=[item[0] for item in sorted_consecutive], y=[item[1] for item in sorted_consecutive])
          plt.title('红球连续对分布')
          plt.xlabel('连续对数量'); plt.ylabel('频率'); plt.show()

     # Example: Plotting Repeat counts distribution if data is available
     repeat_counts = pattern_analysis_data.get('repeat_counts', {})
     if repeat_counts:
          plt.figure(figsize=(8, 5)) # Adjusted size
          # Sort by repeat count
          sorted_repeat = sorted(repeat_counts.items())
          sns.barplot(x=[item[0] for item in sorted_repeat], y=[item[1] for item in sorted_repeat])
          plt.title('红球从上期重复频率')
          plt.xlabel('重复球数量'); plt.ylabel('频率'); plt.show()

     # Add plots for blue ball patterns if data is available
     blue_odd_counts = pattern_analysis_data.get('blue_odd_counts', {})
     if blue_odd_counts:
         plt.figure(figsize=(6, 4))
         # Map True/False to labels
         labels = [str(k) for k in blue_odd_counts.keys()] # Use boolean string initially
         counts = list(blue_odd_counts.values())
         # Better labels for plot
         display_labels = ['奇数' if k else '偶数' for k in blue_odd_counts.keys()]
         sns.barplot(x=display_labels, y=counts)
         plt.title('蓝球奇偶分布')
         plt.xlabel('奇偶性'); plt.ylabel('频率'); plt.show()

     blue_large_counts = pattern_analysis_data.get('blue_large_counts', {})
     if blue_large_counts:
         plt.figure(figsize=(6, 4))
         # Map True/False to labels
         labels = [str(k) for k in blue_large_counts.keys()]
         counts = list(blue_large_counts.values())
         # Better labels for plot
         display_labels = ['大 (>8)' if k else '小 (1-8)' for k in blue_large_counts.keys()]
         sns.barplot(x=display_labels, y=counts)
         plt.title('蓝球大小分布')
         plt.xlabel('大小'); plt.ylabel('频率'); plt.show()

     # blue_prime_counts = pattern_analysis_data.get('blue_prime_counts', {})
     # if blue_prime_counts:
     #     plt.figure(figsize=(6, 4))
     #     labels = [str(k) for k in blue_prime_counts.keys()]
     #     counts = list(blue_prime_counts.values())
     #     display_labels = ['质数' if k else '合数' for k in blue_prime_counts.keys()]
     #     sns.barplot(x=display_labels, y=counts)
     #     plt.title('蓝球质数分布')
     #     plt.xlabel('类型'); plt.ylabel('频率'); plt.show()


     logger.info("图表生成完成。")


# --- 主执行流程 ---

if __name__ == "__main__":
    # --- 配置输出文件 ---
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(SCRIPT_DIR, f"ssq_analysis_output_{timestamp}.txt")

    output_file = None  # Initialize file handle
    original_stdout = sys.stdout # Store original stdout

    try:
        # Open the output file
        output_file = open(output_filename, 'w', encoding='utf-8')
        # Redirect stdout to the output file for the main report content
        sys.stdout = output_file

        print(f"--- 双色球分析报告 ---", file=sys.stdout)
        print(f"运行日期: {now.strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stdout)
        print(f"输出文件: {output_filename}", file=sys.stdout)
        print("-" * 30, file=sys.stdout)
        print("\n", file=sys.stdout)

        # --- Check for LightGBM availability early ---
        if LGBMClassifier is None:
            logger.warning("LightGBM library not found (pip install lightgbm). ML models will be limited to Logistic Regression and SVC.")
            print("\n警告: LightGBM 库未找到，ML 模型将仅限于 Logistic Regression 和 SVC。", file=sys.stdout)


        # --- Start Analysis and Prediction ---

        # 1. Load and prepare data
        # Try loading processed data file first
        processed_data_exists = os.path.exists(PROCESSED_CSV_PATH)
        df = None # Initialize df

        if processed_data_exists:
            logger.info(f"尝试加载处理好的数据文件: {PROCESSED_CSV_PATH}")
            df = load_data(PROCESSED_CSV_PATH)
            if df is not None and not df.empty:
                logger.info("成功加载处理好的数据文件。跳过清洗和特征工程。")
                # Assume processed file already has features, if not, rerun feature_engineer
                # A robust check would be to see if feature columns exist, but for simplicity,
                # assume it's ready or needs reprocessing if the original wasn't used.
                # If you change feature engineering, you might need to force reprocessing.
            else:
                logger.warning("处理好的数据文件加载失败或为空。尝试加载原始数据文件并重新处理。")
                processed_data_exists = False

        # If processed data not available or failed to load, use raw data and process
        if not processed_data_exists:
            logger.info(f"加载原始数据文件: {CSV_FILE_PATH}")
            # Use SuppressOutput to capture stderr during load/clean/feature_engineer if needed
            with SuppressOutput(suppress_stdout=False, capture_stderr=True): # Keep stdout for initial load message if not redirected
                 df = load_data(CSV_FILE_PATH)
                 if df is not None and not df.empty:
                     logger.info("开始数据清洗和结构化...")
                     df = clean_and_structure(df)
                     if df is not None and not df.empty:
                          logger.info("开始特征工程...")
                          df = feature_engineer(df)
                          if df is None or df.empty:
                               logger.error("特征工程失败或导致空数据。")
                               print("\n错误: 特征工程失败。无法继续分析。", file=sys.stdout)
                          else:
                               logger.info("数据清洗和特征工程完成。")
                               # Optional: Save processed data for faster loading next time
                               try:
                                    df.to_csv(PROCESSED_CSV_PATH, index=False)
                                    logger.info(f"处理后的数据已保存到 {PROCESSED_CSV_PATH}")
                               except Exception as e:
                                    logger.warning(f"保存处理后的数据失败: {e}")

                     else:
                         logger.error("数据清洗和结构化失败或导致空数据。")
                         print("\n错误: 数据清洗和结构化失败。无法继续分析。", file=sys.stdout)
                 else:
                     logger.error("数据加载失败或导致空数据。")
                     print("\n错误: 数据加载失败。无法继续分析。", file=sys.stdout)


        # Proceed only if data is successfully loaded and processed
        if df is not None and not df.empty:
            # Extract data range info and last period/date
            min_period = df['期号'].min() if '期号' in df.columns and not df['期号'].empty else "N/A"
            max_period = df['期号'].max() if '期号' in df.columns and not df['期号'].empty else "N/A"
            total_periods = len(df)

            last_period = "未知"
            last_drawing_date = "未知"

            if not df.empty:
                last_row = df.iloc[-1]
                if '期号' in last_row and pd.notna(last_row['期号']):
                    last_period = int(last_row['期号']) # Ensure int
                if '日期' in last_row and pd.notna(last_row['日期']):
                    last_drawing_date = last_row['日期']

            # Add data range info and last date to the report header
            print(f"\n数据概况:", file=sys.stdout)
            print(f"  数据期数范围: 第 {min_period} 期 至 第 {max_period} 期", file=sys.stdout)
            print(f"  总数据条数: {total_periods} 期", file=sys.stdout)
            print(f"  最后日期: {last_drawing_date}({last_period})", file=sys.stdout)
            print("\n", file=sys.stdout)


            # Check if enough data for analysis/ML
            max_lag = max(ML_LAG_FEATURES) if ML_LAG_FEATURES else 0
            min_periods_needed_for_ml = max_lag + 1 # Need enough for at least one set of features/target
            min_periods_needed_for_analysis = 10 # Basic analysis needs a few periods

            if len(df) < min(min_periods_needed_for_ml, min_periods_needed_for_analysis):
                 logger.error(f"清理/特征工程后有效期数不足({len(df)})，无法进行完整分析(至少需要{min(min_periods_needed_for_ml, min_periods_needed_for_analysis)})。")
                 print(f"\n错误: 清理/特征工程后有效期数不足({len(df)})，无法进行分析。无法继续。", file=sys.stdout)
            else:
                # 2. Perform Full Historical Analysis
                print("\n" + "="*50, file=sys.stdout)
                print(" 完整历史分析 ", file=sys.stdout)
                print(f" (基于第 {min_period} 期至第 {max_period} 期数据) ", file=sys.stdout)
                print("="*50, file=sys.stdout)
                # Use SuppressOutput to hide internal analysis function printing from file, but still log stderr
                with SuppressOutput(suppress_stdout=True, capture_stderr=True):
                    full_freq_omission_data = analyze_frequency_omission(df)
                    full_pattern_analysis_data = analyze_patterns(df)
                    full_association_rules = analyze_associations(df, ARM_MIN_SUPPORT, ARM_MIN_CONFIDENCE, ARM_MIN_LIFT) # Analyze associations on full data

                print("\n历史分析摘要（基于完整数据）:", file=sys.stdout)
                print("\n频率和遗漏亮点:", file=sys.stdout)
                # Print selected frequency/omission data to file
                print(f"  热门红球: {full_freq_omission_data.get('hot_reds', [])}", file=sys.stdout)
                print(f"  冷门红球: {full_freq_omission_data.get('cold_reds', [])}", file=sys.stdout)
                print(f"  热门蓝球: {full_freq_omission_data.get('hot_blues', [])}", file=sys.stdout)
                print(f"  冷门蓝球: {full_freq_omission_data.get('cold_blues', [])}", file=sys.stdout)
                print("\n模式分析亮点:", file=sys.stdout)
                print(f"  最常见红球奇:偶数量: {full_pattern_analysis_data.get('most_common_odd_even_count')}", file=sys.stdout)
                print(f"  最常见区域分布(区域1:区域2:区域3): {full_pattern_analysis_data.get('most_common_zone_distribution')}", file=sys.stdout)
                print(f"  最常见红球和: {full_pattern_analysis_data.get('most_common_sum')}", file=sys.stdout)
                print(f"  最常见红球跨度: {full_pattern_analysis_data.get('most_common_span')}", file=sys.stdout)

                if not full_association_rules.empty:
                    print("\n前10条关联规则（按提升度）:", file=sys.stdout)
                    # Format rules for output to file
                    # Ensure antecedents and consequents are sets of standard types for printing
                    formatted_rules = full_association_rules.head(10).copy()
                    formatted_rules['antecedents'] = formatted_rules['antecedents'].apply(lambda x: set(x))
                    formatted_rules['consequents'] = formatted_rules['consequents'].apply(lambda x: set(x))

                    for _, rule in formatted_rules.iterrows():
                        print(f"  {rule['antecedents']} -> {rule['consequents']} (支持度: {rule['support']:.4f}, 置信度: {rule['confidence']:.2f}, 提升度: {rule['lift']:.2f})", file=sys.stdout)
                else:
                    print("\n以当前阈值未找到显著关联规则。", file=sys.stdout)


                print("="*50, file=sys.stdout)
                print(" 历史分析完成 ", file=sys.stdout)
                print("="*50, file=sys.stdout)

                 # 3. Perform Backtest
                # Backtest output is logged to console and summarized in file
                # The backtest function itself handles its console output (progress bar) and internal logging
                backtest_results = backtest(df, ML_LAG_FEATURES, NUM_COMBINATIONS_TO_GENERATE, BACKTEST_PERIODS_COUNT)

                print("\n" + "="*50, file=sys.stdout)
                print(" 回测摘要 ", file=sys.stdout)

                # If backtest results exist, show the period range tested
                if not backtest_results.empty:
                    start_period = backtest_results.attrs.get('start_period', '未知')
                    end_period = backtest_results.attrs.get('end_period', '未知')
                    print(f" (基于第 {start_period} 期至第 {end_period} 期数据) ", file=sys.stdout)

                print("="*50, file=sys.stdout)
                if not backtest_results.empty:
                     # Print backtest summary to file
                     periods_with_results = backtest_results['period'].nunique()
                     print(f"测试的总期数(已生成组合): {periods_with_results}", file=sys.stdout)
                     print(f"生成的总组合数: {len(backtest_results)}", file=sys.stdout)
                     if periods_with_results > 0:
                          print(f"每期生成的组合数(平均): {len(backtest_results) / periods_with_results:.2f}", file=sys.stdout)


                     avg_red_hits = backtest_results['red_hits'].mean()
                     print(f"每个组合的平均红球命中数: {avg_red_hits:.2f}", file=sys.stdout)

                     # Calculate percentage of test periods where at least one combination hit the blue ball
                     blue_hit_by_period = backtest_results.groupby('period')['blue_hit'].any()
                     blue_hit_rate_per_period = blue_hit_by_period.mean() if not blue_hit_by_period.empty else 0.0
                     print(f"至少一个组合击中蓝球的测试期百分比: {blue_hit_rate_per_period:.2%}", file=sys.stdout)


                     print("\n中奖层级命中情况(每个组合):", file=sys.stdout)
                     print(f"  6红+蓝: {len(backtest_results[(backtest_results['red_hits'] == 6) & (backtest_results['blue_hit'] == True)])}", file=sys.stdout)
                     print(f"  6红(无蓝): {len(backtest_results[(backtest_results['red_hits'] == 6) & (backtest_results['blue_hit'] == False)])}", file=sys.stdout)
                     print(f"  5红+蓝: {len(backtest_results[(backtest_results['red_hits'] == 5) & (backtest_results['blue_hit'] == True)])}", file=sys.stdout)
                     print(f"  5红(无蓝): {len(backtest_results[(backtest_results['red_hits'] == 5) & (backtest_results['blue_hit'] == False)])}", file=sys.stdout)
                     print(f"  4红+蓝: {len(backtest_results[(backtest_results['red_hits'] == 4) & (backtest_results['blue_hit'] == True)])}", file=sys.stdout)
                     print(f"  4红(无蓝): {len(backtest_results[(backtest_results['red_hits'] == 4) & (backtest_results['blue_hit'] == False)])}", file=sys.stdout)
                     print(f"  3红+蓝: {len(backtest_results[(backtest_results['red_hits'] == 3) & (backtest_results['blue_hit'] == True)])}", file=sys.stdout)
                     # Sum of all combinations where blue_hit is True, regardless of red hits
                     print(f"  精确蓝球命中(任意红球数): {(backtest_results['blue_hit'] == True).sum()}", file=sys.stdout)

                     print("\n与随机机会比较(近似):", file=sys.stdout)
                     expected_avg_red_hits_random = 6 * (6/33.0)
                     expected_blue_hits_random = 1/16.0
                     print(f"  纯随机每组合的期望平均红球命中数: ~{expected_avg_red_hits_random:.2f}", file=sys.stdout)
                     print(f"  纯随机每组合的期望蓝球命中数: ~{expected_blue_hits_random:.4f}", file=sys.stdout)


                else:
                     print("没有可总结的回测结果。", file=sys.stdout)

                print("="*50, file=sys.stdout)
                print(" 回测摘要完成 ", file=sys.stdout)
                print("="*50, file=sys.stdout)


                # 4. Generate Final Recommendation Combinations for the next draw
                print("\n" + "="*50, file=sys.stdout)
                print(" 生成最终推荐 ", file=sys.stdout)
                print(f" (基于第 {min_period} 期至第 {max_period} 期全部数据) ", file=sys.stdout)
                print("="*50, file=sys.stdout)

                # Use SuppressOutput to hide analyze_and_recommend's internal printing/logging
                # The final recommendation strings will be printed explicitly below
                with SuppressOutput(suppress_stdout=True, capture_stderr=True):
                     # Train ML models on the full dataset for the final prediction
                     final_recommendations_data, final_recommendations_strings, final_analysis_data, final_trained_models = analyze_and_recommend(
                         df,  # Use the full available data for final prediction
                         ML_LAG_FEATURES,
                         NUM_COMBINATIONS_TO_GENERATE,
                         train_ml=True # Train ML on full data for final recommendation
                     )

                # Print the final recommendation combinations to the output file
                if final_recommendations_strings:
                    for line in final_recommendations_strings:
                        print(line, file=sys.stdout)
                else:
                    print("无法生成最终推荐组合。请检查数据和配置。", file=sys.stdout)


                print("="*50, file=sys.stdout)
                print(" 最终推荐完成 ", file=sys.stdout)
                print("="*50, file=sys.stdout)

                # 5. Generate 7+7 Multi-bet Selection
                print("\n" + "="*50, file=sys.stdout)
                print(" 7+7复式选号 ", file=sys.stdout)
                print("="*50, file=sys.stdout)

                # Recalculate scores for 7+7 selection using analysis and ML results from the final run
                # Need to get the predicted probabilities from the final trained models on the full data
                final_predicted_probabilities = {}
                if final_trained_models:
                     final_predicted_probabilities = predict_next_draw_probabilities(df, final_trained_models, ML_LAG_FEATURES)
                # If ML training failed or no models trained, final_predicted_probabilities will be empty,
                # and calculate_scores will use default 0 probability scores.


                final_scores_data_for_7_7 = calculate_scores(
                     final_analysis_data.get('freq_omission', {}),
                     final_analysis_data.get('patterns', {}),
                     final_predicted_probabilities # Pass probabilities from the final run
                )

                red_scores_for_7_7 = final_scores_data_for_7_7.get('red_scores', {})
                blue_scores_for_7_7 = final_scores_data_for_7_7.get('blue_scores', {})

                if not red_scores_for_7_7 or len(red_scores_for_7_7) < 7 or not blue_scores_for_7_7 or len(blue_scores_for_7_7) < 7:
                     logger.error("不足够的评分号码来选择7红7蓝进行7+7复式投注。请检查数据和配置。")  # Log to console/default stderr
                     print("无法生成7+7复式选号。", file=sys.stdout)
                else:
                     # Sort scores and select the top 7 red and top 7 blue balls
                     sorted_red_scores = sorted(red_scores_for_7_7.items(), key=lambda item: item[1], reverse=True)
                     top_7_red_balls = [num for num, score in sorted_red_scores[:7]]

                     sorted_blue_scores = sorted(blue_scores_for_7_7.items(), key=lambda item: item[1], reverse=True)
                     top_7_blue_balls = [num for num, score in sorted_blue_scores[:7]]

                     # Print to file
                     print("基于总体分数，为7+7复式投注选择以下号码:", file=sys.stdout)
                     print(f"选择的7个红球: {sorted(top_7_red_balls)}", file=sys.stdout)
                     print(f"选择的7个蓝球: {sorted(top_7_blue_balls)}", file=sys.stdout)
                     print("\n此7+7选择覆盖C(7,6) * C(7,1) = 49个组合。", file=sys.stdout)
                     print("考虑这些号码如何符合历史模式和您的风险容忍度。", file=sys.stdout)

                     # Also print the 7+7 selection to the console for immediate feedback (using logger)
                     logger.info("\n--- 7+7复式选号 ---")
                     logger.info("基于总体分数，为7+7复式投注选择以下号码:")
                     logger.info(f"选择的7个红球: {sorted(top_7_red_balls)}")
                     logger.info(f"选择的7个蓝球: {sorted(top_7_blue_balls)}")
                     logger.info("此7+7选择覆盖49个组合。")


                print("="*50, file=sys.stdout)
                print(" 7+7选择完成 ", file=sys.stdout)
                print("="*50, file=sys.stdout)

                # 6. Plot results (if enabled) - requires matplotlib to run in an interactive environment or save figures
                if SHOW_PLOTS:
                    try:
                        plot_analysis_results(full_freq_omission_data, full_pattern_analysis_data)
                    except Exception as e:
                        logger.warning(f"绘图时出错: {e}")
                        print(f"\n绘图时出错: {e}", file=sys.stdout)

        else:
            # Errors during data loading or cleaning/engineering have been logged and printed to file.
            pass # Skip analysis due to data issues.

    except Exception as e:
        # Catch any unexpected errors in the main try block
        logger.error(f"执行过程中发生意外错误: {e}", exc_info=True) # Log with traceback to console
        # If the file is open (sys.stdout is redirected), print the error to the file
        print(f"\n执行过程中发生意外错误: {e}", file=sys.stdout)
        # Also print traceback to the file
        import traceback
        traceback.print_exc(file=sys.stdout)
        print("--- 错误跟踪结束 ---", file=sys.stdout)

    finally:
        # --- Close file and restore stdout ---
        if sys.stdout is not None and sys.stdout != sys.__stdout__:
             sys.stdout.close()
             sys.stdout = original_stdout # Restore original stdout

        # Output final message to console
        logger.info(f"\n分析完成。完整报告已保存到 {output_filename}")

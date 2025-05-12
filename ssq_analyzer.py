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
from sklearn.ensemble import RandomForestClassifier
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
TOP_N_RED_FOR_CANDIDATE = 25  # 用于生成组合的红球候选池大小（按分数从高到低选择）
TOP_N_BLUE_FOR_CANDIDATE = 10  # 用于生成组合的蓝球候选池大小（按分数从高到低选择）
ML_LAG_FEATURES = [1, 3, 5]  # ML模型使用的滞后特征期数，例如 [1, 3, 5] 表示使用前1期、前3期、前5期的数据作为特征
BACKTEST_PERIODS_COUNT = 200  # 回测使用的最近历史期数 (This will be maximum periods if data is sufficient)
SHOW_PLOTS = False  # 是否显示分析过程中生成的图表 (True 显示, False 屏蔽)

# 关联规则挖掘配置 (可按需调整)
ARM_MIN_SUPPORT = 0.005
ARM_MIN_CONFIDENCE = 0.3
ARM_MIN_LIFT = 1.0

# 评分权重 (启发式 - 可调整)
FREQ_SCORE_WEIGHT = 30
OMISSION_SCORE_WEIGHT = 20
ODD_EVEN_TENDENCY_BONUS = 10  # 红球匹配预测奇偶趋势的奖励
ZONE_TENDENCY_BONUS_MULTIPLIER = 2  # 最常见区域模式计数的乘数
BLUE_FREQ_SCORE_WEIGHT = 40
BLUE_OMISSION_SCORE_WEIGHT = 30
BLUE_ODD_TENDENCY_BONUS = 20  # 蓝球匹配预测奇偶趋势的奖励
BLUE_SIZE_TENDENCY_BONUS = 10  # 蓝球匹配最常见大小趋势的奖励
COMBINATION_ODD_COUNT_MATCH_BONUS = 20  # 组合匹配预测奇数数量的奖励
COMBINATION_BLUE_ODD_MATCH_BONUS = 15  # 组合匹配预测蓝球奇偶的奖励
COMBINATION_ZONE_MATCH_BONUS = 15  # 组合匹配最常见区域模式的奖励
COMBINATION_BLUE_SIZE_MATCH_BONUS = 10  # 组合匹配最常见蓝球大小的奖励

# ML模型参数 (RandomForest)
RF_ESTIMATORS = 50
RF_MAX_DEPTH = 10


# 配置日志系统
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
        # 假设CSV有如'期号', '开奖日期', '红球', '蓝球'等列
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

    red_balls_list = []
    blue_balls_list = []
    # 同时为排序位置的红球添加列
    red_pos_cols_list = [[] for _ in range(6)]

    rows_skipped_parsing = 0

    for index, row in df.iterrows():
        try:
            red_str = row['红球']
            blue_val = row['蓝球']

            if not isinstance(red_str, str):
                 rows_skipped_parsing += 1
                 continue  # 如果红球数据不是字符串则跳过

            # 提前处理潜在的非整数蓝球数据
            try:
                 blue_num = int(blue_val)
                 if not (1 <= blue_num <= 16):
                     rows_skipped_parsing += 1
                     continue
                 blue_balls_list.append(blue_num)
            except ValueError:
                 rows_skipped_parsing += 1
                 continue

            # 在蓝球验证后处理红球
            try:
                reds = sorted([int(x) for x in red_str.split(',')])  # 排序的红球
                if len(reds) != 6 or not all(1 <= r <= 33 for r in reds):
                    rows_skipped_parsing += 1
                    continue
                red_balls_list.append(reds)

                # 填充位置红球列表
                for i in range(6):
                    red_pos_cols_list[i].append(reds[i])

            except ValueError:
                rows_skipped_parsing += 1
                continue
            except Exception as e:
                rows_skipped_parsing += 1
                logger.warning(f"解析期号为 {row['期号']} 的红球时发生意外错误: {e}。跳过该行。")
                continue

        except Exception as e:
            rows_skipped_parsing += 1
            logger.warning(f"处理期号为 {row['期号']} 的行时发生意外错误: {e}。跳过该行。")
            continue

    if rows_skipped_parsing > 0:
         logger.warning(f"由于红球或蓝球数据解析错误，跳过了 {rows_skipped_parsing} 行。")

    # 创建仅包含成功解析行的新DataFrame
    if not red_balls_list or not blue_balls_list or len(red_balls_list) != len(blue_balls_list):
         logger.error("数据解析导致红球和蓝球列表不一致。")
         return None

    # 重新从解析的数据构建DataFrame以增强健壮性
    parsed_rows_data = []
    rows_skipped_parsing = 0  # 为清晰起见重置计数器

    for index, row in df.iterrows():
         try:
             red_str = row.get('红球')  # 使用.get以保安全
             blue_val = row.get('蓝球')
             period_val = row.get('期号')

             if not isinstance(red_str, str) or not blue_val or period_val is None:
                  rows_skipped_parsing += 1
                  continue  # 如果基本数据缺失或类型错误则跳过

             # 提前处理潜在的非整数蓝球数据
             try:
                  blue_num = int(blue_val)
                  if not (1 <= blue_num <= 16):
                      rows_skipped_parsing += 1
                      continue
             except ValueError:
                  rows_skipped_parsing += 1
                  continue

             # 在蓝球验证后处理红球
             try:
                 reds = sorted([int(x) for x in red_str.split(',')])  # 排序的红球
                 if len(reds) != 6 or not all(1 <= r <= 33 for r in reds):
                     rows_skipped_parsing += 1
                     continue

             except ValueError:
                 rows_skipped_parsing += 1
                 continue
             except Exception as e:
                 rows_skipped_parsing += 1
                 logger.warning(f"解析期号为 {period_val} 的红球时发生意外错误: {e}。跳过该行。")
                 continue

             # 如果到达这里，该行有效。添加到parsed_rows_data。
             row_data = {'期号': int(period_val)}  # 确保期号为int
             if '开奖日期' in row:  # 如果存在，包含'开奖日期'
                 row_data['开奖日期'] = row['开奖日期']
             # 添加排序后的红球和蓝球
             for i in range(6):
                 row_data[f'red{i+1}'] = reds[i]
                 row_data[f'red_pos{i+1}'] = reds[i]  # 同时添加位置信息（排序后与red相同）
             row_data['blue'] = blue_num

             parsed_rows_data.append(row_data)

         except Exception as e:
             rows_skipped_parsing += 1
             period_val = row.get('期号', 'N/A')
             logger.warning(f"处理期号为 {period_val} 的行时发生一般性意外错误: {e}。跳过该行。")
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
    red_pos_cols = [f'red_pos{i+1}' for i in range(6)]  # 排序后这些应该与red_cols相同
    essential_cols = red_cols + red_pos_cols + ['blue']
    if not all(col in df.columns for col in essential_cols):
        logger.error("清理后缺少特征工程所需的必要列。")
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
        if pd.isna(row[red_pos_cols]).any(): return 0  # 处理潜在的缺失位置数据（在健壮清理后不应发生）
        for i in range(5):
            if row[f'red_pos{i+1}'] + 1 == row[f'red_pos{i+2}']:
                count += 1
        return count

    # 仅当df_fe不为空时应用
    if not df_fe.empty:
        df_fe['red_consecutive_pairs'] = df_fe.apply(count_consecutive_pairs, axis=1)
    else:
         df_fe['red_consecutive_pairs'] = pd.Series(dtype=int)

    # 与上期重复（红球）
    # 需要处理第一行。先添加shift，然后计算。
    df_fe['prev_reds_str'] = df_fe['red1'].astype(str) + ',' + df_fe['red2'].astype(str) + ',' + df_fe['red3'].astype(str) + ',' + \
                             df_fe['red4'].astype(str) + ',' + df_fe['red5'].astype(str) + ',' + df_fe['red6'].astype(str)
    df_fe['prev_reds_shifted'] = df_fe['prev_reds_str'].shift(1)

    df_fe['red_repeat_count'] = 0  # 初始化
    if len(df_fe) > 1:
        for i in range(1, len(df_fe)):
            prev_reds_str = df_fe.loc[i, 'prev_reds_shifted']
            if pd.notna(prev_reds_str):
                 try:
                     prev_reds = set(int(x) for x in prev_reds_str.split(','))
                     current_reds = set(df_fe.loc[i, red_cols].tolist())
                     df_fe.loc[i, 'red_repeat_count'] = len(prev_reds.intersection(current_reds))
                 except ValueError:
                     # 在健壮清理后不应发生，但作为安全措施
                     logger.warning(f"解析索引 {i} 处的上期红球字符串时出错。")
                     df_fe.loc[i, 'red_repeat_count'] = 0  # 解析错误时默认为0重复

    df_fe.drop(columns=['prev_reds_str', 'prev_reds_shifted'], errors='ignore', inplace=True)

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
    red_pos_cols = [f'red_pos{i+1}' for i in range(6)]
    # 使用*当前*索引计算相对于该df切片最新期的遗漏
    most_recent_period_index = len(df) - 1

    if most_recent_period_index < 0:  # 在空检查后不应发生，但作为安全保障
        logger.warning("在analyze_frequency_omission中检查后DataFrame为空。")
        return {}

    all_reds = df[red_cols].values.flatten()
    all_blues = df['blue'].values

    # 频率（总体）
    red_freq = Counter(all_reds)
    blue_freq = Counter(all_blues)

    # 位置红球频率
    red_pos_freq = {}
    for col in red_pos_cols:
        red_pos_freq[col] = Counter(df[col])

    # 遗漏（当前）- 上次出现至今的期数
    current_omission = {}
    # 遗漏是相对于提供的df的*结束*计算的。
    # 索引`most_recent_period_index`是最新期。
    # 遗漏为0表示它出现在最新期（索引most_recent_period_index）。
    # 遗漏为1表示它上次出现在索引most_recent_period_index - 1。
    # 遗漏为k表示它上次出现在索引most_recent_period_index - k。
    # 如果从未见过，遗漏为most_recent_period_index + 1（或df长度）

    # 红球（任意位置）
    for number in RED_BALL_RANGE:
        # 找到最新出现的索引*在提供的df内*
        latest_appearance_index = df.index[
            (df[red_cols] == number).any(axis=1)
        ].max()  # 这给出了*在当前df内*的索引

        if pd.isna(latest_appearance_index):
             current_omission[number] = len(df)  # 在这些数据中从未见过
        else:
             current_omission[number] = most_recent_period_index - latest_appearance_index

    # 红球（按位置）
    red_pos_current_omission = {}
    for col in red_pos_cols:
        red_pos_current_omission[col] = {}
        for number in RED_BALL_RANGE:
            latest_appearance_index = df.index[
                 (df[col] == number)
            ].max()
            if pd.isna(latest_appearance_index):
                red_pos_current_omission[col][number] = len(df)
            else:
                red_pos_current_omission[col][number] = most_recent_period_index - latest_appearance_index

    # 蓝球
    for number in BLUE_BALL_RANGE:
         latest_appearance_index = df.index[
             (df['blue'] == number)
         ].max()
         if pd.isna(latest_appearance_index):
             current_omission[number] = len(df)
         else:
            current_omission[number] = most_recent_period_index - latest_appearance_index

    # 平均间隔（平均遗漏的代理）- 在提供的数据上计算
    average_interval = {}
    total_periods = len(df)
    for number in RED_BALL_RANGE:
        # 在频率为零时避免除以零，加1
        average_interval[number] = total_periods / (red_freq.get(number, 0) + 1)
    for number in BLUE_BALL_RANGE:
        average_interval[number] = total_periods / (blue_freq.get(number, 0) + 1)

    # 位置平均间隔
    red_pos_average_interval = {}
    for col in red_pos_cols:
        red_pos_average_interval[col] = {}
        col_freq = red_pos_freq.get(col, {})
        for number in RED_BALL_RANGE:
             red_pos_average_interval[col][number] = total_periods / (col_freq.get(number, 0) + 1)

    # 识别热/冷号（基于总体频率）
    # 优雅地处理空频率数据
    red_freq_items = sorted(red_freq.items(), key=lambda item: item[1], reverse=True) if red_freq else []
    blue_freq_items = sorted(blue_freq.items(), key=lambda item: item[1], reverse=True) if blue_freq else []

    # 基于顶部/底部百分比定义热/冷（确保阈值是有效索引）
    num_red_balls_possible = len(RED_BALL_RANGE)
    num_blue_balls_possible = len(BLUE_BALL_RANGE)

    red_hot_threshold = max(0, min(len(red_freq_items), int(num_red_balls_possible * 0.2)))
    red_cold_threshold = max(0, min(len(red_freq_items), int(num_red_balls_possible * 0.8)))
    blue_hot_threshold = max(0, min(len(blue_freq_items), int(num_blue_balls_possible * 0.3)))
    blue_cold_threshold = max(0, min(len(blue_freq_items), int(num_blue_balls_possible * 0.7)))

    hot_reds = [num for num, freq in red_freq_items[:red_hot_threshold]]
    cold_reds = [num for num, freq in red_freq_items[red_cold_threshold:]]
    hot_blues = [num for num, freq in blue_freq_items[:blue_hot_threshold]]
    cold_blues = [num for num, freq in blue_freq_items[blue_cold_threshold:]]

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

    # 模式分析结果
    pattern_results = {}

    # 红球和分布
    if 'red_sum' in df.columns and not df['red_sum'].empty:
        pattern_results['sum_stats'] = df['red_sum'].describe().to_dict()
        pattern_results['most_common_sum'] = df['red_sum'].mode()[0] if not df['red_sum'].mode().empty else None
    else:
         pattern_results['sum_stats'] = {}
         pattern_results['most_common_sum'] = None

    # 红球跨度分布
    if 'red_span' in df.columns and not df['red_span'].empty:
        pattern_results['span_stats'] = df['red_span'].describe().to_dict()
        pattern_results['most_common_span'] = df['red_span'].mode()[0] if not df['red_span'].mode().empty else None
    else:
        pattern_results['span_stats'] = {}
        pattern_results['most_common_span'] = None

    # 奇偶比分布
    if 'red_odd_count' in df.columns and not df['red_odd_count'].empty:
        odd_even_counts = df['red_odd_count'].value_counts().sort_index()
        pattern_results['odd_even_ratios'] = {f'{odd}:{6-odd}': int(count) for odd, count in odd_even_counts.items()}  # 将numpy int转换为python int
        pattern_results['most_common_odd_even_count'] = odd_even_counts.idxmax() if not odd_even_counts.empty else None
    else:
        pattern_results['odd_even_ratios'] = {}
        pattern_results['most_common_odd_even_count'] = None

    # 区域分布
    zone_cols = [f'red_{zone}_count' for zone in RED_ZONES.keys()]
    if all(col in df.columns for col in zone_cols) and not df.empty:
        zone_counts_df = df[zone_cols]
        if not zone_counts_df.empty:
            # 确保计数为整数，然后形成元组
            zone_counts_df = zone_counts_df.astype(int)
            zone_distribution_counts = zone_counts_df.apply(lambda row: tuple(row), axis=1).value_counts()
            pattern_results['zone_distribution_counts'] = {tuple(int(c) for c in dist): int(count) for dist, count in zone_distribution_counts.items()}  # 将键/值转换为python类型
            pattern_results['most_common_zone_distribution'] = zone_distribution_counts.index[0] if not zone_distribution_counts.empty else (0, 0, 0)
        else:
            pattern_results['zone_distribution_counts'] = {}
            pattern_results['most_common_zone_distribution'] = (0, 0, 0)  # 默认
    else:
         pattern_results['zone_distribution_counts'] = {}
         pattern_results['most_common_zone_distribution'] = (0, 0, 0)  # 默认

    # 连续对分布
    if 'red_consecutive_pairs' in df.columns and not df['red_consecutive_pairs'].empty:
        consecutive_counts = df['red_consecutive_pairs'].value_counts().sort_index()
        pattern_results['consecutive_counts'] = {int(count): int(freq) for count, freq in consecutive_counts.items()}
    else:
        pattern_results['consecutive_counts'] = {}

    # 与上期重复频率
    if 'red_repeat_count' in df.columns and not df['red_repeat_count'].empty:
        repeat_counts = df['red_repeat_count'].value_counts().sort_index()
        pattern_results['repeat_counts'] = {int(count): int(freq) for count, freq in repeat_counts.items()}
    else:
        pattern_results['repeat_counts'] = {}

    # 蓝球模式分析
    if 'blue_is_odd' in df.columns and not df['blue_is_odd'].empty:
        blue_odd_counts = df['blue_is_odd'].value_counts()
        pattern_results['blue_odd_counts'] = {bool(is_odd): int(count) for is_odd, count in blue_odd_counts.items()}  # 转换bool键、int值
    else:
        pattern_results['blue_odd_counts'] = {}

    if 'blue_is_large' in df.columns and not df['blue_is_large'].empty:
        blue_large_counts = df['blue_is_large'].value_counts()
        pattern_results['blue_large_counts'] = {bool(is_large): int(count) for is_large, count in blue_large_counts.items()}
    else:
        pattern_results['blue_large_counts'] = {}

    if 'blue_is_prime' in df.columns and not df['blue_is_prime'].empty:
        blue_prime_counts = df['blue_is_prime'].value_counts()
        pattern_results['blue_prime_counts'] = {bool(is_prime): int(count) for is_prime, count in blue_prime_counts.items()}
    else:
        pattern_results['blue_prime_counts'] = {}

    return pattern_results


def analyze_associations(df: pd.DataFrame, min_support: float = ARM_MIN_SUPPORT, min_confidence: float = ARM_MIN_CONFIDENCE, min_lift: float = ARM_MIN_LIFT) -> pd.DataFrame:
    """查找红球的频繁项集和关联规则。"""
    # 需要至少2期以查找关联
    if df is None or df.empty or len(df) < 2:
        return pd.DataFrame()  # 返回空DataFrame

    red_cols = [f'red{i+1}' for i in range(6)]
    # 检查红球列是否存在且在切片中不全为NaN/空
    if not all(col in df.columns for col in red_cols) or df[red_cols].isnull().all().all():
         return pd.DataFrame()

    # 将球号转换为字符串以用于TransactionEncoder（通常更安全）
    transactions = df[red_cols].astype(str).values.tolist()

    # 过滤掉空交易（如果有）（在清理后不应发生，但作为安全保障）
    transactions = [t for t in transactions if all(item and item != 'nan' for item in t)]
    if not transactions:
         return pd.DataFrame()

    te = TransactionEncoder()
    try:
        te_ary = te.fit(transactions).transform(transactions)
        df_onehot = pd.DataFrame(te_ary, columns=te.columns_)
    except Exception as e:
        logger.warning(f"关联规则转换过程中出错: {e}")
        return pd.DataFrame()

    if df_onehot.empty:
        return pd.DataFrame()

    try:
        # 基于数据大小调整min_support以要求最小绝对频率
        frequent_itemsets = apriori(df_onehot, min_support=min_support, use_colnames=True)

        if frequent_itemsets.empty:
             return pd.DataFrame()

        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    except Exception as e:
        logger.warning(f"Apriori算法执行过程中出错: {e}")
        return pd.DataFrame()

    try:
        # 生成具有最小置信度和提升度的规则
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
        if min_confidence is not None:  # 如果指定了置信度阈值则应用
             rules = rules[rules['confidence'] >= min_confidence]

        rules.sort_values(by='lift', ascending=False, inplace=True)
    except Exception as e:
        logger.warning(f"关联规则生成过程中出错: {e}")
        return pd.DataFrame()

    return rules

# --- 用于特征预测的ML ---

def create_lagged_features(df: pd.DataFrame, lags: List[int]) -> Optional[pd.DataFrame]:
    """为ML模型创建滞后特征。"""
    if df is None or df.empty or not lags:
         return None

    # 选择要滞后的特征 - 确保这些列存在
    lag_base_cols = ['red_sum', 'red_span', 'red_odd_count', 'blue_is_odd', 'red_consecutive_pairs', 'red_repeat_count']
    # 过滤掉可能在小df切片中不存在的列
    existing_lag_cols = [col for col in lag_base_cols if col in df.columns]

    if not existing_lag_cols:
         return None

    df_lagged = df[existing_lag_cols].copy()

    for lag in lags:
        if lag > 0:
            for col in existing_lag_cols:
                 # 使用.name获取原始列名
                 df_lagged[f'{col}_lag{lag}'] = df_lagged[col].shift(lag)

    # 删除由滞后引入的NaN行
    # 检查删除NaN后是否还有行
    initial_rows = len(df_lagged)
    df_lagged.dropna(inplace=True)
    if len(df_lagged) < initial_rows:
         pass  # 在回测过程中无需频繁记录此信息

    if df_lagged.empty:
        return None

    # 特征包括滞后值
    feature_cols = [col for col in df_lagged.columns if any(f'_lag{lag}' in col for lag in lags)]

    # 确保特征列实际存在
    feature_cols = [col for col in feature_cols if col in df_lagged.columns]

    if not feature_cols:
         logger.warning("滞后和删除NaN后没有创建特征列。")
         return None

    # 返回仅包含创建的特征和原始（非滞后）基础列的DataFrame
    # 生成的df_lagged中的非滞后基础列是对应于滞后特征的目标。
    # 因此返回的DF包含X（滞后特征）和y（当前期这些特征的值）。
    return df_lagged[feature_cols + existing_lag_cols]


def train_feature_prediction_models(df_train_raw: pd.DataFrame, lags: List[int]) -> Optional[dict]:
    """训练ML模型以预测下一期的特征。"""

    # 从训练数据创建滞后特征
    df_lagged_prepared = create_lagged_features(df_train_raw.copy(), lags)

    if df_lagged_prepared is None or df_lagged_prepared.empty:
        return None  # 如果无法训练则返回None

    # 从准备好的数据定义特征(X)和目标(y)
    # 特征是滞后列
    feature_cols = [col for col in df_lagged_prepared.columns if any(f'_lag{lag}' in col for lag in lags)]
    # 目标是非滞后基础列
    target_odd_col = 'red_odd_count'
    target_blue_odd_col = 'blue_is_odd'

    # 确保目标列在准备好的数据中存在
    if target_odd_col not in df_lagged_prepared.columns or target_blue_odd_col not in df_lagged_prepared.columns:
         logger.warning("滞后后找不到目标列('red_odd_count'或'blue_is_odd')。")
         return None

    X = df_lagged_prepared[feature_cols]
    y_odd = df_lagged_prepared[target_odd_col]
    # 将布尔目标转换为整数以用于RandomForestClassifier
    y_blue_odd = df_lagged_prepared[target_blue_odd_col].astype(int)

    if X.empty or y_odd.empty or y_blue_odd.empty:
         return None

    # --- 训练模型 ---
    # 使用RandomForestClassifiers进行分类任务(奇偶计数，蓝球奇偶)
    trained_models = {}  # 用于存储模型和特征列的字典

    try:
        # 训练红球奇数计数模型
        model_odd_count = RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=42, max_depth=RF_MAX_DEPTH)
        model_odd_count.fit(X, y_odd)
        trained_models['odd_count_model'] = model_odd_count

        # 如果目标中至少有两个类，则训练蓝球奇偶模型
        if len(y_blue_odd.unique()) > 1:
             model_blue_odd = RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=42, max_depth=RF_MAX_DEPTH)
             model_blue_odd.fit(X, y_blue_odd)
             trained_models['blue_odd_model'] = model_blue_odd
        else:
             pass  # blue_odd_model不会添加到trained_models

    except Exception as e:
        logger.warning(f"ML模型训练过程中出错: {e}")
        return None  # 如果训练失败则返回None

    trained_models['feature_cols'] = feature_cols  # 存储训练使用的特征列

    return trained_models


def predict_feature_tendency_ml(df_historical: pd.DataFrame, trained_models: Optional[dict], lags: List[int]) -> Dict:
    """使用训练好的ML模型和最新数据预测下一期的特征趋势。"""
    predicted_tendency = {}  # 默认为空字典

    if trained_models is None or df_historical is None or df_historical.empty:
        return predicted_tendency  # 返回空字典

    model_odd_count = trained_models.get('odd_count_model')
    model_blue_odd = trained_models.get('blue_odd_model')  # 如果未训练，这可能为None
    feature_cols = trained_models.get('feature_cols')

    if model_odd_count is None or feature_cols is None or not feature_cols:
         return predicted_tendency  # 返回空字典

    # 需要至少max(lags) + 1行来为最后一期创建特征
    max_lag = max(lags) if lags else 0
    if len(df_historical) < max_lag + 1:
         return predicted_tendency

    # 从最新相关历史创建预测特征
    # 需要最后的max(lags) + 1行来计算*最新*期的所有滞后（这成为单一预测行）。
    df_latest_history_for_lagging = df_historical.tail(max_lag + 1).copy()

    # 为这个小df创建滞后特征
    df_predict_prep = create_lagged_features(df_latest_history_for_lagging, lags)

    if df_predict_prep is None or df_predict_prep.empty:
        return predicted_tendency

    # 预测的特征是准备好的滞后特征的*最后*一行
    # 确保我们只选择训练中使用的特征列
    predict_df = df_predict_prep[feature_cols].tail(1).copy()

    if predict_df.empty:
        logger.warning("准备后预测特征DataFrame为空。")
        return predicted_tendency

    # 确保预测特征与训练特征具有相同的列，缺失填充为0
    try:
        predict_df = predict_df.reindex(columns=feature_cols, fill_value=0)
        # 检查重新索引或其他问题引入的NaN，并以0作为回退填充
        if predict_df.isnull().values.any():
             logger.warning("重新索引后预测特征中发现NaN值。用0填充。")
             predict_df.fillna(0, inplace=True)  # 回退fillna

    except Exception as e:
        logger.warning(f"预测特征准备过程中出错(reindex/fillna): {e}")
        return predicted_tendency

    # 进行预测
    try:
        # 使用.tolist()[0]获取单行的预测
        predicted_tendency['predicted_odd_count'] = model_odd_count.predict(predict_df).tolist()[0]
    except Exception as e:
        logger.warning(f"警告: 预测红球奇数计数时出错: {e}")

    try:
        # 仅在蓝球奇偶模型成功训练时预测
        if model_blue_odd is not None:
             # 使用.tolist()[0]并将整数预测转换回布尔值
             predicted_blue_is_odd_int = model_blue_odd.predict(predict_df).tolist()[0]
             predicted_tendency['predicted_blue_is_odd'] = bool(predicted_blue_is_odd_int)
    except Exception as e:
        logger.warning(f"警告: 预测蓝球奇偶时出错: {e}")

    return predicted_tendency


# --- 预测趋势与号码评分 ---

def calculate_scores(freq_omission_data: dict, pattern_analysis_data: dict, predicted_tendency: dict) -> dict:
    """计算每个号码的综合得分。"""
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

    # --- 评分公式（更多因素）---
    max_red_rank = len(RED_BALL_RANGE)
    max_blue_rank = len(BLUE_BALL_RANGE)

    # 趋势预测结果
    predicted_odd_count = predicted_tendency.get('predicted_odd_count', None)
    predicted_blue_is_odd = predicted_tendency.get('predicted_blue_is_odd', None)

    # 如果ML没有预测它们，获取历史模式趋势用于评分
    hist_most_common_odd_count = pattern_analysis_data.get('most_common_odd_even_count')
    hist_most_common_zone_dist = pattern_analysis_data.get('most_common_zone_distribution')
    hist_most_common_blue_large = pattern_analysis_data.get('blue_large_counts', {}).get(True, None) is not None and (
        pattern_analysis_data['blue_large_counts'].get(True, 0) > pattern_analysis_data['blue_large_counts'].get(False, 0))
    # 注意: hist_most_common_blue_large现在是一个布尔值，指示large是否更常见

    # 如果可用则使用ML预测，否则回退到历史模式
    actual_odd_count_tendency = predicted_odd_count if predicted_odd_count is not None else hist_most_common_odd_count
    hist_most_common_blue_odd_val = pattern_analysis_data.get('blue_odd_counts', {}).get(True, None) is not None and (
        pattern_analysis_data['blue_odd_counts'].get(True, 0) > pattern_analysis_data['blue_odd_counts'].get(False, 0))
    actual_blue_odd_tendency = predicted_blue_is_odd if predicted_blue_is_odd is not None else hist_most_common_blue_odd_val  # 修正的蓝球奇偶趋势回退

    actual_zone_dist_tendency = hist_most_common_zone_dist  # 假设ML不预测区域

    # 确保趋势值在评分中使用前不是None
    use_odd_count_tendency = actual_odd_count_tendency is not None
    use_blue_odd_tendency = actual_blue_odd_tendency is not None
    use_zone_dist_tendency = actual_zone_dist_tendency is not None
    use_blue_size_tendency = hist_most_common_blue_large is not None

    for num in RED_BALL_RANGE:
        # 因素1: 频率排名（倒数）- 更高频率获得更高基础分
        # 在调用.get之前确保num在red_freq_rank索引中
        freq_score = (max_red_rank - red_freq_rank.get(num, max_red_rank)) / max_red_rank * FREQ_SCORE_WEIGHT

        # 因素2: 遗漏偏差（接近平均值奖励）
        # 使用.get和默认值（例如，0）如果num不在current_omission或average_interval中
        dev = current_omission.get(num, len(RED_BALL_RANGE) * 2) - average_interval.get(num, len(RED_BALL_RANGE) * 2)  # 如果未见过，使用较大默认偏差
        omission_score = OMISSION_SCORE_WEIGHT * np.exp(-0.005 * dev**2)  # 如果需要，调整衰减率

        # 因素3: 趋势拟合（奇偶）
        tendency_score = 0
        if use_odd_count_tendency:
             # 简单奖励: 如果趋势建议更多奇数，奇数获得奖励；如果更少奇数，偶数获得奖励。
             # 让我们使用奇数计数阈值（例如，>= 3是更多奇数，< 3是更少奇数）
             is_odd_num = num % 2 != 0
             if (is_odd_num and actual_odd_count_tendency >= 3) or (not is_odd_num and actual_odd_count_tendency < 3):
                 tendency_score += ODD_EVEN_TENDENCY_BONUS

        # 因素4: 趋势拟合（区域）- 使用历史最常见区域
        if use_zone_dist_tendency:
             num_zone_idx = None
             if RED_ZONES['Zone1'][0] <= num <= RED_ZONES['Zone1'][1]: num_zone_idx = 0
             elif RED_ZONES['Zone2'][0] <= num <= RED_ZONES['Zone2'][1]: num_zone_idx = 1
             elif RED_ZONES['Zone3'][0] <= num <= RED_ZONES['Zone3'][1]: num_zone_idx = 2

             # 检查数字的区域计数在最常见历史模式中是否为正
             if num_zone_idx is not None and num_zone_idx < len(actual_zone_dist_tendency) and actual_zone_dist_tendency[num_zone_idx] > 0:
                  tendency_score += actual_zone_dist_tendency[num_zone_idx] * ZONE_TENDENCY_BONUS_MULTIPLIER

        # 组合因素（权重可调整）
        red_scores[num] = freq_score + omission_score + tendency_score

    for num in BLUE_BALL_RANGE:
        # 因素1: 频率排名（倒数）
        # 在调用.get之前确保num在blue_freq_rank索引中
        freq_score = (max_blue_rank - blue_freq_rank.get(num, max_blue_rank)) / max_blue_rank * BLUE_FREQ_SCORE_WEIGHT

        # 因素2: 遗漏偏差
        # 使用.get和默认值（例如，0）如果num不在current_omission或average_interval中
        dev = current_omission.get(num, len(BLUE_BALL_RANGE) * 2) - average_interval.get(num, len(BLUE_BALL_RANGE) * 2)  # 如果未见过，使用较大默认偏差
        omission_score = BLUE_OMISSION_SCORE_WEIGHT * np.exp(-0.01 * dev**2)  # 如果需要，调整衰减率

        # 因素3: 趋势拟合（奇偶）
        tendency_score = 0
        if use_blue_odd_tendency:
            actual_blue_is_odd = num % 2 != 0
            if actual_blue_is_odd == actual_blue_odd_tendency:
                tendency_score += BLUE_ODD_TENDENCY_BONUS

        # 因素4: 趋势拟合（大小）- 使用历史模式
        if use_blue_size_tendency:
             is_large = num > 8
             # hist_most_common_blue_large如果large更常见则为True，否则为False
             if is_large == hist_most_common_blue_large:
                  tendency_score += BLUE_SIZE_TENDENCY_BONUS

        # 组合因素
        blue_scores[num] = freq_score + omission_score + tendency_score

    # 将分数归一化到固定范围（例如，0-100）
    all_scores = list(red_scores.values()) + list(blue_scores.values())
    if all_scores:  # 避免对空列表进行min/max操作
        min_score, max_score = min(all_scores), max(all_scores)
        # 添加浮点比较容差以处理所有分数相同的情况
        if (max_score - min_score) > 1e-9:
            red_scores = {num: (score - min_score) / (max_score - min_score) * 100 for num, score in red_scores.items()}
            blue_scores = {num: (score - min_score) / (max_score - min_score) * 100 for num, score in blue_scores.items()}
        else:  # 处理所有分数非常接近或相同的情况
             red_scores = {num: 50.0 for num in RED_BALL_RANGE}
             blue_scores = {num: 50.0 for num in BLUE_BALL_RANGE}
    else:  # 如果all_scores为空，返回默认分数
        red_scores = {num: 50.0 for num in RED_BALL_RANGE}
        blue_scores = {num: 50.0 for num in BLUE_BALL_RANGE}

    return {'red_scores': red_scores, 'blue_scores': blue_scores}


# --- 号码组合生成与过滤 ---

def generate_combinations(scores_data: dict, pattern_analysis_data: dict, predicted_tendency: dict, num_combinations: int = NUM_COMBINATIONS_TO_GENERATE) -> tuple[List[Dict], list[str]]:
    """基于分数和趋势生成潜在组合。
       返回组合字典列表和用于输出的格式化字符串列表。
    """
    red_scores = scores_data.get('red_scores', {})
    blue_scores = scores_data.get('blue_scores', {})
    tendency = predicted_tendency  # ML预测趋势

    # 基于分数选择候选池
    # 处理分数为空或数量不足的情况
    sorted_red_scores = sorted(red_scores.items(), key=lambda item: item[1], reverse=True) if red_scores else []
    red_candidate_pool = [num for num, score in sorted_red_scores[:TOP_N_RED_FOR_CANDIDATE]]

    sorted_blue_scores = sorted(blue_scores.items(), key=lambda item: item[1], reverse=True) if blue_scores else []
    blue_candidate_pool = [num for num, score in sorted_blue_scores[:TOP_N_BLUE_FOR_CANDIDATE]]

    # 确保采样的最小池大小
    if len(red_candidate_pool) < 6:
         logger.warning(f"红球候选池大小({len(red_candidate_pool)})小于6。对红球使用全范围。")
         red_candidate_pool = list(RED_BALL_RANGE)
    if len(blue_candidate_pool) < 1:
         logger.warning(f"蓝球候选池大小({len(blue_candidate_pool)})小于1。对蓝球使用全范围。")
         blue_candidate_pool = list(BLUE_BALL_RANGE)

    # 修订的生成策略: 从池中生成许多，评分/排名，选择前N个
    # 生成一个远大于最终所需组合数量的池
    large_pool_size = num_combinations * 500  # 减少的池大小乘数以提高可能的生成速度
    if large_pool_size < 100: large_pool_size = 100  # 确保最小池大小

    generated_pool = []
    attempts = 0
    max_attempts_pool = large_pool_size * 10  # 安全限制

    # 计算基于分数的概率，处理可能的除以零情况
    red_weights = np.array([red_scores.get(num, 0) for num in red_candidate_pool])
    red_weights[red_weights < 0] = 0  # 确保权重非负
    total_red_weight = np.sum(red_weights)
    red_probabilities = red_weights / total_red_weight if total_red_weight > 1e-9 else np.ones(len(red_candidate_pool)) / len(red_candidate_pool)  # 如果总权重接近零则使用均匀概率

    blue_weights = np.array([blue_scores.get(num, 0) for num in blue_candidate_pool])
    blue_weights[blue_weights < 0] = 0
    total_blue_weight = np.sum(blue_weights)
    blue_probabilities = blue_weights / total_blue_weight if total_blue_weight > 1e-9 else np.ones(len(blue_candidate_pool)) / len(blue_candidate_pool)

    while len(generated_pool) < large_pool_size and attempts < max_attempts_pool:
         attempts += 1
         try:
              # 使用计算的概率从候选池中采样6个不同的红球
              sampled_red_balls = sorted(np.random.choice(
                  red_candidate_pool, size=6, replace=False, p=red_probabilities
              ).tolist())

              # 采样1个蓝球
              sampled_blue_ball = np.random.choice(
                   blue_candidate_pool, size=1, replace=False, p=blue_probabilities
              ).tolist()[0]

              generated_pool.append({'red': sampled_red_balls, 'blue': sampled_blue_ball})

         except ValueError as e:
              # 如果概率没有正确归一化或和接近零，可能会发生这种情况
             try:
                  # 回退到从候选池中的简单随机采样
                  if len(red_candidate_pool) >= 6:
                     sampled_red_balls = sorted(random.sample(red_candidate_pool, 6))
                  else:
                      # 理想情况下，如果初始池检查有效，不应达到此情况，但作为安全保障
                      sampled_red_balls = sorted(random.sample(list(RED_BALL_RANGE), 6))  # 回退到全范围

                  if blue_candidate_pool:
                       sampled_blue_ball = random.choice(blue_candidate_pool)
                  else:
                       sampled_blue_ball = random.choice(list(BLUE_BALL_RANGE))  # 回退到全范围

                  generated_pool.append({'red': sampled_red_balls, 'blue': sampled_blue_ball})
             except ValueError as e_fallback:
                 logger.error(f"回退采样失败: {e_fallback}。停止组合生成尝试。")
                 break  # 如果即使回退也失败，放弃采样
         except Exception as e:
             logger.warning(f"在第{attempts}次组合采样尝试中发生意外错误: {e}。跳过此尝试。")
             continue  # 继续下一次尝试

    if not generated_pool:
         logger.warning("没有生成组合。")
         return [], []  # 返回空列表

    # 现在，基于它们如何符合号码分数AND趋势对生成的组合进行评分/排名
    scored_combinations = []

    # 获取用于组合评分的历史模式趋势（如果ML没有预测它们）
    hist_most_common_odd_count = pattern_analysis_data.get('most_common_odd_even_count')
    hist_most_common_zone_dist = pattern_analysis_data.get('most_common_zone_distribution')
    hist_most_common_blue_large = pattern_analysis_data.get('blue_large_counts', {}).get(True, None) is not None and (
        pattern_analysis_data['blue_large_counts'].get(True, 0) > pattern_analysis_data['blue_large_counts'].get(False, 0))
    hist_most_common_blue_odd_val = pattern_analysis_data.get('blue_odd_counts', {}).get(True, None) is not None and (
        pattern_analysis_data['blue_odd_counts'].get(True, 0) > pattern_analysis_data['blue_odd_counts'].get(False, 0))

    # 如果可用则使用ML预测，否则回退到历史模式
    actual_odd_count_tendency = tendency.get('predicted_odd_count', hist_most_common_odd_count)
    actual_blue_odd_tendency = tendency.get('predicted_blue_is_odd', hist_most_common_blue_odd_val)
    actual_zone_dist_tendency = hist_most_common_zone_dist  # 假设ML不预测区域
    actual_blue_size_tendency = hist_most_common_blue_large  # 使用历史模式

    # 确保趋势值在评分中使用前不是None
    use_odd_count_tendency = actual_odd_count_tendency is not None
    use_blue_odd_tendency = actual_blue_odd_tendency is not None
    use_zone_dist_tendency = actual_zone_dist_tendency is not None
    use_blue_size_tendency = actual_blue_size_tendency is not None

    for combo in generated_pool:
        red_balls = combo['red']
        blue_ball = combo['blue']

        # 计算组合的得分
        # 各球分数之和（使用.get与默认值0，如果号码不在分数中）
        combo_score = sum(scores_data.get('red_scores', {}).get(r, 0) for r in red_balls) + scores_data.get('blue_scores', {}).get(blue_ball, 0)

        # 基于拟合预测或历史特征添加奖励
        feature_match_score = 0

        # 红球奇数计数匹配
        if use_odd_count_tendency:
             actual_odd_count = sum(x % 2 != 0 for x in red_balls)
             if actual_odd_count == actual_odd_count_tendency:
                  feature_match_score += COMBINATION_ODD_COUNT_MATCH_BONUS

        # 蓝球奇偶匹配
        if use_blue_odd_tendency:
            actual_blue_is_odd = blue_ball % 2 != 0
            if actual_blue_is_odd == actual_blue_odd_tendency:
                feature_match_score += COMBINATION_BLUE_ODD_MATCH_BONUS

        # 区域分布匹配（使用历史模式作为预测）
        if use_zone_dist_tendency:
             actual_zone_counts = [0, 0, 0]
             for ball in red_balls:
                 if RED_ZONES['Zone1'][0] <= ball <= RED_ZONES['Zone1'][1]: actual_zone_counts[0] += 1
                 elif RED_ZONES['Zone2'][0] <= ball <= RED_ZONES['Zone2'][1]: actual_zone_counts[1] += 1
                 elif RED_ZONES['Zone3'][0] <= ball <= RED_ZONES['Zone3'][1]: actual_zone_counts[2] += 1
             if tuple(actual_zone_counts) == actual_zone_dist_tendency:
                 feature_match_score += COMBINATION_ZONE_MATCH_BONUS

        # 蓝球大小匹配（使用历史模式作为预测）
        if use_blue_size_tendency:
             is_large = blue_ball > 8
             # actual_blue_size_tendency如果large更常见则为True，否则为False
             if is_large == actual_blue_size_tendency:
                  feature_match_score += COMBINATION_BLUE_SIZE_MATCH_BONUS

        # 组合个体分数和特征匹配分数
        total_combo_score = combo_score + feature_match_score

        scored_combinations.append({'combination': combo, 'score': total_combo_score})

    # 按分数排序组合并选择前N个
    scored_combinations.sort(key=lambda x: x['score'], reverse=True)
    final_recommendations_data = scored_combinations[:num_combinations]

    # --- 格式化输出字符串 ---
    output_strings = []
    output_strings.append("推荐组合:")
    if final_recommendations_data:
         for i, rec in enumerate(final_recommendations_data):
             output_strings.append(f"组合 {i+1}: 红球 {sorted(rec['combination']['red'])} 蓝球 {rec['combination']['blue']} (分数: {rec['score']:.2f})")
    else:
         output_strings.append("无法生成推荐组合。")

    # 返回组合字典列表AND格式化的输出字符串
    return final_recommendations_data, output_strings

# --- 核心分析和推荐函数 (新) ---
# 此函数封装了给定数据集切片的主要逻辑流
def analyze_and_recommend(
    df_historical: pd.DataFrame,
    lags: List[int],
    num_combinations: int,
    train_ml: bool = True,  # 是否在此运行中训练ML模型
    existing_models: Optional[Dict] = None  # 如果train_ml为False，传递现有模型
) -> tuple[List[Dict], list[str], dict, Optional[Dict]]:
    """
    基于提供的历史数据执行分析，预测趋势，计算分数，并为下一期生成组合。
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

    # 2. 预测特征趋势（如果已训练/提供则使用ML，否则依赖历史模式）
    current_trained_models = None
    predicted_tendency = {}

    if train_ml:
        # 使用提供的历史数据训练ML模型
        current_trained_models = train_feature_prediction_models(df_historical, lags)
        if current_trained_models:
            # 使用新训练的模型和最新历史预测趋势
            predicted_tendency = predict_feature_tendency_ml(df_historical, current_trained_models, lags)
        else:
             logger.warning("ML模型训练失败。仅依靠历史模式进行评分趋势。")
             # predicted_tendency仍为{}

    elif existing_models:
        # 使用现有训练好的模型在最新数据上预测趋势
        current_trained_models = existing_models  # 使用提供的模型
        predicted_tendency = predict_feature_tendency_ml(df_historical, current_trained_models, lags)
        if not predicted_tendency:
             logger.warning("使用现有模型的ML预测失败。仅依靠历史模式进行评分趋势。")
             # predicted_tendency仍为{}

    else:
         logger.info("跳过ML训练且未提供现有模型。仅依靠历史模式进行评分趋势。")
         # predicted_tendency仍为{}

    # 3. 计算号码分数（结合历史分析和预测趋势）
    scores_data = calculate_scores(
        freq_omission_data,
        pattern_analysis_data,
        predicted_tendency  # 如果ML失败，此字典可能为空
    )

    # 4. 基于分数和趋势生成组合
    recommendations_data, recommendations_strings = generate_combinations(
        scores_data,
        pattern_analysis_data,  # 为组合评分回退传递模式数据
        predicted_tendency,  # 为组合评分传递预测趋势
        num_combinations=num_combinations
    )

    return recommendations_data, recommendations_strings, analysis_data, current_trained_models


# --- 验证、回测与持续优化 ---

def backtest(df: pd.DataFrame, lags: List[int], num_combinations_per_period: int, backtest_periods_count: int) -> pd.DataFrame:
    """
    在历史数据上执行回测，包括重新训练ML模型。
    """
    logger.info("\n" + "="*50)
    logger.info(" 开始回测 ")
    logger.info("="*50)

    # 在第一个回测期之前需要足够的历史用于初始滞后和分析
    min_periods_for_initial_analysis = max(max(lags) if lags else 0, 1) + 10
    if len(df) < min_periods_for_initial_analysis + 1:
         logger.warning(f"数据不足({len(df)})，无法使用初始分析缓冲区进行回测(至少需要{min_periods_for_initial_analysis + 1}期)。跳过回测。")
         logger.info("="*50)
         logger.info(" 回测已跳过 ")
         logger.info("="*50)
         return pd.DataFrame()

    # 确定要回测的期的范围
    start_prediction_index = min_periods_for_initial_analysis
    end_prediction_index = len(df) - 1

    if start_prediction_index >= end_prediction_index + 1:
         logger.warning(f"初始分析历史({min_periods_for_initial_analysis}期)后没有足够的剩余数据进行回测。开始索引: {start_prediction_index}, 结束索引: {end_prediction_index}。跳过回测。")
         logger.info("="*50)
         logger.info(" 回测已跳过 ")
         logger.info("="*50)
         return pd.DataFrame()

    # 根据可用数据和请求的计数调整实际回测期数
    available_backtest_periods = end_prediction_index - start_prediction_index + 1
    actual_backtest_periods_count = min(backtest_periods_count, available_backtest_periods)

    # 计算我们评估结果的*第一个*期的索引
    backtest_start_period_index = end_prediction_index - actual_backtest_periods_count + 1
    if backtest_start_period_index < start_prediction_index:
        backtest_start_period_index = start_prediction_index

    # 获取实际回测的期号范围
    start_period_number = df.loc[backtest_start_period_index, '期号']
    end_period_number = df.loc[end_prediction_index, '期号']
    
    logger.info(f"回测 {actual_backtest_periods_count} 期的预测，期号范围: {start_period_number} 至 {end_period_number}")
    logger.info(f"预测从索引 {backtest_start_period_index} 到 {end_prediction_index} 的期。")
    logger.info(f"使用直到索引 {backtest_start_period_index - 1} 的数据进行第一次预测的分析/训练。")

    results = []
    red_cols = [f'red{i+1}' for i in range(6)]

    # 显示初始进度条 - 仅在控制台显示，不在报告中显示
    total_steps = end_prediction_index - backtest_start_period_index + 1
    
    # 保存当前的stdout
    original_stdout = sys.stdout
    
    # 迭代期
    for i in range(backtest_start_period_index, end_prediction_index + 1):
        # 更新进度条 - 恢复原始stdout显示进度，然后再切回文件
        current_progress = i - backtest_start_period_index + 1
        
        # 将stdout临时还原为原始控制台stdout以显示进度条
        sys.stdout = sys.__stdout__
        show_progress(current_progress, total_steps, prefix='回测进度:', suffix='完成', length=50)
        # 恢复stdout到之前的状态（如果在主程序中已重定向到文件）
        sys.stdout = original_stdout
        
        # 数据准备和预测逻辑保持不变
        train_data = df.iloc[:i].copy()

        if train_data.empty or len(train_data) < (max(lags) if lags else 0) + 1:
             logger.warning(f"训练数据不足({len(train_data)}行)用于预测期索引{i}。跳过。")
             continue

        actual_row_index = i
        actual_period = df.loc[actual_row_index, '期号']

        if actual_row_index not in df.index:
             logger.error(f"DataFrame中找不到期索引{actual_row_index}(期号: {actual_period})的实际结果。跳过此期的预测。")
             continue

        try:
            actual_red = set(df.loc[actual_row_index, red_cols].tolist())
            actual_blue = df.loc[actual_row_index, 'blue']
        except KeyError as e:
             logger.error(f"期索引{actual_row_index}(期号: {actual_period})的实际结果中缺少红球或蓝球数据: {e}。跳过预测。")
             continue

        with SuppressOutput(suppress_stdout=True, capture_stderr=True):
             predicted_combinations_data, predicted_combinations_strings, analysis_data, trained_models = analyze_and_recommend(
                 train_data,
                 lags,
                 num_combinations=num_combinations_per_period,
                 train_ml=True
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
        else:
            logger.warning(f"未为期 {actual_period} 生成组合。")

    # 在控制台显示最终完成的进度条
    sys.stdout = sys.__stdout__  # 确保最终进度显示到控制台
    show_progress(total_steps, total_steps, prefix='回测进度:', suffix='完成', length=50)
    # 恢复重定向
    sys.stdout = original_stdout

    logger.info("="*50)
    logger.info(" 回测完成 ")
    logger.info("="*50)

    if not results:
        logger.warning("未记录回测结果。")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    
    # 添加回测期号范围到结果DataFrame中，以便在报告中使用
    results_df.attrs['start_period'] = start_period_number
    results_df.attrs['end_period'] = end_period_number

    return results_df


# --- 绘图函数（移到分析函数之外）---
def plot_analysis_results(freq_omission_data: dict, pattern_analysis_data: dict):
     """从分析结果生成图表。"""
     if not SHOW_PLOTS:
          plt.close('all')  # 关闭任何遗留的图形
          return

     logger.info("生成图表...")

     # 在绘图前检查数据是否可用
     if not freq_omission_data or not pattern_analysis_data:
          logger.warning("分析数据不可用于绘图。")
          return

     # 频率图
     red_freq = freq_omission_data.get('red_freq', {})
     blue_freq = freq_omission_data.get('blue_freq', {})
     red_pos_freq = freq_omission_data.get('red_pos_freq', {})
     red_pos_cols = [f'red_pos{i+1}' for i in range(6)]  # 假设这些键存在于red_pos_freq结构中

     if red_freq or blue_freq:
          plt.figure(figsize=(14, 6))
          if red_freq:
              plt.subplot(1, 2, 1)
              sns.barplot(x=list(red_freq.keys()), y=list(red_freq.values()))
              plt.title('红球总体频率')
              plt.xlabel('数字'); plt.ylabel('频率')
          if blue_freq:
              plt.subplot(1, 2, 2)
              sns.barplot(x=list(blue_freq.keys()), y=list(blue_freq.values()))
              plt.title('蓝球频率')
              plt.xlabel('数字'); plt.ylabel('频率')
          plt.tight_layout()
          plt.show()

     # 位置红球频率图
     if red_pos_freq and any(red_pos_freq.values()):  # 检查red_pos_freq是否非空且有数据
          fig, axes = plt.subplots(2, 3, figsize=(15, 10))
          axes = axes.flatten()
          for i, col in enumerate(red_pos_cols):
               if col in red_pos_freq and red_pos_freq[col]:
                  # 排序键以保持一致的绘图顺序
                  sorted_freq_items = sorted(red_pos_freq[col].items())
                  sns.barplot(x=[item[0] for item in sorted_freq_items], y=[item[1] for item in sorted_freq_items], ax=axes[i])
                  axes[i].set_title(f'红球位置 {i+1} 频率')
                  axes[i].set_xlabel('数字')
                  axes[i].set_ylabel('频率')
               else:
                  axes[i].set_title(f'红球位置 {i+1} 频率(无数据)')
                  axes[i].set_xlabel('数字')
                  axes[i].set_ylabel('频率')
          plt.tight_layout()
          plt.show()

     # 模式分布图（和、跨度、奇偶、连续、重复）
     # 例如：如果数据可用，绘制奇偶比分布
     odd_even_ratios = pattern_analysis_data.get('odd_even_ratios', {})
     if odd_even_ratios:
          plt.figure(figsize=(6, 4))
          sns.barplot(x=list(odd_even_ratios.keys()), y=list(odd_even_ratios.values()))
          plt.title('红球奇:偶比分布')
          plt.xlabel('奇:偶比'); plt.ylabel('频率'); plt.show()

     # 例如：如果数据可用，绘制连续对分布
     consecutive_counts = pattern_analysis_data.get('consecutive_counts', {})
     if consecutive_counts:
          plt.figure(figsize=(6, 4))
          sns.barplot(x=list(consecutive_counts.keys()), y=list(consecutive_counts.values()))
          plt.title('红球连续对分布')
          plt.xlabel('连续对数量'); plt.ylabel('频率'); plt.show()

     # 例如：如果数据可用，绘制重复次数分布
     repeat_counts = pattern_analysis_data.get('repeat_counts', {})
     if repeat_counts:
          plt.figure(figsize=(6, 4))
          sns.barplot(x=list(repeat_counts.keys()), y=list(repeat_counts.values()))
          plt.title('红球从上期重复频率')
          plt.xlabel('重复球数量'); plt.ylabel('频率'); plt.show()

     logger.info("图表生成完成。")


# --- 主执行流程 ---

if __name__ == "__main__":
    # --- 配置输出文件 ---
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(SCRIPT_DIR, f"ssq_analysis_output_{timestamp}.txt")

    output_file = None  # 初始化文件句柄

    try:
        # 打开输出文件
        output_file = open(output_filename, 'w', encoding='utf-8')
        # 将stdout重定向到输出文件以获取主报告内容
        sys.stdout = output_file

        print(f"--- 双色球分析报告 ---", file=sys.stdout)
        print(f"运行日期: {now.strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stdout)
        print(f"输出文件: {output_filename}", file=sys.stdout)
        print("-" * 30, file=sys.stdout)
        print("\n", file=sys.stdout)

        # --- 开始分析和预测 ---

        # 1. 加载和准备数据
        # 尝试首先加载处理好的数据文件
        processed_data_exists = os.path.exists(PROCESSED_CSV_PATH)
        if processed_data_exists:
            logger.info(f"找到处理好的数据文件: {PROCESSED_CSV_PATH}")
            df = load_data(PROCESSED_CSV_PATH)
            if df is not None and not df.empty:
                logger.info("成功加载处理好的数据文件。")
            else:
                logger.warning("处理好的数据文件加载失败或为空。尝试加载原始数据文件。")
                processed_data_exists = False

        # 如果没有处理好的数据或加载失败，使用原始数据
        if not processed_data_exists:
            with SuppressOutput(suppress_stdout=False, capture_stderr=True):
                df = load_data(CSV_FILE_PATH)
                if df is not None and not df.empty:
                    df = clean_and_structure(df)
                    if df is not None and not df.empty:
                         df = feature_engineer(df)
                         if df is None or df.empty:
                              logger.error("特征工程失败或导致空数据。")
                              print("\n错误: 特征工程失败。无法继续分析。", file=sys.stdout)
                    else:
                        logger.error("数据清理和结构化失败或导致空数据。")
                        print("\n错误: 数据清理和结构化失败。无法继续分析。", file=sys.stdout)
                else:
                    logger.error("数据加载失败或导致空数据。")
                    print("\n错误: 数据加载失败。无法继续分析。", file=sys.stdout)

        if df is not None and not df.empty:
            # 提取数据范围信息
            min_period = df['期号'].min()
            max_period = df['期号'].max()
            total_periods = len(df)
            # 获取最后开奖日期
            last_drawing_date = df['开奖日期'].iloc[-1] if '开奖日期' in df.columns and not df.empty else "未知"

            # 在报告开头添加数据范围信息和最后开奖日期
            print(f"\n数据概况:", file=sys.stdout)
            print(f"  数据期数范围: 第 {min_period} 期 至 第 {max_period} 期", file=sys.stdout)
            print(f"  总数据条数: {total_periods} 期", file=sys.stdout)
            print(f"  最后开奖日期: {last_drawing_date}", file=sys.stdout) # Added this line
            print("\n", file=sys.stdout)

            # 检查是否有足够的数据用于分析
            max_lag = max(ML_LAG_FEATURES) if ML_LAG_FEATURES else 0
            min_periods_needed = max(max_lag + 1, 10)
            if len(df) < min_periods_needed:
                 logger.error(f"清理/特征工程后有效期数不足({len(df)})，无法使用当前滞后设置和分析缓冲区进行分析(至少需要{min_periods_needed})。")
                 print(f"\n错误: 清理/特征工程后有效期数不足({len(df)})，无法进行分析(至少需要{min_periods_needed})。无法继续分析。", file=sys.stdout)
            else:
                # 2. 执行完整历史分析
                print("\n" + "="*50, file=sys.stdout)
                print(" 完整历史分析 ", file=sys.stdout)
                print(f" (基于第 {min_period} 期至第 {max_period} 期数据) ", file=sys.stdout)
                print("="*50, file=sys.stdout)
                # 使用SuppressOutput隐藏内部分析函数打印从文件，但仍记录stderr
                with SuppressOutput(suppress_stdout=True, capture_stderr=True):
                    full_freq_omission_data = analyze_frequency_omission(df)
                    full_pattern_analysis_data = analyze_patterns(df)
                    full_association_rules = analyze_associations(df, ARM_MIN_SUPPORT, ARM_MIN_CONFIDENCE, ARM_MIN_LIFT)  # 在完整数据上分析关联

                print("\n历史分析摘要（基于完整数据）:", file=sys.stdout)
                print("\n频率和遗漏亮点:", file=sys.stdout)
                # 打印选定的频率/遗漏数据到文件
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
                    # 格式化规则以输出到文件
                    for _, rule in full_association_rules.head(10).iterrows():
                        print(f"  {set(rule['antecedents'])} -> {set(rule['consequents'])} (支持度: {rule['support']:.4f}, 置信度: {rule['confidence']:.2f}, 提升度: {rule['lift']:.2f})", file=sys.stdout)
                else:
                    print("\n以当前阈值未找到显著关联规则。", file=sys.stdout)

                print("="*50, file=sys.stdout)
                print(" 历史分析完成 ", file=sys.stdout)
                print("="*50, file=sys.stdout)

                 # 3. 执行回测
                # 回测输出记录到控制台并在文件中总结
                # 回测函数本身处理其控制台输出（进度条）和内部日志记录
                backtest_results = backtest(df, ML_LAG_FEATURES, NUM_COMBINATIONS_TO_GENERATE, BACKTEST_PERIODS_COUNT)

                print("\n" + "="*50, file=sys.stdout)
                print(" 回测摘要 ", file=sys.stdout)

                # 如果有回测结果，显示回测的期号范围
                if not backtest_results.empty:
                    start_period = backtest_results.attrs.get('start_period', '未知')
                    end_period = backtest_results.attrs.get('end_period', '未知')
                    print(f" (基于第 {start_period} 期至第 {end_period} 期数据) ", file=sys.stdout)

                print("="*50, file=sys.stdout)
                if not backtest_results.empty:
                     # 打印回测摘要到文件
                     periods_with_results = backtest_results['period'].nunique()
                     print(f"测试的总期数(已生成组合): {periods_with_results}", file=sys.stdout)
                     print(f"生成的总组合数: {len(backtest_results)}", file=sys.stdout)
                     if periods_with_results > 0:
                          print(f"每期生成的组合数(平均): {len(backtest_results) / periods_with_results:.2f}", file=sys.stdout)

                     avg_red_hits = backtest_results['red_hits'].mean()
                     print(f"每个组合的平均红球命中数: {avg_red_hits:.2f}", file=sys.stdout)

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

                # 4. 为下一期生成最终推荐组合
                print("\n" + "="*50, file=sys.stdout)
                print(" 生成最终推荐 ", file=sys.stdout)
                print(f" (基于第 {min_period} 期至第 {max_period} 期全部数据) ", file=sys.stdout)
                print("="*50, file=sys.stdout)

                # 使用SuppressOutput隐藏analyze_and_recommend的内部打印/日志输出
                # 最终推荐字符串将在下面明确打印
                with SuppressOutput(suppress_stdout=True, capture_stderr=True):
                     final_recommendations_data, final_recommendations_strings, final_analysis_data, final_trained_models = analyze_and_recommend(
                         df,  # 使用全部可用数据进行最终预测
                         ML_LAG_FEATURES,
                         NUM_COMBINATIONS_TO_GENERATE,
                         train_ml=True  # 对完整数据最后一次训练模型以进行最终预测
                     )

                # 将最终推荐组合打印到输出文件
                if final_recommendations_strings:
                    for line in final_recommendations_strings:
                        print(line, file=sys.stdout)
                else:
                    print("无法生成最终推荐组合。", file=sys.stdout)

                print("="*50, file=sys.stdout)
                print(" 最终推荐完成 ", file=sys.stdout)
                print("="*50, file=sys.stdout)

                # 5. 生成7+7复式选号
                print("\n" + "="*50, file=sys.stdout)
                print(" 7+7复式选号 ", file=sys.stdout)
                print("="*50, file=sys.stdout)

                final_scores_data = calculate_scores(
                     final_analysis_data.get('freq_omission', {}),
                     final_analysis_data.get('patterns', {}),
                     {}  # 使用空字典
                )

                red_scores_for_7_7 = final_scores_data.get('red_scores', {})
                blue_scores_for_7_7 = final_scores_data.get('blue_scores', {})

                if not red_scores_for_7_7 or len(red_scores_for_7_7) < 7 or not blue_scores_for_7_7 or len(blue_scores_for_7_7) < 7:
                     logger.error("不足够的评分号码来选择7红7蓝进行7+7复式投注。")  # 记录到控制台/默认stderr
                     print("无法生成7+7复式选号。", file=sys.stdout)
                else:
                     # 排序分数并选择前7个红球和前7个蓝球
                     sorted_red_scores = sorted(red_scores_for_7_7.items(), key=lambda item: item[1], reverse=True)
                     top_7_red_balls = [num for num, score in sorted_red_scores[:7]]

                     sorted_blue_scores = sorted(blue_scores_for_7_7.items(), key=lambda item: item[1], reverse=True)
                     top_7_blue_balls = [num for num, score in sorted_blue_scores[:7]]

                     # 打印到文件
                     print("基于总体分数，为7+7复式投注选择以下号码:", file=sys.stdout)
                     print(f"选择的7个红球: {sorted(top_7_red_balls)}", file=sys.stdout)
                     print(f"选择的7个蓝球: {sorted(top_7_blue_balls)}", file=sys.stdout)
                     print("\n此7+7选择覆盖C(7,6) * C(7,1) = 49个组合。", file=sys.stdout)
                     print("考虑这些号码如何符合历史模式和您的风险容忍度。", file=sys.stdout)

                     # 也将选择的7+7打印到控制台以获得即时反馈(使用logger)
                     logger.info("\n--- 7+7复式选号 ---")
                     logger.info("基于总体分数，为7+7复式投注选择以下号码:")
                     logger.info(f"选择的7个红球: {sorted(top_7_red_balls)}")
                     logger.info(f"选择的7个蓝球: {sorted(top_7_blue_balls)}")
                     logger.info("此7+7选择覆盖49个组合。")


                print("="*50, file=sys.stdout)
                print(" 7+7选择完成 ", file=sys.stdout)
                print("="*50, file=sys.stdout)

                # 6. 绘制结果（如果启用）- 要求matplotlib在交互式环境中运行或保存图表
                # 这里调用绘图，但plt.show()会阻塞或图表不会出现，除非环境支持。
                # 如果非交互式运行，考虑将图表保存到文件。
                # 现在，如果SHOW_PLOTS为True，则调用该函数。
                if SHOW_PLOTS:
                    try:
                        plot_analysis_results(full_freq_omission_data, full_pattern_analysis_data)
                    except Exception as e:
                        logger.warning(f"绘图时出错: {e}")
                        print(f"\n绘图时出错: {e}", file=sys.stdout)

        else:
            # 数据加载或清理/工程期间的错误已记录并打印到文件。
            pass  # 由于数据问题跳过分析。

    except Exception as e:
        # 捕获主try块中的任何意外错误
        logger.error(f"执行过程中发生意外错误: {e}", exc_info=True)  # 使用traceback记录到控制台
        # 如果文件已打开（sys.stdout已重定向），将错误打印到文件
        print(f"\n执行过程中发生意外错误: {e}", file=sys.stdout)
        # 同时将traceback打印到文件
        import traceback
        traceback.print_exc(file=sys.stdout)
        print("--- 错误跟踪结束 ---", file=sys.stdout)

    finally:
        # --- 关闭文件并恢复stdout ---
        if sys.stdout is not None and sys.stdout != sys.__stdout__:
             sys.stdout.close()
             sys.stdout = sys.__stdout__  # 恢复原始stdout

        # 向控制台输出最终消息
        logger.info(f"\n分析完成。完整报告已保存到 {output_filename}")

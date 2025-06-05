import pandas as pd
import numpy as np
from collections import Counter
import random
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

# --- 全局配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # 脚本文件所在目录
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'shuangseqiu.csv') # 原始双色球数据CSV文件路径
PROCESSED_CSV_PATH = os.path.join(SCRIPT_DIR, 'shuangseqiu_processed.csv') # 预处理后的数据CSV文件路径
WEIGHTS_CONFIG_FILE = os.path.join(SCRIPT_DIR, 'weights_config.json') # 存储算法权重的配置文件路径
ENABLE_OPTUNA_OPTIMIZATION = False  # 设置为 True 以在本次运行中启用Optuna优化；False 则尝试加载或使用默认权重

RED_BALL_RANGE = range(1, 34) # 红球号码的有效范围 (1到33，共33个)
BLUE_BALL_RANGE = range(1, 17) # 蓝球号码的有效范围 (1到16，共16个)
RED_ZONES = {'Zone1': (1, 11), 'Zone2': (12, 22), 'Zone3': (23, 33)} # 红球分区定义，用于统计和特征工程

ML_LAG_FEATURES = [1, 3, 5, 10] # 机器学习模型使用的滞后特征阶数（例如，参考前1期、3期等数据）
ML_INTERACTION_PAIRS = [('red_sum', 'red_odd_count')] # 用于生成乘积交互特征的特征对 (例如，和值 * 奇数个数)
ML_INTERACTION_SELF = ['red_span'] # 用于生成自身平方交互特征的特征 (例如，跨度^2)

BACKTEST_PERIODS_COUNT = 100 # 完整回测阶段评估的期数
OPTIMIZATION_BACKTEST_PERIODS = 10 # Optuna优化时用于快速回测的期数（通常少于完整回测期数以加速优化）
OPTIMIZATION_TRIALS = 100 # Optuna优化试验的迭代次数
RECENT_FREQ_WINDOW = 20 # 计算号码近期出现频率的窗口大小（最近多少期）

CANDIDATE_POOL_SCORE_THRESHOLDS = {'High': 70, 'Medium': 25} # *红球候选池分数段的划分阈值（大于70为高分，大于40小于等于70为中分，小于等于40为低分）
CANDIDATE_POOL_SEGMENT_NAMES = ['High', 'Medium', 'Low'] # 候选池分段的名称，用于内部逻辑

# 默认算法权重，如果在配置文件中找不到或加载失败，将使用这些值
DEFAULT_WEIGHTS = {
    # --- 最终推荐组合层面的反向思维配置 ---
    'FINAL_COMBO_REVERSE_ENABLED': True,       # 是否启用最终推荐组合层面的反向思维 (True: 启用, False: 禁用)
    'FINAL_COMBO_REVERSE_REMOVE_TOP_PERCENT': 0.5,     # 如果启用最终反向思维，从最终推荐中移除得分最高的百分比数量的组合 (例如0.1表示移除前10%)
    'FINAL_COMBO_REVERSE_REFILL': False,        # 如果在最终反向思维中移除了组合，是否尝试从备选池补充到原数量 (True: 补充, False: 不补充)

    # --- 组合生成与候选池大小控制 ---
    'NUM_COMBINATIONS_TO_GENERATE': 10,        # 最终推荐生成的组合（注数）数量
    'TOP_N_RED_FOR_CANDIDATE': 25,             # 用于构建红球候选池的顶尖红球数量（目标大小）
    'TOP_N_BLUE_FOR_CANDIDATE': 6,             # 用于构建蓝球候选池的顶尖蓝球数量

    # --- 单个球号评分权重 (红球) ---
    'FREQ_SCORE_WEIGHT': 28.19,                # 红球历史总频率得分的权重
    'OMISSION_SCORE_WEIGHT': 19.92,            # 红球当前遗漏（与平均遗漏的偏差）得分的权重
    'MAX_OMISSION_RATIO_SCORE_WEIGHT_RED': 16.12, # 红球当前遗漏与其历史最大遗漏比率得分的权重
    'RECENT_FREQ_SCORE_WEIGHT_RED': 15.71,      # 红球近期（如20期）出现频率得分的权重
    'ML_PROB_SCORE_WEIGHT_RED': 22.43,         # 红球机器学习模型预测出现概率得分的权重

    # --- 单个球号评分权重 (蓝球) ---
    'BLUE_FREQ_SCORE_WEIGHT': 27.11,           # 蓝球历史总频率得分的权重
    'BLUE_OMISSION_SCORE_WEIGHT': 23.26,        # 蓝球当前遗漏（与平均遗漏的偏差）得分的权重
    'ML_PROB_SCORE_WEIGHT_BLUE': 43.48,        # 蓝球机器学习模型预测出现概率得分的权重

    # --- 组合属性匹配奖励权重 ---
    'COMBINATION_ODD_COUNT_MATCH_BONUS': 13.10,  # 推荐组合的红球奇数个数与历史最常见模式匹配时的奖励权重
    'COMBINATION_BLUE_ODD_MATCH_BONUS': 0.40,    # 推荐组合的蓝球奇偶性与历史最常见模式匹配时的奖励权重
    'COMBINATION_ZONE_MATCH_BONUS': 13.12,   # 推荐组合的红球区间分布与历史最常见模式匹配时的奖励权重
    'COMBINATION_BLUE_SIZE_MATCH_BONUS': 0.84,   # 推荐组合的蓝球大小与历史最常见模式匹配时的奖励权重

    # --- 关联规则挖掘 (ARM) 参数与奖励 ---
    'ARM_MIN_SUPPORT': 0.01,                 # ARM算法的最小支持度阈值
    'ARM_MIN_CONFIDENCE': 0.53,               # ARM算法的最小置信度阈值
    'ARM_MIN_LIFT': 1.53,                      # ARM算法的最小提升度阈值
    'ARM_COMBINATION_BONUS_WEIGHT': 18.86,     # 推荐组合如果命中了挖掘出的关联规则，其基础奖励权重
    'ARM_BONUS_LIFT_FACTOR': 0.48,             # ARM奖励中，规则提升度(lift)的贡献因子
    'ARM_BONUS_CONF_FACTOR': 0.25,             # ARM奖励中，规则置信度(confidence)的贡献因子

    # --- 红球候选池分段抽选比例与约束 ---
    'CANDIDATE_POOL_PROPORTIONS_HIGH': 0.28,   # 红球候选池中，期望从'高分段'选取的球所占的比例
    'CANDIDATE_POOL_PROPORTIONS_MEDIUM': 0.34, # 红球候选池中，期望从'中分段'选取的球所占的比例(低分段比例将通过 1 - High - Medium 计算)
    'CANDIDATE_POOL_MIN_PER_SEGMENT': 5,      # 红球候选池构建时，每个分数段（高/中/低）至少要选出的球数

    # --- 组合多样性控制参数 ---
    'DIVERSITY_MIN_DIFFERENT_REDS': 3,        # 最终推荐的多个组合之间，任意两个组合的红球至少要有几个不同
    'DIVERSITY_SELECTION_MAX_ATTEMPTS': 41,   # 在多样性选择阶段，为找到一个符合要求的组合所进行的最大尝试次数
    'DIVERSITY_SUM_DIFF_THRESHOLD': 15,       # 不同组合间红球和值的最小差异阈值，用于增强多样性
    'DIVERSITY_ODDEVEN_DIFF_MIN_COUNT': 1,    # 不同组合间红球奇数个数的最小差异数量，用于增强多样性
    'DIVERSITY_ZONE_DIST_MIN_DIFF_ZONES': 3,  # 不同组合间红球区间分布（如[2,2,2]）必须有个数差异的最小区间数量

    # --- 反向思维策略参数 (用于红球候选池构建) ---
    'REVERSE_THINKING_ITERATIONS': 1,         # 执行反向筛选（从候选池移除最高分红球）的迭代次数 (0表示禁用)
    'REVERSE_THINKING_RED_BALLS_TO_REMOVE_PER_ITER': 7, # 每次反向筛选迭代中，要从候选池移除的最高分红球的数量 (0表示禁用)

    # --- Optuna 优化时目标函数的权重 (用于评估回测性能) ---
    'OPTUNA_PRIZE_6_WEIGHT': 0.25,             # Optuna评估时，六等奖的价值权重
    'OPTUNA_PRIZE_5_WEIGHT': 0.20,             # Optuna评估时，五等奖的价值权重
    'OPTUNA_PRIZE_4_WEIGHT': 1.69,             # Optuna评估时，四等奖的价值权重
    'OPTUNA_PRIZE_3_WEIGHT': 4.02,             # Optuna评估时，三等奖的价值权重
    'OPTUNA_PRIZE_2_WEIGHT': 9.94,             # Optuna评估时，二等奖的价值权重
    'OPTUNA_PRIZE_1_WEIGHT': 16.48,            # Optuna评估时，一等奖的价值权重
    'OPTUNA_BLUE_HIT_RATE_WEIGHT': 19.67,      # Optuna评估时，蓝球在各期命中率的价值权重 (越高越好)
    'OPTUNA_RED_HITS_WEIGHT': 4.84            # Optuna评估时，红球平均命中数的价值权重 (越高越好，通常会做加权处理如 hit^1.5)
}
CURRENT_WEIGHTS = DEFAULT_WEIGHTS.copy() # 当前运行使用的权重，可能被Optuna优化结果或文件加载结果覆盖

# 红球分数段边界，用于历史中奖红球的分数段分布分析和报告
SCORE_SEGMENT_BOUNDARIES = [0, 25, 50, 75, 100]
# 根据分数段边界生成标签，例如 '0-25', '26-50'。这里假设标签为 [start, end) 形式，最后一个为 [start, end]
SCORE_SEGMENT_LABELS = [f'{SCORE_SEGMENT_BOUNDARIES[i]}-{SCORE_SEGMENT_BOUNDARIES[i+1]-1 if i < len(SCORE_SEGMENT_BOUNDARIES)-2 else SCORE_SEGMENT_BOUNDARIES[i+1]}' for i in range(len(SCORE_SEGMENT_BOUNDARIES)-1)]

# 确保标签数量与边界匹配，并在必要时调整最后一个标签以包含上限
if len(SCORE_SEGMENT_LABELS) != len(SCORE_SEGMENT_BOUNDARIES) - 1:
     raise ValueError("分数段标签数量与边界数量不匹配，请检查配置。")

# 机器学习模型参数（这些参数不通过Optuna主研究优化，但可在单独研究中调整）
LGBM_PARAMS = {'objective': 'binary', 'metric': 'binary_logloss', 'n_estimators': 100, 'learning_rate': 0.04, 'feature_fraction': 0.7, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'lambda_l1': 0.15, 'lambda_l2': 0.15, 'num_leaves': 15, 'min_child_samples': 15, 'verbose': -1, 'n_jobs': 1, 'seed': 42, 'boosting_type': 'gbdt'}
LOGISTIC_REG_PARAMS = {'penalty': 'l2', 'C': 0.1, 'solver': 'lbfgs', 'random_state': 42, 'max_iter': 5000, 'tol': 1e-3}
SVC_PARAMS = {'C': 0.1, 'kernel': 'rbf', 'gamma': 'scale', 'probability': True, 'random_state': 42, 'cache_size': 200, 'max_iter': 25000, 'tol': 1e-3}
XGB_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'n_estimators': 100, 'learning_rate': 0.04, 'max_depth': 3, 'subsample': 0.7, 'colsample_bytree': 0.7, 'gamma': 0.1, 'lambda': 0.15, 'alpha': 0.15, 'seed': 42, 'n_jobs': 1}
MIN_POSITIVE_SAMPLES_FOR_ML = 25 # 训练ML模型所需的最小正样本数（至少出现这么多次才能训练）

# 日志记录器配置
console_formatter = logging.Formatter('%(message)s') # 简洁格式，只显示消息
detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') # 详细格式，包含时间、级别
logger = logging.getLogger('ssq_analyzer')
logger.setLevel(logging.DEBUG) # Logger本身的级别设为DEBUG，以便能处理所有消息类型
logger.propagate = False # 防止日志消息传递到根记录器（避免重复输出）

global_console_handler = logging.StreamHandler(sys.stdout) # 控制台输出处理器
global_console_handler.setFormatter(console_formatter)
logger.addHandler(global_console_handler)

def set_console_verbosity(level=logging.INFO, use_simple_formatter=False):
    """
    设置控制台日志的级别和格式。
    Args:
        level (int): 日志级别，如 logging.INFO, logging.DEBUG。
        use_simple_formatter (bool): 如果为 True，使用简洁格式；否则使用详细格式。
    """
    global_console_handler.setLevel(level)
    if use_simple_formatter:
        global_console_handler.setFormatter(console_formatter)
    else:
        global_console_handler.setFormatter(detailed_formatter)

class SuppressOutput:
    """
    一个用于临时抑制标准输出和捕获标准错误的上下文管理器。
    主要用于在Optuna等库内部运行，避免其过多日志污染控制台。
    """
    def __init__(self, suppress_stdout=True, capture_stderr=True):
        self.suppress_stdout = suppress_stdout
        self.capture_stderr = capture_stderr
        self.old_stdout = None
        self.old_stderr = None
        self.stdout_io = None # 新增，用于存储重定向的stdout流
        self.stderr_io = None

    def __enter__(self):
        # 重定向标准输出和标准错误
        if self.suppress_stdout:
            self.old_stdout = sys.stdout
            self.stdout_io = io.StringIO() # 将重定向的流存储在实例变量中
            sys.stdout = self.stdout_io
        if self.capture_stderr:
            self.old_stderr = sys.stderr
            self.stderr_io = io.StringIO()
            sys.stderr = self.stderr_io
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复标准输出和标准错误，并记录捕获到的错误信息

        # 恢复 stderr 并获取捕获内容
        if self.capture_stderr and self.old_stderr is not None:
            sys.stderr = self.old_stderr
            captured_stderr_content = ""
            if self.stderr_io: # 确保 stderr_io 存在
                try:
                    captured_stderr_content = self.stderr_io.getvalue()
                    self.stderr_io.close() # 关闭流
                except Exception as e:
                    logger.warning(f"关闭或获取 stderr_io 内容时出错: {e}")
            
            if captured_stderr_content.strip():
                logger.warning(f"捕获到的标准错误输出:\n{captured_stderr_content.strip()}")
            self.stderr_io = None # 清理引用

        # 恢复 stdout
        if self.suppress_stdout and self.old_stdout is not None:
            if self.stdout_io: # 确保 stdout_io 存在
                try:
                    # 只有当我们自己创建并重定向了 sys.stdout 时才尝试关闭
                    if sys.stdout is self.stdout_io: # 确认当前 sys.stdout 确实是我们的 StringIO
                        sys.stdout.close()
                except Exception as e:
                    logger.warning(f"关闭 stdout_io 时出错: {e}")
            sys.stdout = self.old_stdout
            self.stdout_io = None # 清理引用
            
        return False # 不抑制异常，让异常继续传播

def load_weights_from_file(filepath: str, defaults: Dict) -> Tuple[Dict, str]:
    """
    从JSON文件加载权重配置。
    如果文件不存在或格式错误，则使用默认权重并尝试保存默认配置。
    Args:
        filepath (str): 权重配置文件的路径。
        defaults (Dict): 默认权重字典。
    Returns:
        Tuple[Dict, str]: (加载/合并后的权重字典, 状态字符串)。
    状态字符串：
    - 'loaded_active_config': 文件存在，有效，并已加载。
    - 'defaults_used_new_config_saved': 文件不存在，使用默认值，并保存了新文件。
    - 'defaults_used_config_error': 文件存在但无效，使用默认值。原始文件未修改。
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_weights = json.load(f)
        
        # 合并加载的权重和默认权重，优先使用加载的权重，但确保类型与默认值一致
        merged_weights = defaults.copy()
        for key in defaults:
            if key in loaded_weights:
                default_val_type = type(defaults[key])
                loaded_val = loaded_weights[key]

                # 尝试根据默认值类型进行转换，如果类型不一致则跳过或警告
                if isinstance(defaults[key], (int, float)):
                    if isinstance(loaded_val, (int, float)):
                        merged_weights[key] = default_val_type(loaded_val)
                    else:
                        logger.warning(f"权重文件 '{filepath}' 中键 '{key}' 的值 '{loaded_val}' 类型与默认值 ({default_val_type.__name__}) 不匹配。将使用默认值。")
                        merged_weights[key] = defaults[key]
                elif isinstance(defaults[key], str):
                    if isinstance(loaded_val, str):
                        merged_weights[key] = loaded_val
                    else:
                        logger.warning(f"权重文件 '{filepath}' 中键 '{key}' 的值 '{loaded_val}' 类型与默认值 ({default_val_type.__name__}) 不匹配。将使用默认值。")
                        merged_weights[key] = defaults[key]
                elif isinstance(defaults[key], bool):
                    if isinstance(loaded_val, bool):
                        merged_weights[key] = loaded_val
                    else:
                        # 尝试将非布尔值转换为布尔值，例如 "True" -> True
                        if isinstance(loaded_val, str):
                            if loaded_val.lower() == 'true': merged_weights[key] = True
                            elif loaded_val.lower() == 'false': merged_weights[key] = False
                            else:
                                logger.warning(f"权重文件 '{filepath}' 中键 '{key}' 的值 '{loaded_val}' 无法转换为布尔值。将使用默认值。")
                                merged_weights[key] = defaults[key]
                        else:
                            logger.warning(f"权重文件 '{filepath}' 中键 '{key}' 的值 '{loaded_val}' 类型与默认值 ({default_val_type.__name__}) 不匹配。将使用默认值。")
                            merged_weights[key] = defaults[key]
                elif type(defaults[key]) == type(loaded_val): # 对于列表、字典等复杂类型，直接赋值
                    merged_weights[key] = loaded_val
                else:
                    logger.warning(f"权重文件 '{filepath}' 中键 '{key}' 的类型 '{type(loaded_val).__name__}' 与默认值类型 '{default_val_type.__name__}' 不匹配。将使用默认值。")
                    merged_weights[key] = defaults[key]

        # 检查是否有默认权重中的键未在加载的权重中（例如，新添加的默认权重）
        for key_default in defaults:
            if key_default not in merged_weights:
                logger.info(f"权重文件 {filepath} 缺少键 '{key_default}'。将使用该键的默认值。")
                merged_weights[key_default] = defaults[key_default]

        # 验证候选池比例的有效性
        prop_h = merged_weights.get('CANDIDATE_POOL_PROPORTIONS_HIGH', 0.5)
        prop_m = merged_weights.get('CANDIDATE_POOL_PROPORTIONS_MEDIUM', 0.3)
        if not (0 <= prop_h <= 1 and 0 <= prop_m <= 1 and (prop_h + prop_m) <= 1 + 1e-9): # 允许一点浮点误差
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
    """
    将权重字典保存到JSON文件。
    Args:
        filepath (str): 目标文件路径。
        weights_to_save (Dict): 要保存的权重字典。
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(weights_to_save, f, indent=4)
        logger.info(f"权重已成功保存到 {filepath}")
    except Exception as e:
        logger.error(f"保存权重时出错: {e}")

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    从CSV文件加载数据，尝试多种常用编码。
    Args:
        file_path (str): CSV文件路径。
    Returns:
        Optional[pd.DataFrame]: 加载成功的DataFrame，或None。
    """
    try:
        encodings = ['utf-8', 'gbk', 'latin-1'] # 常用编码列表
        for enc in encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                logger.debug(f"成功使用编码 '{enc}' 加载文件: {file_path}")
                return df
            except UnicodeDecodeError:
                logger.debug(f"尝试编码 '{enc}' 失败，继续尝试其他编码。")
                continue # 尝试下一种编码
        logger.error(f"无法使用任何尝试的编码打开文件 {file_path}。")
        return None
    except FileNotFoundError:
        logger.error(f"错误: 文件 {file_path} 未找到。")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"错误: 文件 {file_path} 为空。")
        return None
    except Exception as e:
        logger.error(f"加载 {file_path} 时出错: {e}")
        return None

def clean_and_structure(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    清洗和结构化原始DataFrame，确保数据格式正确并符合逻辑。
    Args:
        df (pd.DataFrame): 原始加载的DataFrame。
    Returns:
        Optional[pd.DataFrame]: 清洗和结构化后的DataFrame，或None。
    """
    if df is None or df.empty:
        logger.warning("清洗和结构化：输入DataFrame为空。")
        return None

    # 删除关键列（期号、红球、蓝球）缺失的行
    df.dropna(subset=['期号', '红球', '蓝球'], inplace=True)
    if df.empty:
        logger.warning("清洗和结构化：删除缺失关键列的行后DataFrame为空。")
        return None

    try:
        # 清洗期号，确保为整数并排序
        df['期号'] = pd.to_numeric(df['期号'], errors='coerce').astype('Int64') # 转换为可空整数
        df.dropna(subset=['期号'], inplace=True) # 删除期号转换失败的行
        df['期号'] = df['期号'].astype(int)
        df.sort_values(by='期号', ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
    except Exception as e:
        logger.error(f"清洗和结构化：期号处理失败: {e}")
        return None
    if df.empty:
        logger.warning("清洗和结构化：期号处理后DataFrame为空。")
        return None

    parsed_rows = []
    for _, row in df.iterrows():
        try:
            period_val = row.get('期号')
            red_str = str(row.get('红球', '')).strip()
            blue_val = row.get('蓝球')
            date_val = str(row.get('日期', '')).strip()

            if pd.isna(period_val) or not red_str or pd.isna(blue_val):
                continue # 跳过无效行

            # 验证蓝球号码
            blue_num = int(float(blue_val)) # 先转浮点再转整数，处理 "1.0" 这样的情况
            if not (min(BLUE_BALL_RANGE) <= blue_num <= max(BLUE_BALL_RANGE)):
                logger.debug(f"清洗和结构化：期号 {period_val} 蓝球 '{blue_val}' 超出范围。跳过。")
                continue

            # 验证红球号码
            reds_str_list = red_str.split(',')
            if len(reds_str_list) != 6:
                logger.debug(f"清洗和结构化：期号 {period_val} 红球数量不为6。跳过。红球: '{red_str}'")
                continue
            
            reds_int_list = []
            for r_s in reds_str_list:
                try:
                    r_n = int(float(r_s)) # 处理 "1.0" 这样的红球
                    if not (min(RED_BALL_RANGE) <= r_n <= max(RED_BALL_RANGE)):
                        logger.debug(f"清洗和结构化：期号 {period_val} 红球 '{r_s}' 超出范围。跳过。")
                        reds_int_list = [] # 清空列表，标记为无效
                        break
                    reds_int_list.append(r_n)
                except ValueError:
                    logger.debug(f"清洗和结构化：期号 {period_val} 红球 '{r_s}' 无法解析为数字。跳过。")
                    reds_int_list = [] # 清空列表，标记为无效
                    break
            
            if len(reds_int_list) != 6:
                continue # 如果红球解析失败或数量不符，跳过

            reds_int_list.sort() # 确保红球按顺序存储

            record = {'期号': int(period_val)}
            if date_val: record['日期'] = date_val
            for i in range(6):
                record[f'red{i+1}'] = reds_int_list[i]
                record[f'red_pos{i+1}'] = reds_int_list[i] # 暂时用排序后的红球作为位置特征
            record['blue'] = blue_num
            parsed_rows.append(record)
        except Exception as e:
            logger.debug(f"清洗和结构化：处理行时发生未知错误: {e}。原始行: {row.to_dict()}")
            continue # 跳过解析错误的行

    if not parsed_rows:
        logger.warning("清洗和结构化：所有行都被过滤掉，没有有效数据。")
        return None
    
    # 再次排序并重置索引，确保最终DataFrame是按期号升序的
    final_df = pd.DataFrame(parsed_rows).sort_values(by='期号').reset_index(drop=True)
    logger.info(f"清洗和结构化完成。处理后数据包含 {len(final_df)} 期。")
    return final_df

def feature_engineer(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    对DataFrame进行特征工程，计算各种衍生特征。
    Args:
        df (pd.DataFrame): 经过清洗和结构化后的DataFrame。
    Returns:
        Optional[pd.DataFrame]: 包含新特征的DataFrame，或None。
    """
    if df is None or df.empty:
        logger.warning("特征工程：输入DataFrame为空。")
        return None
    
    df_fe = df.copy()
    red_cols = [f'red{i+1}' for i in range(6)]

    # 确保红球和蓝球列存在且为数值类型
    if not all(c in df_fe.columns for c in red_cols + ['blue', '期号']):
        logger.error("特征工程：缺少必要的红球、蓝球或期号列。")
        return None
    
    # 将红球列转换为数值类型，并删除转换失败的行
    for r_col in red_cols:
        df_fe[r_col] = pd.to_numeric(df_fe[r_col], errors='coerce')
    df_fe.dropna(subset=red_cols, inplace=True)
    
    if df_fe.empty:
        logger.warning("特征工程：红球列转换后DataFrame为空。")
        return None

    # 基本红球特征
    df_fe['red_sum'] = df_fe[red_cols].sum(axis=1) # 和值
    df_fe['red_span'] = df_fe[red_cols].max(axis=1) - df_fe[red_cols].min(axis=1) # 跨度

    # 奇数个数
    df_fe['red_odd_count'] = df_fe[red_cols].apply(lambda r: sum(x % 2 != 0 for x in r.astype(int)), axis=1)

    # 区间特征
    for zone, (start, end) in RED_ZONES.items():
        df_fe[f'red_{zone}_count'] = df_fe[red_cols].apply(lambda r: sum(start <= x <= end for x in r.astype(int)), axis=1)
    
    # 重号个数 (与上一期比较)
    df_fe['current_reds_str'] = df_fe[red_cols].astype(int).astype(str).agg(','.join, axis=1)
    df_fe['prev_reds_str'] = df_fe['current_reds_str'].shift(1)
    df_fe['red_repeat_count'] = df_fe.apply(
        lambda r: len(set(int(x) for x in r['prev_reds_str'].split(',')) & set(int(x) for x in r['current_reds_str'].split(',')))
        if pd.notna(r['prev_reds_str']) and pd.notna(r['current_reds_str']) else 0,
        axis=1
    )
    df_fe.drop(columns=['current_reds_str', 'prev_reds_str'], inplace=True, errors='ignore')

    # 连号对数 (基于排序后的红球)
    red_pos_cols = [f'red_pos{i+1}' for i in range(6)] # red_pos*列在clean_and_structure中已创建并排序
    if all(c in df_fe.columns for c in red_pos_cols) and not df_fe.empty:
        # 确保 red_pos_cols 是数值类型
        for col in red_pos_cols:
            df_fe[col] = pd.to_numeric(df_fe[col], errors='coerce')
        df_fe.dropna(subset=red_pos_cols, inplace=True) # 删除转换失败的行

        if not df_fe.empty:
            df_fe['red_consecutive_pairs'] = df_fe.apply(
                lambda r: sum(1 for i in range(5) if r[red_pos_cols[i]] + 1 == r[red_pos_cols[i+1]]), axis=1
            )
        else:
            df_fe['red_consecutive_pairs'] = 0 # If conversion makes it empty, no consecutive pairs
    else:
        df_fe['red_consecutive_pairs'] = 0 # If columns don't exist, no consecutive pairs

    # 蓝球特征
    if 'blue' in df_fe.columns and pd.api.types.is_numeric_dtype(df_fe['blue']):
        df_fe['blue'] = pd.to_numeric(df_fe['blue'], errors='coerce').astype(int) # 确保蓝球为整数
        df_fe['blue_is_odd'] = df_fe['blue'] % 2 != 0 # 奇偶性
        df_fe['blue_is_large'] = df_fe['blue'] > 8 # 大小性 (大于8为大)
        primes = {2, 3, 5, 7, 11, 13}
        df_fe['blue_is_prime'] = df_fe['blue'].apply(lambda x: x in primes) # 质数性
    else:
        # 如果蓝球列缺失或无效，则填充NaN
        df_fe['blue_is_odd'] = np.nan
        df_fe['blue_is_large'] = np.nan
        df_fe['blue_is_prime'] = np.nan
    
    logger.info(f"特征工程完成。生成了 {len(df_fe.columns) - len(df.columns)} 个新特征。")
    return df_fe

def analyze_frequency_omission(df: pd.DataFrame, weights_config: Dict) -> dict:
    """
    分析号码的频率和遗漏数据。
    Args:
        df (pd.DataFrame): 包含历史数据的DataFrame。
        weights_config (Dict): 权重配置（此函数主要用于传递，实际权重未使用）。
    Returns:
        dict: 包含红蓝球频率、当前遗漏、平均间隔、热冷号及历史最大遗漏等信息。
    """
    if df is None or df.empty:
        logger.warning("频率和遗漏分析：输入DataFrame为空。")
        return {}
    
    red_cols = [f'red{i+1}' for i in range(6)]
    most_recent_idx = len(df) - 1 # 最新一期的索引
    if most_recent_idx < 0:
        logger.warning("频率和遗漏分析：DataFrame没有数据。")
        return {}

    # 确定有效的数值型红球列和蓝球列
    num_red_cols = [c for c in red_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    num_blue_col = 'blue' if 'blue' in df.columns and pd.api.types.is_numeric_dtype(df['blue']) else None
    if not num_red_cols and not num_blue_col:
        logger.error("频率和遗漏分析：没有可分析的红球或蓝球列。")
        return {}

    # 计算历史总频率
    all_reds_flat = df[num_red_cols].values.flatten() if num_red_cols else np.array([])
    # 过滤掉非整数值（例如NaN）
    red_freq = Counter(all_reds_flat[~np.isnan(all_reds_flat)].astype(int))
    
    blue_freq = Counter()
    if num_blue_col:
        blue_freq = Counter(df[num_blue_col].dropna().astype(int))

    current_omission = {}
    max_historical_omission_red = {num: 0 for num in RED_BALL_RANGE}
    recent_N_freq_red = {num: 0 for num in RED_BALL_RANGE} # 近N期频率

    if num_red_cols:
        for num in RED_BALL_RANGE:
            # 查找号码在所有红球列中的出现情况
            appearances = (df[num_red_cols].astype(float) == float(num)).any(axis=1)
            app_indices = df.index[appearances].tolist() # 标记号码出现的行索引

            if app_indices:
                current_omission[num] = most_recent_idx - app_indices[-1]
                # 计算最大历史遗漏
                # 遗漏 = (当前期索引 - 上次出现期索引) - 1
                max_o = app_indices[0] # 从数据开始到第一次出现的遗漏期数 (0-indexed)
                for i in range(len(app_indices) - 1):
                    max_o = max(max_o, app_indices[i+1] - app_indices[i] - 1) # 两次出现之间的遗漏
                max_o = max(max_o, most_recent_idx - app_indices[-1]) # 从最后一次出现到现在的遗漏
                max_historical_omission_red[num] = max_o
            else: # 号码从未出现
                current_omission[num] = len(df) # 遗漏期数等于总期数
                max_historical_omission_red[num] = len(df)

        # 计算近N期频率
        recent_df_slice = df.tail(RECENT_FREQ_WINDOW)
        if not recent_df_slice.empty:
            recent_reds_flat = recent_df_slice[num_red_cols].values.flatten()
            recent_freq_counts = Counter(recent_reds_flat[~np.isnan(recent_reds_flat)].astype(int))
            for num in RED_BALL_RANGE:
                recent_N_freq_red[num] = recent_freq_counts.get(num, 0)
    else:
        logger.warning("频率和遗漏分析：没有有效红球列，跳过红球遗漏和近期频率计算。")

    # 计算蓝球当前遗漏
    if num_blue_col:
        for num in BLUE_BALL_RANGE:
            app_indices = df.index[df[num_blue_col].astype(float) == float(num)].tolist()
            if app_indices:
                current_omission[f'blue_{num}'] = most_recent_idx - app_indices[-1]
            else: # 蓝球从未出现
                current_omission[f'blue_{num}'] = len(df)
    else:
        logger.warning("频率和遗漏分析：没有有效蓝球列，跳过蓝球遗漏计算。")

    # 计算平均出现间隔
    avg_interval = {num: len(df) / (red_freq.get(num, 0) + 1e-9) for num in RED_BALL_RANGE}
    for num in BLUE_BALL_RANGE:
        avg_interval[f'blue_{num}'] = len(df) / (blue_freq.get(num, 0) + 1e-9)

    # 识别冷热号 (基于历史总频率)
    red_items = sorted(red_freq.items(), key=lambda item: item[1], reverse=True)
    blue_items = sorted(blue_freq.items(), key=lambda item: item[1], reverse=True)
    
    # 频率最高的20%为热号，最低的20%为冷号
    hot_reds = [n for n, _ in red_items[:max(0, int(len(RED_BALL_RANGE) * 0.2))]]
    cold_reds = [n for n, _ in red_items[max(0, len(red_items) - int(len(RED_BALL_RANGE) * 0.2)):] if n not in hot_reds]
    
    # 蓝球频率最高的30%为热号，最低的30%为冷号
    hot_blues = [n for n, _ in blue_items[:max(0, int(len(BLUE_BALL_RANGE) * 0.3))]]
    cold_blues = [n for n, _ in blue_items[max(0, len(blue_items) - int(len(BLUE_BALL_RANGE) * 0.3)):] if n not in hot_blues]

    return {
        'red_freq': red_freq,
        'blue_freq': blue_freq,
        'current_omission': current_omission,
        'average_interval': avg_interval,
        'hot_reds': hot_reds,
        'cold_reds': cold_reds,
        'hot_blues': hot_blues,
        'cold_blues': cold_blues,
        'max_historical_omission_red': max_historical_omission_red,
        'recent_N_freq_red': recent_N_freq_red
    }

def analyze_patterns(df: pd.DataFrame, weights_config: Dict) -> dict:
    """
    分析历史数据的模式，如和值、跨度、奇偶比、区间分布、蓝球大小等。
    Args:
        df (pd.DataFrame): 包含历史数据的DataFrame。
        weights_config (Dict): 权重配置（此函数主要用于传递，实际权重未使用）。
    Returns:
        dict: 包含各种模式统计信息的字典。
    """
    if df is None or df.empty:
        logger.warning("模式分析：输入DataFrame为空。")
        return {}
    
    res = {}
    
    def safe_mode(series): # 安全地获取众数
        # 确保series不是空的，并且众数计算结果也不是空的
        if not series.empty and not series.mode().empty:
            return int(series.mode().iloc[0])
        return None

    # 和值、跨度统计
    for col, name in [('red_sum', 'sum'), ('red_span', 'span')]:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and not df[col].empty:
            # describe() 返回 Series，to_dict() 转换为字典
            res[f'{name}_stats'] = df[col].describe().to_dict()
            res[f'most_common_{name}'] = safe_mode(df[col])
    
    # 红球奇偶比
    if 'red_odd_count' in df.columns and pd.api.types.is_numeric_dtype(df['red_odd_count']) and not df['red_odd_count'].empty:
        res['most_common_odd_even_count'] = safe_mode(df['red_odd_count'].dropna())
    
    # 红球区间分布
    zone_cols = [f'red_{zone}_count' for zone in RED_ZONES.keys()]
    # 检查所有区间列都存在且为数值类型，并且DataFrame不为空
    if all(c in df.columns and pd.api.types.is_numeric_dtype(df[c]) for c in zone_cols) and not df.empty:
        zc_df = df[zone_cols].dropna().astype(int)
        if not zc_df.empty:
            # 将每行的区间分布转换为元组，然后计数
            dist_counts = zc_df.apply(tuple, axis=1).value_counts()
            res['most_common_zone_distribution'] = dist_counts.index[0] if not dist_counts.empty else None
    
    # 蓝球奇偶、大小分布
    # blue_is_odd 和 blue_is_large 是在 feature_engineer 中创建的布尔列
    for col_name, data_key in [('blue_is_odd', 'blue_odd_counts'), ('blue_is_large', 'blue_large_counts')]:
        if col_name in df.columns and not df[col_name].dropna().empty:
            # value_counts() automatically counts True/False occurrences
            counts = df[col_name].dropna().value_counts()
            # 将布尔键转换为Python内置的bool类型，确保JSON序列化正确
            res[data_key] = {bool(k): int(v) for k, v in counts.items()}
    
    logger.debug(f"模式分析完成。分析了 {len(res)}种模式。")
    return res

def analyze_associations(df: pd.DataFrame, weights_config: Dict) -> pd.DataFrame:
    """
    使用Apriori算法分析红球之间的关联规则。
    Args:
        df (pd.DataFrame): 包含历史数据的DataFrame。
        weights_config (Dict): 包含ARM算法参数的权重。
    Returns:
        pd.DataFrame: 包含关联规则的DataFrame。
    """
    # 从权重配置中获取ARM参数
    min_s = weights_config.get('ARM_MIN_SUPPORT', 0.008)
    min_c = weights_config.get('ARM_MIN_CONFIDENCE', 0.35)
    min_l = weights_config.get('ARM_MIN_LIFT', 1.1)

    if df is None or df.empty or len(df) < 2:
        logger.debug("关联规则分析：输入DataFrame为空或数据量不足 (至少需要2期)。")
        return pd.DataFrame()
    
    red_cols = [f'red{i+1}' for i in range(6)]
    # 检查所有红球列是否存在且为数值类型
    if not all(c in df.columns and pd.api.types.is_numeric_dtype(df[c]) for c in red_cols):
        logger.warning("关联规则分析：红球列缺失或类型无效。")
        return pd.DataFrame()

    tx_df = df.dropna(subset=red_cols).copy() # 删除包含NaN红球的行
    if tx_df.empty:
        logger.warning("关联规则分析：删除红球NaN值后DataFrame为空。")
        return pd.DataFrame()
    
    try:
        tx_df[red_cols] = tx_df[red_cols].astype(int) # 确保红球列为整数
        # 将每期红球组合转换为字符串列表的列表，作为交易数据
        # 例如: [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]] -> [["1", "2", "3", "4", "5", "6"], ...]
        txs = tx_df[red_cols].astype(str).values.tolist()
    except ValueError as e:
        logger.error(f"关联规则分析：红球列转换失败: {e}")
        return pd.DataFrame()
    
    if not txs:
        logger.warning("关联规则分析：没有有效的交易数据。")
        return pd.DataFrame()

    te = TransactionEncoder()
    try:
        te_ary = te.fit_transform(txs)
    except Exception as e:
        logger.error(f"关联规则分析：TransactionEncoder 转换失败: {e}")
        return pd.DataFrame()

    df_oh = pd.DataFrame(te_ary, columns=te.columns_) # 转换为One-Hot编码的DataFrame
    if df_oh.empty:
        logger.warning("关联规则分析：One-Hot编码后DataFrame为空。")
        return pd.DataFrame()
    
    try:
        # 确保最小支持度对于数据集大小是合理的。至少需要出现2次才能形成规则。
        # 如果数据集非常小，min_support 可能需要调整为更小的值
        actual_min_support = max(2 / len(df_oh) if len(df_oh) > 0 else min_s, min_s)
        
        f_items = apriori(df_oh, min_support=actual_min_support, use_colnames=True) # 查找频繁项集
        if f_items.empty:
            logger.debug(f"关联规则分析：未找到频繁项集 (min_support={actual_min_support})。")
            return pd.DataFrame()
        
        rules = association_rules(f_items, metric="lift", min_threshold=min_l) # 生成关联规则
        
        # 按置信度和提升度筛选规则
        if 'confidence' in rules.columns and isinstance(rules['confidence'], pd.Series):
            filtered_rules = rules[rules['confidence'] >= min_c].sort_values(by='lift', ascending=False)
            logger.debug(f"关联规则分析：找到 {len(filtered_rules)} 条关联规则。")
            return filtered_rules
        else:
            logger.warning("关联规则分析：关联规则DataFrame中'confidence'列存在问题。将返回未经置信度筛选的规则。")
            return rules.sort_values(by='lift', ascending=False) if 'lift' in rules.columns else pd.DataFrame()

    except Exception as e_apriori:
        logger.error(f"关联规则分析：Apriori/AssociationRules 执行失败: {e_apriori}")
        return pd.DataFrame()

def get_score_segment(score: float, boundaries: List[int], labels: List[str]) -> str:
    """
    根据分数和预定义的分数段边界确定分数所属的标签。
    Args:
        score (float): 要判断的分数。
        boundaries (List[int]): 分数段的数值边界列表，例如 [0, 25, 50, 75, 100]。
        labels (List[str]): 对应分数段的标签列表，例如 ['0-25', '26-50', ...]。
    Returns:
        str: 分数段标签，如果分数不在任何定义段内则返回 "未知"。
    """
    if score is None or pd.isna(score) or not isinstance(score, (int, float)):
        return "未知" # 处理NaN或其他非数值分数

    # 边界和标签必须匹配
    if len(labels) != len(boundaries) - 1:
        logger.error("get_score_segment: 标签数量与边界数量不匹配，请检查配置。")
        return "未知"

    # 处理浮点数比较的容差
    tolerance = 1e-9

    # 遍历所有分数段
    for i in range(len(boundaries) - 1):
        lower_bound = boundaries[i]
        upper_bound = boundaries[i+1]

        # 特殊处理：最后一个分数段应包含上边界
        if i == len(boundaries) - 2: # 最后一个标签的索引
            if lower_bound <= score <= upper_bound + tolerance:
                return labels[i]
        # 其他分数段：包含下边界，不包含上边界（因为下一个区间的下边界会再次包含它）
        else:
            if lower_bound <= score < upper_bound - tolerance: # 这里的 upper_bound - tolerance 是为了避免重复包含上边界值
                return labels[i]

    # 如果分数刚好等于最后一个边界值，且没有被包含，则将其包含
    if score == boundaries[-1]:
        return labels[-1]

    return "未知" # 如果分数不在任何定义的段内

def analyze_winning_red_ball_score_segments(df: pd.DataFrame, red_ball_scores: dict, score_boundaries: List[int], score_labels: List[str]) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    分析历史上中奖红球的分数段分布。
    Args:
        df (pd.DataFrame): 历史开奖数据DataFrame。
        red_ball_scores (dict): 红球及其对应分数的字典。
        score_boundaries (List[int]): 分数段的数值边界。
        score_labels (List[str]): 分数段的标签。
    Returns:
        Tuple[Dict[str, int], Dict[str, float]]: (各分数段命中次数, 各分数段命中百分比)。
    """
    seg_counts = {label: 0 for label in score_labels}
    total_win_reds = 0
    red_cols = [f'red{i+1}' for i in range(6)]

    if df is None or df.empty or not red_ball_scores or not all(c in df.columns for c in red_cols):
        logger.warning("中奖红球分数段分析：输入数据或分数字典无效。")
        return seg_counts, {label: 0.0 for label in score_labels}

    for _, row in df.iterrows():
        win_reds = []
        valid_row = True
        for c in red_cols:
            val = row.get(c)
            if pd.isna(val) or not pd.api.types.is_numeric_dtype(type(val)):
                valid_row = False
                break
            try:
                num_val = int(float(val)) # 确保转换为整数
                if not (min(RED_BALL_RANGE) <= num_val <= max(RED_BALL_RANGE)):
                    valid_row = False
                    break
                win_reds.append(num_val)
            except (ValueError, TypeError):
                valid_row = False
                break
        
        if not valid_row or len(win_reds) != 6:
            continue

        for ball in win_reds:
            score = red_ball_scores.get(ball)
            if score is not None and pd.notna(score) and isinstance(score, (int, float)):
                segment = get_score_segment(score, score_boundaries, score_labels)
                if segment in seg_counts and segment != "未知":
                    seg_counts[segment] += 1
                    total_win_reds += 1

    seg_pcts = {seg: (cnt / total_win_reds) * 100 if total_win_reds > 0 else 0.0 for seg, cnt in seg_counts.items()}
    logger.debug(f"中奖红球分数段分析完成。总计分析 {total_win_reds} 个红球。")
    return seg_counts, seg_pcts

def create_lagged_features(df: pd.DataFrame, lags: List[int]) -> Optional[pd.DataFrame]:
    """
    为DataFrame创建滞后特征和交互特征。
    Args:
        df (pd.DataFrame): 原始数据DataFrame。
        lags (List[int]): 滞后阶数列表。
    Returns:
        Optional[pd.DataFrame]: 包含滞后和交互特征的DataFrame，或None。
    """
    if df is None or df.empty or not lags:
        logger.warning("滞后特征创建：输入DataFrame为空或滞后列表为空。")
        return None
    
    df_temp = df.copy()

    # 确保蓝球列是数值类型
    if 'blue' in df_temp.columns and not pd.api.types.is_numeric_dtype(df_temp['blue']):
        df_temp['blue'] = pd.to_numeric(df_temp['blue'], errors='coerce')

    # 定义基础特征候选列表
    base_cols_candidates = ['red_sum', 'red_span', 'red_odd_count', 'red_consecutive_pairs', 'red_repeat_count'] + \
                           [f'red_{zone}_count' for zone in RED_ZONES.keys()] + \
                           ['blue', 'blue_is_odd', 'blue_is_large', 'blue_is_prime']
    
    # 筛选出实际存在且为数值型的基础特征列
    existing_base_cols = []
    for col in base_cols_candidates:
        if col in df_temp.columns:
            if pd.api.types.is_bool_dtype(df_temp[col].dtype):
                df_temp[col] = df_temp[col].astype(int) # 布尔值转换为0/1
                existing_base_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df_temp[col].dtype):
                existing_base_cols.append(col)
            else:
                try: # 尝试将非数值列转换为数值，如果成功则加入
                    df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
                    if pd.api.types.is_numeric_dtype(df_temp[col].dtype):
                        existing_base_cols.append(col)
                except Exception:
                    logger.debug(f"特征 '{col}' 无法转换为数值类型，跳过。")
                    pass
    
    if not existing_base_cols:
        logger.warning("滞后特征创建：没有可用于生成滞后特征的数值列。")
        return None

    # 创建交互特征
    interaction_cols = []
    for col1, col2 in ML_INTERACTION_PAIRS:
        if col1 in df_temp.columns and col2 in df_temp.columns and \
           pd.api.types.is_numeric_dtype(df_temp[col1]) and pd.api.types.is_numeric_dtype(df_temp[col2]):
            interaction_col_name = f'{col1}_x_{col2}'
            df_temp[interaction_col_name] = df_temp[col1] * df_temp[col2]
            interaction_cols.append(interaction_col_name)

    for col_s in ML_INTERACTION_SELF:
        if col_s in df_temp.columns and pd.api.types.is_numeric_dtype(df_temp[col_s]):
            interaction_col_name = f'{col_s}_sq'
            df_temp[interaction_col_name] = df_temp[col_s] ** 2
            interaction_cols.append(interaction_col_name)

    # 包含基础特征和交互特征的完整特征列表
    all_features_for_lagging = existing_base_cols + interaction_cols
    
    # 从这些特征中生成滞后特征
    df_lagged = df_temp[all_features_for_lagging].copy()
    
    final_lagged_feature_names = []
    for lag_val in sorted(lags): # 确保按滞后阶数排序
        if lag_val > 0:
            for col in all_features_for_lagging:
                lagged_col_name = f'{col}_lag{lag_val}'
                df_lagged[lagged_col_name] = df_lagged[col].shift(lag_val)
                final_lagged_feature_names.append(lagged_col_name)

    df_lagged.dropna(inplace=True) # 删除包含NaN（因shift操作产生）的行
    
    if df_lagged.empty:
        logger.warning("滞后特征创建：生成滞后特征后DataFrame为空。")
        return None
    
    # 返回只包含滞后特征的DataFrame
    if not final_lagged_feature_names:
        logger.warning("滞后特征创建：未能生成任何滞后特征。")
        return None

    logger.debug(f"滞后特征创建完成。生成了 {len(final_lagged_feature_names)} 个特征。")
    return df_lagged[final_lagged_feature_names]

def train_single_model(model_type, ball_type_str, ball_number, X_train, y_train, params, min_pos_samples,
                       LGBMClassifier_ref, SVC_ref, StandardScaler_ref, Pipeline_ref, LogisticRegression_ref, XGBClassifier_ref):
    """
    训练单个机器学习模型（在独立的进程中运行）。
    此函数设计为可由 concurrent.futures.ProcessPoolExecutor 调用，因此需要传入类引用。
    Args:
        model_type (str): 模型类型 ('lgbm', 'xgb', 'logreg', 'svc')。
        ball_type_str (str): 球类型 ('红' 或 '蓝')。
        ball_number (int): 球号码。
        X_train (pd.DataFrame): 训练特征。
        y_train (pd.Series): 训练目标 (0/1)。
        params (Dict): 模型参数。
        min_pos_samples (int): 训练所需的最小正样本数。
        LGBMClassifier_ref, SVC_ref, ...: 模型的类引用，因为进程池无法直接序列化类实例。
    Returns:
        Tuple[Any, str]: (训练好的模型实例, 模型键), 如果训练失败则返回 (None, None)。
    """
    if y_train.sum() < min_pos_samples or len(y_train.unique()) < 2:
        # logger.debug(f"跳过 {model_type} 模型训练 for {ball_type_str} {ball_number}: 正样本数不足或目标变量不平衡。")
        return None, None
    
    model_key = f'{model_type}_{ball_number}'
    model_params = params.copy()

    positive_count = y_train.sum()
    negative_count = len(y_train) - positive_count
    
    # 对于LGBM和XGB，使用scale_pos_weight处理类别不平衡
    scale_pos_weight_val = negative_count / (positive_count + 1e-9) if positive_count > 0 else 1.0
    # 对于LogisticRegression和SVC，使用class_weight处理类别不平衡
    class_weight_val = 'balanced' if positive_count > 0 and negative_count > 0 else None

    model = None
    try:
        if model_type == 'lgbm':
            model_params['scale_pos_weight'] = scale_pos_weight_val
            model = LGBMClassifier_ref(**model_params)
            model.fit(X_train, y_train)
        elif model_type == 'xgb':
            model_params['scale_pos_weight'] = scale_pos_weight_val
            model = XGBClassifier_ref(**model_params)
            model.fit(X_train, y_train)
        elif model_type == 'logreg':
            if class_weight_val: model_params['class_weight'] = class_weight_val
            model_params.pop('scale_pos_weight', None) # 确保不传递不兼容的参数
            # Logistic Regression 通常需要特征缩放，所以使用Pipeline
            model = Pipeline_ref([('scaler', StandardScaler_ref()), ('logreg', LogisticRegression_ref(**model_params))])
            model.fit(X_train, y_train)
        elif model_type == 'svc':
            if class_weight_val: model_params['class_weight'] = class_weight_val
            model_params.pop('scale_pos_weight', None) # 确保不传递不兼容的参数
            svc_actual_params = model_params.copy()
            svc_actual_params['probability'] = True # SVC需要显式设置才能predict_proba
            
            # SVC 通常需要特征缩放，所以使用Pipeline
            model = Pipeline_ref([('scaler', StandardScaler_ref()), ('svc', SVC_ref(**svc_actual_params))])
            model.fit(X_train, y_train)
            
            # 再次确认SVC是否成功启用了概率预测
            svc_estimator = model.named_steps.get('svc')
            if not (svc_estimator and hasattr(svc_estimator, 'probability') and svc_estimator.probability):
                logger.debug(f"SVC for {ball_type_str} {ball_number} did not enable probability correctly.")
                model = None # 如果未能启用概率，则认为模型无效
        
        return model, model_key
    except Exception as e_train:
        logger.debug(f"训练 {model_type} 模型 for {ball_type_str} {ball_number} 失败: {e_train}")
        return None, None

def train_prediction_models(df_train_raw: pd.DataFrame, ml_lags_list: List[int], weights_config: Dict) -> Optional[dict]:
    """
    为每个球号训练多种类型的预测模型。
    Args:
        df_train_raw (pd.DataFrame): 用于训练的原始历史数据。
        ml_lags_list (List[int]): 机器学习模型使用的滞后特征阶数。
        weights_config (Dict): 权重配置。
    Returns:
        Optional[dict]: 包含所有训练好的模型和特征列的字典，或None。
    """
    # 1. 创建滞后特征
    X = create_lagged_features(df_train_raw.copy(), ml_lags_list)
    if X is None or X.empty:
        logger.warning("ML模型训练：滞后特征创建失败或为空。")
        return None

    # 确保目标数据与特征数据对齐（按索引）
    target_df = df_train_raw.loc[X.index].copy()
    if target_df.empty:
        logger.warning("ML模型训练：特征与目标对齐后DataFrame为空。")
        return None

    red_cols = [f'red{i+1}' for i in range(6)]
    if not all(c in target_df.columns for c in red_cols + ['blue']):
        logger.error("ML模型训练：目标DataFrame中缺少球号列。")
        return None
    try:
        # 确保红球和蓝球列是整数类型，以便后续进行 == 比较
        for col in red_cols + ['blue']:
            target_df[col] = pd.to_numeric(target_df[col], errors='coerce').astype(int)
    except (ValueError, TypeError) as e:
        logger.error(f"ML模型训练：转换球号列为整数失败: {e}")
        return None

    trained_models = {'red': {}, 'blue': {}, 'feature_cols': X.columns.tolist()}
    min_pos = MIN_POSITIVE_SAMPLES_FOR_ML

    futures_map = {} # 存储Future对象及其对应的球信息
    num_cpus = os.cpu_count()
    max_workers = num_cpus if num_cpus and num_cpus > 1 else 1 # 根据CPU核数设置并行工作者数量

    # 使用进程池并行训练模型
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for ball_num in RED_BALL_RANGE:
            # 构建红球的二元目标变量：当前期是否包含该红球
            y_red = target_df[red_cols].apply(lambda row: ball_num in row.values, axis=1).astype(int)
            for mt, mp in [('lgbm', LGBM_PARAMS), ('xgb', XGB_PARAMS), ('logreg', LOGISTIC_REG_PARAMS), ('svc', SVC_PARAMS)]:
                # 提交训练任务到进程池
                f = executor.submit(train_single_model, mt, '红', ball_num, X, y_red, mp, min_pos,
                                    LGBMClassifier, SVC, StandardScaler, Pipeline, LogisticRegression, xgb.XGBClassifier)
                futures_map[f] = ('red', mt, ball_num) # 记录Future对应的球信息

        for ball_num in BLUE_BALL_RANGE:
            # 构建蓝球的二元目标变量：当前期是否是该蓝球
            y_blue = (target_df['blue'] == ball_num).astype(int)
            for mt, mp in [('lgbm', LGBM_PARAMS), ('xgb', XGB_PARAMS), ('logreg', LOGISTIC_REG_PARAMS), ('svc', SVC_PARAMS)]:
                f = executor.submit(train_single_model, mt, '蓝', ball_num, X, y_blue, mp, min_pos,
                                    LGBMClassifier, SVC, StandardScaler, Pipeline, LogisticRegression, xgb.XGBClassifier)
                futures_map[f] = ('blue', mt, ball_num)

    models_trained_count = 0
    # 收集训练结果
    for future in concurrent.futures.as_completed(futures_map):
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
    """
    使用训练好的模型预测下一次开奖的号码出现概率。
    Args:
        df_historical (pd.DataFrame): 历史数据DataFrame。
        trained_models (Optional[dict]): 包含训练好的模型的字典。
        ml_lags_list (List[int]): 机器学习模型使用的滞后特征阶数。
        weights_config (Dict): 权重配置。
    Returns:
        Dict: 包含红蓝球预测概率的字典。
    """
    probs = {'red': {}, 'blue': {}}
    if not trained_models or df_historical is None or df_historical.empty:
        logger.warning("ML预测：无训练好的模型或历史数据。")
        return probs

    feat_cols = trained_models.get('feature_cols')
    if not feat_cols:
        logger.warning("ML预测：trained_models中无feature_cols，无法确定预测特征。")
        return probs

    max_hist_lag = max(ml_lags_list) if ml_lags_list else 0
    # 预测需要足够长的历史数据来生成滞后特征
    if len(df_historical) < max_hist_lag + 1:
        logger.warning(f"ML预测：历史数据不足 ({len(df_historical)}期)，无法满足滞后 ({max_hist_lag})。至少需要 {max_hist_lag + 1} 条。")
        return probs

    # 为预测目标期（最近一期之后的一期）生成特征，所以取历史数据的尾部进行特征生成
    # create_lagged_features 会处理内部的shift，我们只需要传入足够的原始数据
    predict_X_raw_data = df_historical.tail(max_hist_lag + 1).copy()
    predict_X = create_lagged_features(predict_X_raw_data, ml_lags_list)
    
    if predict_X is None or predict_X.empty:
        logger.warning("ML预测：为预测生成滞后特征失败或结果为空。")
        return probs
    # 确保预测输入只有一个样本（即下一期的数据）
    if len(predict_X) != 1:
        logger.error(f"ML预测：预测特征应为1行，实际得到 {len(predict_X)} 行。")
        return probs

    try:
        # 确保预测特征列与训练特征列一致，并处理可能出现的NaN值
        predict_X = predict_X.reindex(columns=feat_cols, fill_value=0)
        for col in predict_X.columns:
            predict_X[col] = pd.to_numeric(predict_X[col], errors='coerce')
        predict_X.fillna(0, inplace=True) # 用0填充剩余的NaN
        if predict_X.isnull().values.any():
            logger.error("ML预测：处理后预测特征中仍存在NaN。")
            return probs
    except Exception as e_pred_preprocess:
        logger.error(f"ML预测：预处理预测特征时出错: {e_pred_preprocess}.")
        return probs

    # 对红球和蓝球分别进行预测
    for ball_type_key, ball_val_range, models_sub_dict in [('red', RED_BALL_RANGE, trained_models.get('red', {})),
                                                           ('blue', BLUE_BALL_RANGE, trained_models.get('blue', {}))]:
        if not models_sub_dict:
            continue
        
        for ball_val in ball_val_range:
            ball_preds = [] # 存储不同模型对同一号码的预测概率
            for model_variant in ['lgbm', 'xgb', 'logreg', 'svc']:
                model_instance = models_sub_dict.get(f'{model_variant}_{ball_val}')
                # 检查模型是否存在且支持概率预测
                if model_instance and hasattr(model_instance, 'predict_proba'):
                    try:
                        # predict_proba 返回 [P(class 0), P(class 1)]，我们取P(class 1)
                        proba = model_instance.predict_proba(predict_X)[0][1]
                        ball_preds.append(proba)
                    except Exception as e_proba:
                        logger.debug(f"ML预测：{ball_type_key} {ball_val} {model_variant}的predict_proba失败: {e_proba}")
            
            if ball_preds:
                # 平均所有有效模型的预测概率作为最终概率
                probs[ball_type_key][ball_val] = np.mean(ball_preds)
    
    logger.debug(f"ML预测完成。红球预测概率数: {len(probs['red'])}, 蓝球预测概率数: {len(probs['blue'])}。")
    return probs


def calculate_scores(freq_omission_data: dict, pattern_analysis_data: dict, predicted_probabilities: dict, weights_config: Dict) -> dict:
    """
    根据频率、遗漏、模式和ML预测概率计算每个球的分数。
    Args:
        freq_omission_data (dict): 频率和遗漏分析结果。
        pattern_analysis_data (dict): 模式分析结果。
        predicted_probabilities (dict): ML预测概率结果。
        weights_config (Dict): 包含所有评分权重的配置。
    Returns:
        dict: 包含红球和蓝球分数的字典。
    """
    r_scores, b_scores = {}, {}

    # 获取频率和遗漏数据
    r_freq = freq_omission_data.get('red_freq', {}); b_freq = freq_omission_data.get('blue_freq', {})
    omission = freq_omission_data.get('current_omission', {}); avg_int = freq_omission_data.get('average_interval', {})
    max_hist_omission_r = freq_omission_data.get('max_historical_omission_red', {})
    recent_N_freq_r = freq_omission_data.get('recent_N_freq_red', {})

    # 将频率转换为Series并进行排名，用于计算频率得分
    r_freq_series = pd.Series(r_freq).reindex(list(RED_BALL_RANGE), fill_value=0)
    # rank(method='min', ascending=False) 会给最高频率的号码最低的排名（例如1），次高频率的号码次低排名
    r_freq_rank = r_freq_series.rank(method='min', ascending=False)
    
    b_freq_series = pd.Series(b_freq).reindex(list(BLUE_BALL_RANGE), fill_value=0)
    b_freq_rank = b_freq_series.rank(method='min', ascending=False)

    # 获取ML预测概率
    r_pred_probs = predicted_probabilities.get('red', {}); b_pred_probs = predicted_probabilities.get('blue', {})
    
    max_r_rank = len(RED_BALL_RANGE) # 红球总数
    max_b_rank = len(BLUE_BALL_RANGE) # 蓝球总数

    # 归一化近期频率，使其在0-1之间，用于计算近期频率得分
    recent_freq_values = [v for v in recent_N_freq_r.values() if v is not None]
    min_rec_freq = min(recent_freq_values) if recent_freq_values else 0
    max_rec_freq = max(recent_freq_values) if recent_freq_values else 0

    # --- 红球分数计算 ---
    for num in RED_BALL_RANGE:
        # 频率得分：频率排名越靠前（数值越小），得分越高
        # (max_r_rank - (rank-1))/max_r_rank 将排名转换为 0-1 的比例，排名越前值越大
        freq_s = max(0, (max_r_rank - (r_freq_rank.get(num, max_r_rank+1)-1))/max_r_rank * weights_config['FREQ_SCORE_WEIGHT'])
        
        # 遗漏得分：当前遗漏与平均间隔的偏差越小，得分越高 (使用高斯函数)
        dev = omission.get(num, max_r_rank*2) - avg_int.get(num, max_r_rank*2)
        omit_s = max(0, weights_config['OMISSION_SCORE_WEIGHT'] * np.exp(-0.005 * dev**2))

        # 历史最大遗漏比率得分：当前遗漏与历史最大遗漏的比率。
        # 如果当前遗漏接近历史最大遗漏，得分会高。如果当前遗漏远超历史最大遗漏，也可能获得额外奖励。
        max_o = max_hist_omission_r.get(num, 0)
        cur_o = omission.get(num, 0)
        max_omit_ratio_s = 0
        if max_o > 0:
            ratio_o = cur_o / max_o
            max_omit_ratio_s = max(0, min(1.0, ratio_o)) * weights_config['MAX_OMISSION_RATIO_SCORE_WEIGHT_RED']
            if ratio_o > 1.2: # 如果当前遗漏显著超过历史最大，额外奖励
                max_omit_ratio_s *= 1.2
            if ratio_o < 0.2: # 如果当前遗漏远低于历史最大，略微惩罚
                max_omit_ratio_s *= 0.5
        else: # 如果从未出现过（max_o为0），且当前遗漏大于0，则给满分，否则为0
            max_omit_ratio_s = weights_config['MAX_OMISSION_RATIO_SCORE_WEIGHT_RED'] if cur_o > 0 else 0

        # 近期频率得分：近期频率越高，得分越高
        rec_f = recent_N_freq_r.get(num, 0)
        norm_rec_f_score = 0
        if max_rec_freq > min_rec_freq:
            norm_rec_f_score = (rec_f - min_rec_freq) / (max_rec_freq - min_rec_freq)
        elif max_rec_freq > 0 : # 如果所有近期频率相同但大于0
             norm_rec_f_score = 0.5 if rec_f > 0 else 0
        recent_freq_s = max(0, norm_rec_f_score * weights_config['RECENT_FREQ_SCORE_WEIGHT_RED'])

        # ML预测概率得分
        ml_s = max(0, r_pred_probs.get(num, 0.0) * weights_config['ML_PROB_SCORE_WEIGHT_RED'])

        r_scores[num] = freq_s + omit_s + ml_s + max_omit_ratio_s + recent_freq_s

    # --- 蓝球分数计算 ---
    for num in BLUE_BALL_RANGE:
        # 频率得分
        freq_s = max(0, (max_b_rank - (b_freq_rank.get(num, max_b_rank+1)-1))/max_b_rank * weights_config['BLUE_FREQ_SCORE_WEIGHT'])
        # 遗漏得分
        dev = omission.get(f'blue_{num}', max_b_rank*2) - avg_int.get(f'blue_{num}', max_b_rank*2)
        omit_s = max(0, weights_config['BLUE_OMISSION_SCORE_WEIGHT'] * np.exp(-0.01 * dev**2))
        # ML预测概率得分
        ml_s = max(0, b_pred_probs.get(num, 0.0) * weights_config['ML_PROB_SCORE_WEIGHT_BLUE'])
        
        b_scores[num] = freq_s + omit_s + ml_s

    # 对所有球的分数进行0-100的归一化
    all_s_vals = [s for s in list(r_scores.values()) + list(b_scores.values()) if pd.notna(s) and np.isfinite(s)]
    if all_s_vals:
        min_s_val, max_s_val = min(all_s_vals), max(all_s_vals) # <-- 修正
        if (max_s_val - min_s_val) > 1e-9: # 避免除以零
            r_scores = {n: max(0,min(100,(s-min_s_val)/(max_s_val-min_s_val)*100)) if pd.notna(s) and np.isfinite(s) else 0 for n,s in r_scores.items()}
            b_scores = {n: max(0,min(100,(s-min_s_val)/(max_s_val-min_s_val)*100)) if pd.notna(s) and np.isfinite(s) else 0 for n,s in b_scores.items()}
        else: # 所有分数都相同，或只有一个有效分数，则都设为50
            r_scores = {n:50.0 for n in RED_BALL_RANGE}; b_scores = {n:50.0 for n in BLUE_BALL_RANGE}
    else: # 没有有效分数，都设为0
        r_scores = {n:0.0 for n in RED_BALL_RANGE}; b_scores = {n:0.0 for n in BLUE_BALL_RANGE}

    logger.debug("球号分数计算完成。")
    return {'red_scores': r_scores, 'blue_scores': b_scores}

def get_combo_properties(red_balls: List[int]) -> Dict:
    """
    获取红球组合的属性 (和值, 奇数个数, 区间分布)。
    Args:
        red_balls (List[int]): 一个红球组合列表。
    Returns:
        Dict: 包含组合属性的字典。
    """
    props = {}
    props['sum'] = sum(red_balls)
    props['odd_count'] = sum(x % 2 != 0 for x in red_balls)
    
    zones_count = [0,0,0] # 对应 Zone1, Zone2, Zone3
    for i, (zone_name, (start, end)) in enumerate(RED_ZONES.items()):
        zones_count[i] = sum(1 for ball_num in red_balls if start <= ball_num <= end)
    props['zone_dist'] = tuple(zones_count)
    return props

def generate_combinations(scores_data: dict, pattern_analysis_data: dict, association_rules_df: pd.DataFrame,
                          winning_segment_percentages: Dict[str, float], weights_config: Dict) -> Tuple[List[Dict], List[str]]:
    """
    根据球号分数、模式和关联规则生成推荐组合。
    Args:
        scores_data (dict): 包含红蓝球分数的字典。
        pattern_analysis_data (dict): 模式分析结果。
        association_rules_df (pd.DataFrame): 关联规则DataFrame。
        winning_segment_percentages (Dict[str, float]): 历史中奖红球分数段百分比。
        weights_config (Dict): 权重配置。
    Returns:
        Tuple[List[Dict], List[str]]: (推荐组合列表, 推荐组合的格式化字符串列表)。
    """
    num_combinations_to_generate = weights_config.get('NUM_COMBINATIONS_TO_GENERATE', 10)
    target_red_pool_size = weights_config.get('TOP_N_RED_FOR_CANDIDATE', 18)
    top_n_blue = weights_config.get('TOP_N_BLUE_FOR_CANDIDATE', 8)

    min_different_reds = weights_config.get('DIVERSITY_MIN_DIFFERENT_REDS', 3)
    max_common_reds_allowed = 6 - min_different_reds # 两个组合之间最多允许相同的红球数量
    diversity_max_attempts = weights_config.get('DIVERSITY_SELECTION_MAX_ATTEMPTS', 20)
    
    diversity_sum_diff_thresh = weights_config.get('DIVERSITY_SUM_DIFF_THRESHOLD', 15)
    diversity_oddeven_diff_min = weights_config.get('DIVERSITY_ODDEVEN_DIFF_MIN_COUNT', 1)
    diversity_zone_dist_min_diff_zones = weights_config.get('DIVERSITY_ZONE_DIST_MIN_DIFF_ZONES', 2)

    prop_h = weights_config.get('CANDIDATE_POOL_PROPORTIONS_HIGH', 0.5)
    prop_m = weights_config.get('CANDIDATE_POOL_PROPORTIONS_MEDIUM', 0.3)
    prop_l = max(0, 1.0 - prop_h - prop_m) # 低分段比例
    segment_proportions_config = {'High': prop_h, 'Medium': prop_m, 'Low': prop_l} # 基于配置的比例
    min_per_segment = weights_config.get('CANDIDATE_POOL_MIN_PER_SEGMENT', 2)

    # Initialize win_seg_pcts_local at the beginning of the function
    win_seg_pcts_local = winning_segment_percentages
    # Add a safety check for the passed argument type
    if not isinstance(win_seg_pcts_local, dict):
        logger.error(f"Passed winning_segment_percentages is not a dictionary: {type(win_seg_pcts_local)}. Using empty dict for safety.")
        win_seg_pcts_local = {} # Fallback to empty dict to prevent further errors

    r_scores = scores_data.get('red_scores', {})
    b_scores = scores_data.get('blue_scores', {})

    # 1. 初始化 initial_red_candidate_pool 和 final_red_candidate_pool 确保其在任何情况下都被定义
    initial_red_candidate_pool: List[int] = []
    final_red_candidate_pool: List[int] = []

    if r_scores:
        # 1. 根据分数段划分红球
        segmented_balls_dict = {name: [] for name in CANDIDATE_POOL_SEGMENT_NAMES}
        for ball_num, score_val in r_scores.items():
            if score_val > CANDIDATE_POOL_SCORE_THRESHOLDS['High']:
                segmented_balls_dict['High'].append(ball_num)
            elif score_val > CANDIDATE_POOL_SCORE_THRESHOLDS['Medium']:
                segmented_balls_dict['Medium'].append(ball_num)
            else:
                segmented_balls_dict['Low'].append(ball_num)
        
        # 2. 对每个分数段的球按分数降序排序
        for seg_name_iter in CANDIDATE_POOL_SEGMENT_NAMES:
            segment_balls_with_scores = {b: r_scores.get(b, 0) for b in segmented_balls_dict[seg_name_iter]}
            segmented_balls_dict[seg_name_iter] = [b for b, _ in sorted(segment_balls_with_scores.items(), key=lambda x: x[1], reverse=True)]

        temp_pool_set = set() # 用于收集最终候选池的集合，避免重复

        # 3. 根据历史中奖红球分数段分布调整候选池抽样比例
        is_winning_segment_percentages_valid = bool(
            win_seg_pcts_local and
            sum(win_seg_pcts_local.values()) > 1e-6 and
            all(label in win_seg_pcts_local for label in SCORE_SEGMENT_LABELS)
        )
        
        base_proportions_for_pool = segment_proportions_config # 默认使用配置的比例
        if is_winning_segment_percentages_valid:
            try:
                adjusted_props_map = {'High': 0.0, 'Medium': 0.0, 'Low': 0.0}
                for seg_label, pct_val in win_seg_pcts_local.items():
                    try:
                        lower_str, _ = seg_label.split('-')
                        lower_bound = int(lower_str)
                        
                        if lower_bound >= CANDIDATE_POOL_SCORE_THRESHOLDS['High']:
                            adjusted_props_map['High'] = adjusted_props_map.get('High', 0.0) + pct_val
                        elif lower_bound >= CANDIDATE_POOL_SCORE_THRESHOLDS['Medium']:
                            adjusted_props_map['Medium'] = adjusted_props_map.get('Medium', 0.0) + pct_val
                        else:
                            adjusted_props_map['Low'] = adjusted_props_map.get('Low', 0.0) + pct_val
                    except ValueError:
                        logger.warning(f"无法解析分数段标签 '{seg_label}'，将均匀分配其历史贡献。")
                        # If parsing fails, distribute percentage evenly across segments as a fallback
                        for seg_name in CANDIDATE_POOL_SEGMENT_NAMES:
                            adjusted_props_map[seg_name] = adjusted_props_map.get(seg_name, 0.0) + pct_val / len(CANDIDATE_POOL_SEGMENT_NAMES)

                total_adj_prop = sum(adjusted_props_map.values())
                if total_adj_prop > 1e-6:
                    base_proportions_for_pool = {k: v / total_adj_prop for k, v in adjusted_props_map.items()}
                    logger.debug(f"候选池构建: 调整后的各分段选球基础比例 (基于历史中奖): {base_proportions_for_pool}")
                else:
                    logger.debug("候选池构建: 历史分数段调整后总比例过小，使用均匀概率因子。")
                    base_proportions_for_pool = {lbl: 1.0/len(CANDIDATE_POOL_SEGMENT_NAMES) for lbl in CANDIDATE_POOL_SEGMENT_NAMES}
            except Exception as e_adj:
                logger.warning(f"候选池构建: 历史中奖分布调整过程中发生错误 ({e_adj})。将使用配置的固定比例。")
                base_proportions_for_pool = segment_proportions_config

        # 4. 计算每个分数段需要选取的球数
        num_to_pick_segments = {}
        for seg_name_iter in CANDIDATE_POOL_SEGMENT_NAMES:
            prop = base_proportions_for_pool.get(seg_name_iter, segment_proportions_config[seg_name_iter])
            num_to_pick_segments[seg_name_iter] = max(min_per_segment, int(round(prop * target_red_pool_size)))
        
        current_total_pick = sum(num_to_pick_segments.values())
        if current_total_pick > target_red_pool_size:
            scale_down_factor = (target_red_pool_size / current_total_pick) if current_total_pick > 0 else 1.0
            for seg_name_iter_scale in num_to_pick_segments:
                num_to_pick_segments[seg_name_iter_scale] = max(min_per_segment, int(round(num_to_pick_segments[seg_name_iter_scale] * scale_down_factor)))
            logger.debug(f"候选池构建: 调整后各分段计划选取数量 (总量控制后): {num_to_pick_segments}")

        # 5. 从每个分数段中选取球添加到候选池
        for seg_name_select in CANDIDATE_POOL_SEGMENT_NAMES:
            balls_from_segment = segmented_balls_dict[seg_name_select]
            num_to_add = num_to_pick_segments.get(seg_name_select, min_per_segment)
            added_count = 0
            for ball_to_add in balls_from_segment:
                if len(temp_pool_set) >= target_red_pool_size: break # 达到目标总数
                if ball_to_add not in temp_pool_set and added_count < num_to_add:
                    temp_pool_set.add(ball_to_add)
                    added_count += 1
            if len(temp_pool_set) >= target_red_pool_size: break # 达到目标总数

        initial_red_candidate_pool = list(temp_pool_set) # <-- 在此分支中赋值

        # 6. 如果分段选取后球数不足目标，从总分排序中补充
        if len(initial_red_candidate_pool) < target_red_pool_size:
            logger.debug(f"候选池构建: 分段选取后球数不足 ({len(initial_red_candidate_pool)}/{target_red_pool_size})，从总分排序中补充。")
            all_sorted_reds_overall = [n for n, _ in sorted(r_scores.items(), key=lambda item: item[1], reverse=True)]
            for ball_fill in all_sorted_reds_overall:
                if len(initial_red_candidate_pool) >= target_red_pool_size: break
                if ball_fill not in initial_red_candidate_pool:
                    initial_red_candidate_pool.append(ball_fill)
        
        # 7. 如果最终候选池不足6个球，进行紧急补充
        if len(initial_red_candidate_pool) < 6:
            logger.warning(f"红球初始候选池只有 {len(initial_red_candidate_pool)} 个球。紧急扩展到至少6个。")
            current_pool_set_fallback = set(initial_red_candidate_pool)
            all_sorted_reds_overall_fallback = [n for n, _ in sorted(r_scores.items(), key=lambda item: item[1], reverse=True)] if r_scores else list(RED_BALL_RANGE)
            
            for ball_fallback in all_sorted_reds_overall_fallback:
                if len(initial_red_candidate_pool) >= 6: break
                if ball_fallback not in current_pool_set_fallback:
                    initial_red_candidate_pool.append(ball_fallback)
                    current_pool_set_fallback.add(ball_fallback)
            
            if len(initial_red_candidate_pool) < 6: # 如果仍不足，从全部红球中随机补充
                remaining_needed_fb = 6 - len(initial_red_candidate_pool)
                # Ensure there are enough available balls to sample without replacement
                fallback_balls_list = [b for b in RED_BALL_RANGE if b not in current_pool_set_fallback]
                if len(fallback_balls_list) >= remaining_needed_fb:
                    initial_red_candidate_pool.extend(random.sample(fallback_balls_list, remaining_needed_fb))
                else: # Fallback pool itself is too small, add all remaining and warn
                    initial_red_candidate_pool.extend(fallback_balls_list)
                    logger.warning(f"无法从红球范围中补充足够的球到6个，当前 {len(initial_red_candidate_pool)}。")

                logger.debug(f"候选池构建: 紧急填充后红球池大小: {len(initial_red_candidate_pool)}")

        final_red_candidate_pool = list(initial_red_candidate_pool) # <-- 将最终结果赋值给 final_red_candidate_pool

    else: # 如果r_scores为空，无法进行基于分数的构建，使用随机红球
        logger.debug("候选池构建: r_scores 为空，无法进行基于分数的红球候选池构建。将使用随机红球。")
        # 直接给 initial_red_candidate_pool 和 final_red_candidate_pool 赋值
        initial_red_candidate_pool = random.sample(list(RED_BALL_RANGE), k=min(target_red_pool_size, len(RED_BALL_RANGE)))
        if len(initial_red_candidate_pool) < 6: # 确保至少有6个红球
            remaining_needed_fb = 6 - len(initial_red_candidate_pool)
            fill_pool_candidates_fallback = [b for b in RED_BALL_RANGE if b not in initial_red_candidate_pool]
            if len(fill_pool_candidates_fallback) >= remaining_needed_fb:
                initial_red_candidate_pool.extend(random.sample(fill_pool_candidates_fallback, remaining_needed_fb))
            else:
                initial_red_candidate_pool.extend(fill_pool_candidates_fallback)
                logger.warning(f"紧急情况：红球池不足6个，填充所有可用球。当前 {len(initial_red_candidate_pool)}")
        final_red_candidate_pool = list(initial_red_candidate_pool) # <-- 在此分支中赋值

    r_cand_pool = final_red_candidate_pool # 确保 r_cand_pool 总是被赋值

    logger.debug(f"候选池构建: 最终用于抽样的红球候选池 (r_cand_pool) 大小: {len(r_cand_pool)}, 内容: {sorted(r_cand_pool)}")

    # --- 红球候选池层面的反向思维 ---
    reverse_iterations_count = weights_config.get('REVERSE_THINKING_ITERATIONS', 0)
    balls_to_remove_per_iter = weights_config.get('REVERSE_THINKING_RED_BALLS_TO_REMOVE_PER_ITER', 0)

    if reverse_iterations_count > 0 and balls_to_remove_per_iter > 0 and r_scores and len(final_red_candidate_pool) > 6:
        logger.debug(f"候选池构建: 应用反向思维: 迭代次数={reverse_iterations_count}, 每次移除={balls_to_remove_per_iter}")
        for i in range(reverse_iterations_count):
            if len(final_red_candidate_pool) <= max(6, balls_to_remove_per_iter): # 至少保留6个球，或满足移除的数量
                logger.debug(f"  反向思维迭代 {i+1}: 池中球数 ({len(final_red_candidate_pool)}) 不足以移除 {balls_to_remove_per_iter} 并保留至少6个。中止反向思维。")
                break
            balls_with_scores_in_pool = [(ball, r_scores.get(ball, 0)) for ball in final_red_candidate_pool]
            sorted_current_pool_by_score = sorted(balls_with_scores_in_pool, key=lambda x: x[1], reverse=True)
            
            # 移除最高分的球
            balls_to_discard_this_iter = [ball for ball, _ in sorted_current_pool_by_score[:balls_to_remove_per_iter]]
            final_red_candidate_pool = [ball for ball in final_red_candidate_pool if ball not in balls_to_discard_this_iter]
            logger.debug(f"  反向思维迭代 {i+1}: 移除了 {balls_to_discard_this_iter}。当前池大小: {len(final_red_candidate_pool)}")
        
        # 反向思维后如果池过小，需要补充
        if len(final_red_candidate_pool) < 6:
            logger.warning(f"候选池构建: 反向思维后红球池过小 ({len(final_red_candidate_pool)})。将从初始池或全部红球中补充。")
            temp_set_final_reverse = set(final_red_candidate_pool)
            
            # 优先从 initial_red_candidate_pool 中补充 (它在 if r_scores 或 else 块中被正确初始化)
            for ball_re_add in initial_red_candidate_pool: # <-- 这里现在应该能够安全引用
                if len(final_red_candidate_pool) >= 6: break
                if ball_re_add not in temp_set_final_reverse:
                    final_red_candidate_pool.append(ball_re_add)
                    temp_set_final_reverse.add(ball_re_add)
            
            if len(final_red_candidate_pool) < 6: # 如果仍不足，从全部红球范围中补充
                remaining_needed_fb = 6 - len(final_red_candidate_pool)
                fallback_balls_list = [b for b in RED_BALL_RANGE if b not in temp_set_final_reverse] # Ensure not picking already picked balls
                if len(fallback_balls_list) >= remaining_needed_fb:
                    final_red_candidate_pool.extend(random.sample(fallback_balls_list, remaining_needed_fb))
                else: # Fallback pool itself is too small, add all remaining and warn
                    final_red_candidate_pool.extend(fallback_balls_list)
                    logger.warning(f"反向思维后无法补充足够的球到6个，当前 {len(final_red_candidate_pool)}。")

                logger.debug(f"候选池构建: 紧急填充后红球池大小: {len(final_red_candidate_pool)}")

    # r_cand_pool 已经在此函数开头被赋值为 final_red_candidate_pool
    # 所以此处不需要再重新赋值 r_cand_pool = final_red_candidate_pool
    # 再次打印确认，确保r_cand_pool是正确的
    logger.debug(f"候选池构建: 最终用于抽样的红球候选池 (r_cand_pool) 大小: {len(r_cand_pool)}, 内容: {sorted(r_cand_pool)}")
    
    # 蓝球候选池：取分数最高的N个蓝球
    b_cand_pool = [n for n, _ in sorted(b_scores.items(), key=lambda item: item[1], reverse=True)[:top_n_blue]]
    if len(b_cand_pool) < 1: # 确保蓝球候选池至少有一个球
        b_cand_pool = list(BLUE_BALL_RANGE)[:max(1, top_n_blue)] # 如果分数为空或不足，使用默认范围
        if not b_cand_pool: b_cand_pool = [random.choice(list(BLUE_BALL_RANGE))] # 最终保障
    logger.debug(f"候选池构建: 最终用于抽样的蓝球候选池 (b_cand_pool) 大小: {len(b_cand_pool)}, 内容: {sorted(b_cand_pool)}")
    
    # 构建组合池（可能远大于最终推荐数量，用于后续多样性筛选）
    large_pool_size = max(num_combinations_to_generate * 100, 200) # 生成一个较大的组合池
    max_attempts_pool = large_pool_size * 20 # 避免无限循环

    # 根据历史中奖分数段分布调整红球抽样概率
    valid_seg_pcts_for_probs = bool(
        win_seg_pcts_local and
        sum(win_seg_pcts_local.values()) > 1e-6 and
        all(label in win_seg_pcts_local for label in SCORE_SEGMENT_LABELS)
    )
    
    seg_factors_for_probs = {name: 1.0 for name in CANDIDATE_POOL_SEGMENT_NAMES} # 默认均匀分布
    if valid_seg_pcts_for_probs:
        adjusted_props_map = {'High': 0.0, 'Medium': 0.0, 'Low': 0.0}
        
        for seg_label, pct_val in win_seg_pcts_local.items():
            try:
                lower_str, _ = seg_label.split('-')
                lower_bound = int(lower_str)
                
                if lower_bound >= CANDIDATE_POOL_SCORE_THRESHOLDS['High']:
                    adjusted_props_map['High'] = adjusted_props_map.get('High', 0.0) + pct_val
                elif lower_bound >= CANDIDATE_POOL_SCORE_THRESHOLDS['Medium']:
                    adjusted_props_map['Medium'] = adjusted_props_map.get('Medium', 0.0) + pct_val
                else:
                    adjusted_props_map['Low'] = adjusted_props_map.get('Low', 0.0) + pct_val
            except ValueError:
                logger.warning(f"无法解析分数段标签 '{seg_label}'，将均匀分配其历史贡献。")
                for seg_name in CANDIDATE_POOL_SEGMENT_NAMES:
                    adjusted_props_map[seg_name] = adjusted_props_map.get(seg_name, 0.0) + pct_val / len(CANDIDATE_POOL_SEGMENT_NAMES)
        
        total_adjusted_props = sum(adjusted_props_map.values())
        if total_adjusted_props > 1e-9:
            for seg_name in CANDIDATE_POOL_SEGMENT_NAMES:
                seg_factors_for_probs[seg_name] = (adjusted_props_map.get(seg_name, 0.0) / total_adjusted_props) + 0.05 # Add a small base to avoid zero factor
            sum_factors = sum(seg_factors_for_probs.values())
            if sum_factors > 1e-9:
                seg_factors_for_probs = {k: v / sum_factors for k, v in seg_factors_for_probs.items()}
            logger.debug(f"概率抽样: 使用的历史分数段调整因子 (seg_factors_for_probs): {seg_factors_for_probs}")
        else:
            logger.debug("概率抽样: 历史分数段调整后总比例过小，使用均匀概率因子。")
            seg_factors_for_probs = {lbl: 1.0/len(CANDIDATE_POOL_SEGMENT_NAMES) for lbl in CANDIDATE_POOL_SEGMENT_NAMES}
    else:
        logger.debug(f"概率抽样: 未使用历史分数段调整因子 (winning_segment_percentages 无效或不完整)。")

    # 计算红球的原始抽样概率（分数 * 历史分数段因子）
    r_probs_raw = {}
    r_cand_pool_for_probs = [] # 实际用于概率抽样的红球池
    if not r_cand_pool: # 极端情况：红球候选池为空
        logger.error("严重: r_cand_pool在概率计算前为空。将使用默认范围进行回退抽样。")
        # Fallback to a minimum viable pool
        r_cand_pool = random.sample(list(RED_BALL_RANGE), k=min(target_red_pool_size if target_red_pool_size >= 6 else 6, len(RED_BALL_RANGE)))
    
    for n_prob_calc in r_cand_pool:
        # 找到球对应的分数段，并获取其调整因子
        score_for_ball = r_scores.get(n_prob_calc, 0)
        segment_name_for_ball = "Low"
        if score_for_ball > CANDIDATE_POOL_SCORE_THRESHOLDS['High']:
            segment_name_for_ball = "High"
        elif score_for_ball > CANDIDATE_POOL_SCORE_THRESHOLDS['Medium']:
            segment_name_for_ball = "Medium"
        
        default_factor = 1.0 / len(CANDIDATE_POOL_SEGMENT_NAMES) if len(CANDIDATE_POOL_SEGMENT_NAMES) > 0 else 1.0
        segment_adj_factor = seg_factors_for_probs.get(segment_name_for_ball, default_factor)

        # 原始概率 = (分数 + 1) * 分数段调整因子 (加1避免0分球完全没有概率)
        raw_prob_val = (score_for_ball + 1.0) * segment_adj_factor
        if raw_prob_val > 0: # 只保留正概率的球
            r_probs_raw[n_prob_calc] = raw_prob_val
            r_cand_pool_for_probs.append(n_prob_calc)
    
    # 归一化红球抽样概率数组
    r_probs_arr = np.array([])
    if not r_cand_pool_for_probs:
        logger.debug("没有红球具有正的原始概率。将从r_cand_pool或范围中均匀抽样。")
        # 回退到均匀抽样，确保池中有足够数量的球
        r_cand_pool_for_probs = list(r_cand_pool) if len(r_cand_pool) >= 6 else list(RED_BALL_RANGE)
        if len(r_cand_pool_for_probs) < 6: # Final safeguard if not enough balls
            r_cand_pool_for_probs = list(RED_BALL_RANGE) # Use all red balls
        if r_cand_pool_for_probs:
            r_probs_arr = np.ones(len(r_cand_pool_for_probs)) / len(r_cand_pool_for_probs)
    else:
        r_probs_arr = np.array([r_probs_raw.get(n, 0) for n in r_cand_pool_for_probs])
        tot_r_prob_raw = np.sum(r_probs_arr)
        if tot_r_prob_raw > 1e-9:
            r_probs_arr = r_probs_arr / tot_r_prob_raw # 归一化
            # 确保概率和为1，防止浮点误差导致random.choice报错
            if len(r_probs_arr) > 1 : r_probs_arr[-1]=max(0,1.0-np.sum(r_probs_arr[:-1]))
            elif len(r_probs_arr) == 1: r_probs_arr[0] = 1.0
            else: # If r_probs_arr is empty after this logic, fallback
                logger.debug("红球概率数组异常为空，使用均匀概率。")
                r_probs_arr = np.ones(len(r_cand_pool_for_probs)) / len(r_cand_pool_for_probs) if r_cand_pool_for_probs else np.array([])
        else:
            logger.debug("红球原始概率总和过小，使用均匀概率。")
            if r_cand_pool_for_probs:
                r_probs_arr = np.ones(len(r_cand_pool_for_probs)) / len(r_cand_pool_for_probs)
            else: # Fallback if cand pool is also empty
                logger.debug("红球候选池在概率计算后为空，使用全范围随机。")
                r_cand_pool_for_probs = list(RED_BALL_RANGE)
                r_probs_arr = np.ones(len(r_cand_pool_for_probs)) / len(r_cand_pool_for_probs)


    # 蓝球抽样概率
    b_weights_arr = np.array([b_scores.get(n, 0) + 1.0 for n in b_cand_pool]) # 蓝球权重 (加1避免0权重)
    b_probs_arr = np.zeros(len(b_cand_pool))
    if not b_cand_pool:
        logger.warning("蓝球候选池为空，将使用随机蓝球。")
        b_cand_pool = random.sample(list(BLUE_BALL_RANGE), k=min(top_n_blue if top_n_blue >=1 else 1, len(BLUE_BALL_RANGE)))
        if not b_cand_pool: b_cand_pool = [random.choice(list(BLUE_BALL_RANGE))] # 最终保障
        b_weights_arr = np.array([b_scores.get(n, 0) + 1.0 for n in b_cand_pool])

    if np.sum(b_weights_arr) > 1e-9 and len(b_cand_pool) > 0:
        b_probs_arr = b_weights_arr / np.sum(b_weights_arr)
        if len(b_probs_arr) > 1: b_probs_arr[-1] = max(0, 1.0 - np.sum(b_probs_arr[:-1]))
        elif len(b_probs_arr) == 1: b_probs_arr[0] = 1.0
        else: # If b_probs_arr is empty after this logic, fallback
            logger.debug("蓝球概率数组异常为空，使用均匀概率。")
            b_probs_arr = np.ones(len(b_cand_pool)) / len(b_cand_pool) if b_cand_pool else np.array([])
    elif len(b_cand_pool) > 0:
        logger.debug("蓝球权重总和过小，使用均匀概率。")
        b_probs_arr = np.ones(len(b_cand_pool)) / len(b_cand_pool)
    else:
        logger.error("严重: 蓝球候选池在概率计算后仍为空。使用默认蓝球1。")
        b_probs_arr = np.array([1.0]); b_cand_pool = [1] # 最终保障，至少选一个蓝球

    sample_size_red = 6
    replace_red_sampling = False # 红球不重复抽样
    use_fallback_sampling_flag = False # 标记是否回退到随机抽样

    # 检查红球概率池和概率数组的有效性
    if len(r_cand_pool_for_probs) < sample_size_red or \
       r_probs_arr.size == 0 or len(r_probs_arr) != len(r_cand_pool_for_probs) or \
       (r_probs_arr.size > 0 and not np.isclose(np.sum(r_probs_arr), 1.0)) or \
       (r_probs_arr.size > 0 and np.any(r_probs_arr < 0)):
        use_fallback_sampling_flag = True
        logger.debug(f"红球概率抽样条件不满足 (池大小:{len(r_cand_pool_for_probs)}, 概率数组len:{r_probs_arr.size}, sum:{np.sum(r_probs_arr) if r_probs_arr.size > 0 else 'N/A'})。将使用回退抽样。")
    
    # 检查蓝球概率池和概率数组的有效性
    if not use_fallback_sampling_flag and (len(b_cand_pool) < 1 or \
       b_probs_arr.size == 0 or len(b_probs_arr) != len(b_cand_pool) or \
       (b_probs_arr.size > 0 and not np.isclose(np.sum(b_probs_arr), 1.0)) or \
       (b_probs_arr.size > 0 and np.any(b_probs_arr < 0))):
        use_fallback_sampling_flag = True
        logger.debug(f"蓝球概率抽样条件不满足 (池大小:{len(b_cand_pool)}, 概率数组len:{b_probs_arr.size}, sum:{np.sum(b_probs_arr) if b_probs_arr.size > 0 else 'N/A'})。将使用回退抽样。")

    if use_fallback_sampling_flag:
        logger.debug("在 generate_combinations 中使用回退（随机）抽样。")

    gen_pool = [] # 存储生成的初始组合（未多样性筛选和最终反向思维）
    attempts = 0
    unique_combo_tracker = set() # 用于追踪已生成的唯一组合

    while len(gen_pool) < large_pool_size and attempts < max_attempts_pool:
        attempts +=1
        r_balls_s_final = []
        b_ball_s_final = None

        try:
            if use_fallback_sampling_flag:
                # 随机抽样，确保池中有足够数量的球
                safe_r_pool_fallback = list(r_cand_pool) if len(r_cand_pool) >= 6 else list(RED_BALL_RANGE)
                if len(safe_r_pool_fallback) < 6:
                    logger.debug(f"回退抽样: 红球池不足6个 ({len(safe_r_pool_fallback)})。跳过此组合尝试。")
                    continue
                r_balls_s_final = sorted(random.sample(safe_r_pool_fallback, 6))
                
                safe_b_pool_fallback = list(b_cand_pool) if len(b_cand_pool) >=1 else list(BLUE_BALL_RANGE)
                if not safe_b_pool_fallback: safe_b_pool_fallback = [random.choice(list(BLUE_BALL_RANGE))] # 最终保障
                b_ball_s_final = random.choice(safe_b_pool_fallback)
            else:
                # 概率抽样
                # random.choice 可能会因为浮点误差导致 sum(pvals) != 1，这里处理一下
                # 确保 r_probs_arr 长度正确且有效
                if not r_cand_pool_for_probs or r_probs_arr.size == 0 or len(r_probs_arr) != len(r_cand_pool_for_probs):
                    raise ValueError("红球概率抽样池或概率数组无效。")
                
                # 从红球候选池中按概率抽样6个球
                r_balls_sampled_temp = np.random.choice(r_cand_pool_for_probs, size=sample_size_red, replace=replace_red_sampling, p=r_probs_arr).tolist()
                
                # 如果抽样结果不足6个（不应发生，除非池本身不足6个且replace=False），则尝试补充
                if len(r_balls_sampled_temp) < 6:
                    remaining_needed_sample = 6 - len(r_balls_sampled_temp)
                    fill_pool_candidates_sample = [b for b in r_cand_pool_for_probs if b not in r_balls_sampled_temp] # 从完整候选池中选未被抽样的球
                    
                    if len(fill_pool_candidates_sample) < remaining_needed_sample:
                        # 如果候选池都不够了，从整个红球范围补充
                        fill_pool_candidates_sample.extend([b for b in RED_BALL_RANGE if b not in r_balls_sampled_temp and b not in fill_pool_candidates_sample])

                    if len(fill_pool_candidates_sample) >= remaining_needed_sample:
                         r_balls_sampled_temp.extend(random.sample(fill_pool_candidates_sample, remaining_needed_sample))
                    else:
                         logger.debug(f"无法补充红球到6个，当前 {len(r_balls_sampled_temp)}。跳过此组合尝试。")
                         continue # 跳过此组合尝试, because red ball quantity is not met

                r_balls_s_final = sorted(list(set(r_balls_sampled_temp))) # 去重并排序
                if len(r_balls_s_final) != 6:
                    logger.debug(f"红球数量不匹配 ({len(r_balls_s_final)})，跳过组合: {r_balls_s_final}")
                    continue # 确保最终红球数量为6

                # 蓝球概率抽样
                if not b_cand_pool or b_probs_arr.size == 0 or len(b_probs_arr) != len(b_cand_pool):
                    raise ValueError("蓝球概率抽样池或概率数组无效。")
                b_ball_s_final = np.random.choice(b_cand_pool, size=1, p=b_probs_arr).tolist()[0]

            combo_tuple_for_tracking = (tuple(r_balls_s_final), b_ball_s_final)
            if combo_tuple_for_tracking not in unique_combo_tracker:
                gen_pool.append({'red': r_balls_s_final, 'blue': b_ball_s_final})
                unique_combo_tracker.add(combo_tuple_for_tracking)

        except ValueError as e_val_sampling:
            # 捕获因概率数组问题导致的ValueError，并切换到回退抽样
            use_fallback_sampling_flag = True
            if attempts <= 5: # 首次几次尝试才打印警告，避免过多日志
                logger.warning(f"generate_combinations中的概率抽样失败 ({e_val_sampling})，此次运行永久切换到回退方案。检查概率数组和候选池。")
            continue # 继续循环，但下次会使用回退抽样
        except Exception as e_gen_combo:
            logger.debug(f"组合生成尝试期间发生异常: {e_gen_combo}")
            continue
            
    if not gen_pool:
        logger.warning("无法生成任何初始组合。")
        return [], ["推荐组合:", "  无法生成推荐组合 (初始池为空)。"]
    
    # --- 组合评分 ---
    scored_combos = []
    patt_data_local = pattern_analysis_data
    hist_odd_cnt_local = patt_data_local.get('most_common_odd_even_count')
    hist_zone_dist_local = patt_data_local.get('most_common_zone_distribution')
    
    # 蓝球大小和奇偶的常见模式
    blue_l_counts_local = patt_data_local.get('blue_large_counts',{})
    hist_blue_large_local = blue_l_counts_local.get(True,0) > blue_l_counts_local.get(False,0) if blue_l_counts_local else None
    
    blue_o_counts_local = patt_data_local.get('blue_odd_counts',{})
    hist_blue_odd_local = blue_o_counts_local.get(True,0) > blue_o_counts_local.get(False,0) if blue_o_counts_local else None

    arm_rules_processed_local = pd.DataFrame()
    if association_rules_df is not None and not association_rules_df.empty:
        arm_rules_processed_local = association_rules_df.copy()
        if not arm_rules_processed_local.empty:
            try:
                # 将antecedents和consequents列从frozenset转换为set，并确保元素是整数
                if 'antecedents' in arm_rules_processed_local.columns and 'consequents' in arm_rules_processed_local.columns:
                    arm_rules_processed_local['antecedents_set'] = arm_rules_processed_local['antecedents'].apply(
                        lambda x: set(map(int, x)) if isinstance(x, frozenset) else set()
                    )
                    arm_rules_processed_local['consequents_set'] = arm_rules_processed_local['consequents'].apply(
                        lambda x: set(map(int, x)) if isinstance(x, frozenset) else set()
                    )
                else:
                    logger.warning("ARM规则DataFrame缺少 'antecedents' 或 'consequents' 列。ARM奖励将无法正确应用。")
                    arm_rules_processed_local = pd.DataFrame()
            except (TypeError, ValueError) as e_arm_conv_local:
                logger.warning(f"转换ARM规则项为整数集合时出错 ({e_arm_conv_local})。ARM奖励可能无法正确应用。")
                arm_rules_processed_local = pd.DataFrame()
    
    for combo_item_score in gen_pool:
        r_list_score, b_val_score = combo_item_score['red'], combo_item_score['blue']
        
        # 基础分数：所有红球分数之和 + 蓝球分数
        base_s_score = sum(r_scores.get(ball_num_score,0) for ball_num_score in r_list_score) + b_scores.get(b_val_score,0)
        bonus_s_score = 0
        
        # 组合属性匹配奖励
        if hist_odd_cnt_local is not None and sum(x%2!=0 for x in r_list_score)==hist_odd_cnt_local:
            bonus_s_score += weights_config['COMBINATION_ODD_COUNT_MATCH_BONUS']
        
        if hist_blue_odd_local is not None and (b_val_score%2!=0)==hist_blue_odd_local:
            bonus_s_score += weights_config['COMBINATION_BLUE_ODD_MATCH_BONUS']
        
        combo_props_current_score = get_combo_properties(r_list_score)
        if hist_zone_dist_local and combo_props_current_score['zone_dist'] == hist_zone_dist_local:
            bonus_s_score += weights_config['COMBINATION_ZONE_MATCH_BONUS']

        if hist_blue_large_local is not None and (b_val_score > 8)==hist_blue_large_local:
            bonus_s_score += weights_config['COMBINATION_BLUE_SIZE_MATCH_BONUS']

        # 关联规则命中奖励
        arm_specific_bonus_score = 0
        combo_red_set_score = set(r_list_score)
        if not arm_rules_processed_local.empty and 'antecedents_set' in arm_rules_processed_local.columns and 'consequents_set' in arm_rules_processed_local.columns:
            for _, rule_iter in arm_rules_processed_local.iterrows():
                # 检查规则的前项和后项是否都包含在当前红球组合中
                if isinstance(rule_iter.get('antecedents_set'), set) and isinstance(rule_iter.get('consequents_set'), set):
                    if rule_iter['antecedents_set'].issubset(combo_red_set_score) and rule_iter['consequents_set'].issubset(combo_red_set_score):
                        # 奖励基于规则的lift和confidence
                        lift_bonus_val = (rule_iter.get('lift', 1.0) - 1.0) * weights_config.get('ARM_BONUS_LIFT_FACTOR', 0.2)
                        conf_bonus_val = rule_iter.get('confidence', 0.0) * weights_config.get('ARM_BONUS_CONF_FACTOR', 0.1)
                        current_rule_bonus_val = (lift_bonus_val + conf_bonus_val) * weights_config['ARM_COMBINATION_BONUS_WEIGHT']
                        arm_specific_bonus_score += current_rule_bonus_val
            # 对ARM奖励进行上限限制，避免某个规则过于主导
            arm_specific_bonus_score = min(arm_specific_bonus_score, weights_config['ARM_COMBINATION_BONUS_WEIGHT'] * 2.0)

        bonus_s_score += arm_specific_bonus_score
        
        scored_combos.append({
            'combination': combo_item_score,
            'score': base_s_score + bonus_s_score,
            'red_tuple': tuple(sorted(r_list_score)), # 存储为元组以便hashable和排序
            'properties': combo_props_current_score
        })
    
    if not scored_combos:
        logger.warning("评分后组合列表为空。")
        return [], ["推荐组合:", "  无法生成推荐组合 (评分后为空)。"]

    # --- 多样性选择 ---
    # 按分数排序所有评分过的组合，这是多样性选择和后续补充的基础
    sorted_scored_combos = sorted(scored_combos, key=lambda x: x['score'], reverse=True)
    
    final_recs_data = [] # 存储经过多样性选择的最终推荐组合
    if sorted_scored_combos:
        final_recs_data.append(sorted_scored_combos[0]) # 总是选择最高分的一个组合作为种子

    attempts_for_diversity_select = 0
    # 遍历其余的评分组合，尝试加入到最终推荐中，同时满足多样性要求
    # 从 sorted_scored_combos 的第二个元素开始
    candidate_pool_for_diversity = sorted_scored_combos[1:]
    
    for candidate_combo_dict_select in candidate_pool_for_diversity:
        if len(final_recs_data) >= num_combinations_to_generate: break # 已达到目标数量
        if attempts_for_diversity_select > diversity_max_attempts * num_combinations_to_generate :
            logger.debug(f"多样性选择达到最大尝试次数 ({attempts_for_diversity_select})，当前已选 {len(final_recs_data)} 组合。")
            break
            
        candidate_red_set_select = set(candidate_combo_dict_select['red_tuple'])
        candidate_props_select = candidate_combo_dict_select['properties']
        is_diverse_enough_select = True
        
        for existing_rec_dict_select in final_recs_data:
            existing_red_set_select = set(existing_rec_dict_select['red_tuple'])
            existing_props_select = existing_rec_dict_select['properties']
            
            common_reds_count = len(candidate_red_set_select.intersection(existing_red_set_select))
            if common_reds_count > max_common_reds_allowed: # 共同红球数超过允许值
                is_diverse_enough_select = False; break
            
            if abs(candidate_props_select['sum'] - existing_props_select['sum']) < diversity_sum_diff_thresh: # 和值差异不足
                is_diverse_enough_select = False; break
            
            if abs(candidate_props_select['odd_count'] - existing_props_select['odd_count']) < diversity_oddeven_diff_min: # 奇数个数差异不足
                is_diverse_enough_select = False; break
            
            # 区间分布差异：计算不同区间计数的位置数量
            diff_zones_count_select = sum(1 for i in range(len(candidate_props_select['zone_dist'])) if candidate_props_select['zone_dist'][i] != existing_props_select['zone_dist'][i])
            if diff_zones_count_select < diversity_zone_dist_min_diff_zones: # 区间分布差异不够
                is_diverse_enough_select = False; break
        
        if is_diverse_enough_select:
            final_recs_data.append(candidate_combo_dict_select)
        attempts_for_diversity_select +=1

    # 如果多样性选择后组合数不足，从 sorted_scored_combos (所有评分组合的排序列表) 中补充
    if len(final_recs_data) < num_combinations_to_generate:
        logger.debug(f"多样性选择后组合数不足 ({len(final_recs_data)})，将从剩余高分组合中补充。")
        
        # 创建当前已选组合的唯一标识集合 (红球元组 + 蓝球)
        current_selected_combo_tuples_for_fill = set()
        for rec_fill_check in final_recs_data:
            if 'combination' in rec_fill_check and 'red' in rec_fill_check['combination'] and 'blue' in rec_fill_check['combination']:
                 current_selected_combo_tuples_for_fill.add((tuple(sorted(rec_fill_check['combination']['red'])), rec_fill_check['combination']['blue']))

        needed_more_fill_after_diversity = num_combinations_to_generate - len(final_recs_data)
        added_count_after_diversity = 0
        
        for combo_dict_fill_ad in sorted_scored_combos: # 遍历所有评分过的组合 (已按分排序)
            if added_count_after_diversity >= needed_more_fill_after_diversity: break
            
            if 'combination' in combo_dict_fill_ad and 'red' in combo_dict_fill_ad['combination'] and 'blue' in combo_dict_fill_ad['combination']:
                combo_tuple_to_check_fill = (tuple(sorted(combo_dict_fill_ad['combination']['red'])), combo_dict_fill_ad['combination']['blue'])
                if combo_tuple_to_check_fill not in current_selected_combo_tuples_for_fill:
                     final_recs_data.append(combo_dict_fill_ad)
                     current_selected_combo_tuples_for_fill.add(combo_tuple_to_check_fill)
                     added_count_after_diversity += 1
    
    # 再次确保最终数量和排序 (补充后可能数量超出，或顺序被打乱)
    final_recs_data = sorted(final_recs_data, key=lambda x: x['score'], reverse=True)[:num_combinations_to_generate]


    # --- 最终推荐组合层面的反向思维 ---
    enable_final_reverse = weights_config.get('FINAL_COMBO_REVERSE_ENABLED', False)
    remove_top_percent_final = weights_config.get('FINAL_COMBO_REVERSE_REMOVE_TOP_PERCENT', 0.0)
    refill_after_reverse = weights_config.get('FINAL_COMBO_REVERSE_REFILL', False)
    
    applied_final_reverse_message_suffix = ""

    num_to_remove_final_actual = 0
    if enable_final_reverse and final_recs_data and 0.0 < remove_top_percent_final <= 1.0:
        num_to_remove_final_actual = int(round(len(final_recs_data) * remove_top_percent_final))
        # 确保至少移除一个（如果百分比计算结果小于1但大于0），或者不超过总数减一（至少保留一个）
        if num_to_remove_final_actual == 0 and remove_top_percent_final > 0 and len(final_recs_data) > 1:
            num_to_remove_final_actual = 1
        # 如果计算出的移除数量大于或等于当前总数，则最多移除 (总数 - 1)
        if num_to_remove_final_actual >= len(final_recs_data) and len(final_recs_data) > 0:
            num_to_remove_final_actual = len(final_recs_data) - 1
        if num_to_remove_final_actual < 0: # 不应该发生，但作为保护
             num_to_remove_final_actual = 0

    if enable_final_reverse and num_to_remove_final_actual > 0 and final_recs_data and len(final_recs_data) > num_to_remove_final_actual :
        logger.debug(f"最终组合反向思维: 从 {len(final_recs_data)} 注当前最佳推荐中，移除得分最高的 {num_to_remove_final_actual} 注 (基于 {remove_top_percent_final*100:.1f}% 的配置)。")
        
        removed_combos_info_final = final_recs_data[:num_to_remove_final_actual]
        final_recs_data = final_recs_data[num_to_remove_final_actual:] # 移除最高分的组合
        
        logger.debug(f"  被移除的组合分数: {[round(rc['score'],2) for rc in removed_combos_info_final]}")
        logger.debug(f"  移除后剩余组合数: {len(final_recs_data)}")
        applied_final_reverse_message_suffix = f" (应用最终反向思维移除前{remove_top_percent_final*100:.0f}%"

        if refill_after_reverse:
            # 需要补充到原始目标数量
            needed_to_refill_fr = num_combinations_to_generate - len(final_recs_data)
            
            # 构建备选池：从所有评分过的组合中排除当前已选和刚被移除的组合
            current_final_plus_removed_tuples = set()
            for rec_fr_check in final_recs_data + removed_combos_info_final:
                 if 'combination' in rec_fr_check and 'red' in rec_fr_check['combination'] and 'blue' in rec_fr_check['combination']:
                    current_final_plus_removed_tuples.add((tuple(sorted(rec_fr_check['combination']['red'])), rec_fr_check['combination']['blue']))
            
            alternative_combos_for_fr_refill = []
            for combo_dict_fr_alt in sorted_scored_combos: # 遍历所有评分过的组合
                if 'combination' in combo_dict_fr_alt and 'red' in combo_dict_fr_alt['combination'] and 'blue' in combo_dict_fr_alt['combination']:
                    combo_tuple_fr_alt = (tuple(sorted(combo_dict_fr_alt['combination']['red'])), combo_dict_fr_alt['combination']['blue'])
                    if combo_tuple_fr_alt not in current_final_plus_removed_tuples:
                        alternative_combos_for_fr_refill.append(combo_dict_fr_alt)
            
            if needed_to_refill_fr > 0 and alternative_combos_for_fr_refill:
                logger.debug(f"  尝试从备选池补充 {needed_to_refill_fr} 注组合。备选池大小: {len(alternative_combos_for_fr_refill)}")
                
                refilled_count_fr = 0
                current_tuples_in_final_recs_data_for_refill = set() # 跟踪当前final_recs_data中的组合
                if final_recs_data:
                     for rec_fr_curr in final_recs_data:
                        if 'combination' in rec_fr_curr and 'red' in rec_fr_curr['combination'] and 'blue' in rec_fr_curr['combination']:
                            current_tuples_in_final_recs_data_for_refill.add((tuple(sorted(rec_fr_curr['combination']['red'])), rec_fr_curr['combination']['blue']))
                
                for alt_combo_fr in alternative_combos_for_fr_refill:
                    if refilled_count_fr >= needed_to_refill_fr: break
                    if 'combination' in alt_combo_fr and 'red' in alt_combo_fr['combination'] and 'blue' in alt_combo_fr['combination']:
                        alt_combo_tuple_fr = (tuple(sorted(alt_combo_fr['combination']['red'])), alt_combo_fr['combination']['blue'])
                        if alt_combo_tuple_fr not in current_tuples_in_final_recs_data_for_refill: # 确保不重复补充
                            final_recs_data.append(alt_combo_fr)
                            current_tuples_in_final_recs_data_for_refill.add(alt_combo_tuple_fr)
                            refilled_count_fr += 1
                
                final_recs_data = sorted(final_recs_data, key=lambda x: x['score'], reverse=True)
                final_recs_data = final_recs_data[:num_combinations_to_generate] # 确保最终数量不超过原始目标
                logger.debug(f"  补充后组合数: {len(final_recs_data)}")
                applied_final_reverse_message_suffix += "并补充"
            elif needed_to_refill_fr > 0:
                 logger.debug(f"  不进行补充 (需要补充 {needed_to_refill_fr} 注，但备选池为空)。")
        else:
            logger.debug("  配置为不进行补充。")
        
        applied_final_reverse_message_suffix += ")"
    elif enable_final_reverse and remove_top_percent_final > 0.0 :
         logger.debug(f"最终组合反向思维: 未执行，可能因为当前推荐组合数 ({len(final_recs_data) if final_recs_data else 0}) 不足以按百分比 ({remove_top_percent_final*100:.1f}%) 有效移除，或计算移除数为0。")

    # --- 生成最终的输出字符串 ---
    output_title = f"推荐组合 (Top {len(final_recs_data)}{applied_final_reverse_message_suffix}):"
    output_strs_final = []
    for i, combo_dict in enumerate(final_recs_data):
        red_str = ' '.join(f'{n:02d}' for n in combo_dict['combination']['red'])
        blue_str = f'{combo_dict["combination"]["blue"]:02d}'
        score = combo_dict['score']
        output_strs_final.append(f"  注 {i+1}: 红球 [{red_str}] 蓝球 [{blue_str}] (得分: {score:.2f})")
    
    if not final_recs_data:
        output_strs_final.append("  无法生成推荐组合。")

    return final_recs_data, [output_title] + output_strs_final

# --- INSERTED analyze_and_recommend FUNCTION DEFINITION HERE ---
def analyze_and_recommend(df_hist: pd.DataFrame,
                          ml_lags_list_param: List[int],
                          weights_conf_param: Dict,
                          arm_rules_hist: pd.DataFrame, # ARM rules based on df_hist or broader history
                          train_ml: bool = True
                         ) -> Tuple[List[Dict], List[str], Dict, Optional[Dict], Dict, Dict[str, float]]:
    """
    Core analysis and recommendation pipeline.
    Takes historical data up to a point, applies all analysis steps,
    and generates recommendations for the *next* draw.

    Args:
        df_hist (pd.DataFrame): Historical data for analysis and model training.
        ml_lags_list_param (List[int]): List of lags for ML features.
        weights_conf_param (Dict): Current weights configuration.
        arm_rules_hist (pd.DataFrame): Pre-calculated association rules based on relevant history.
                                       This is crucial for backtesting to use rules known at that point in time.
        train_ml (bool): Whether to train ML models. If False, ML predictions will be skipped.

    Returns:
        Tuple containing:
        - final_recs_list (List[Dict]): List of recommended combination dicts (e.g., [{'combination': {'red': [...], 'blue': ...}, 'score': ...}]).
        - final_rec_strs_list (List[str]): List of formatted recommendation strings.
        - analysis_results_summary (Dict): Summary of various analyses (freq, patterns).
        - ml_models (Optional[Dict]): Trained ML models, if any.
        - scores_data (Dict): Calculated scores for red and blue balls (e.g., {'red_scores': {...}, 'blue_scores': {...}}).
        - winning_segment_percentages (Dict[str, float]): Historical winning red ball segment percentages
                                                          based on the *current* df_hist and scores_data.
    """
    logger.debug(f"analyze_and_recommend: Called with {len(df_hist)} historical periods. Train ML: {train_ml}")

    # 1. Basic Analyses (Frequency, Omission, Patterns)
    # These use df_hist directly.
    # Pass copies to avoid unintentional modifications of the input DataFrame
    freq_om_data = analyze_frequency_omission(df_hist.copy(), weights_conf_param)
    patt_data = analyze_patterns(df_hist.copy(), weights_conf_param)

    # Note: arm_rules_hist is PASSED IN.
    # For backtesting, this represents rules known *up to that point in history*.
    # For a live recommendation, arm_rules_hist would typically be based on all available history.

    # 2. ML Model Training and Prediction (if enabled)
    ml_models = None
    pred_probs = {'red': {}, 'blue': {}} # Default empty probabilities

    if train_ml:
        # train_prediction_models uses df_hist to train models for predicting the period *after* df_hist ends.
        ml_models = train_prediction_models(df_hist.copy(), ml_lags_list_param, weights_conf_param)

    if ml_models:
        # predict_next_draw_probabilities uses the tail of df_hist to create lagged features
        # for predicting the period *after* df_hist ends.
        pred_probs = predict_next_draw_probabilities(df_hist.copy(), ml_models, ml_lags_list_param, weights_conf_param)
    else:
        logger.debug("analyze_and_recommend: ML model training skipped or failed, using zero probabilities for scoring.")
        # Ensure pred_probs structure is present with zero probabilities if ML is skipped or fails
        pred_probs['red'] = {r: 0.0 for r in RED_BALL_RANGE}
        pred_probs['blue'] = {b: 0.0 for b in BLUE_BALL_RANGE}


    # 3. Calculate Ball Scores
    # Scores are calculated for the *next* draw, based on analyses of df_hist and ML predictions.
    scores_data = calculate_scores(freq_om_data, patt_data, pred_probs, weights_conf_param)

    # 4. Analyze historical winning red ball score segments
    # This uses the scores_data (for the *next* draw) to see how *past* winning balls in df_hist
    # would have been scored by the *current* scoring logic. This result is input to generate_combinations.
    winning_seg_counts = {label: 0 for label in SCORE_SEGMENT_LABELS}
    winning_seg_pcts = {label: 0.0 for label in SCORE_SEGMENT_LABELS}

    if scores_data.get('red_scores') and not df_hist.empty:
         # Ensure red_scores are passed, not the whole scores_data
         winning_seg_counts, winning_seg_pcts = analyze_winning_red_ball_score_segments(
             df_hist.copy(), # Analyze winning balls from the historical set
             scores_data['red_scores'], # Using the scores just calculated for the *next* potential draw
             SCORE_SEGMENT_BOUNDARIES,
             SCORE_SEGMENT_LABELS
         )
    else:
        logger.debug("analyze_and_recommend: Skipping winning segment analysis (no red scores or empty df_hist).")


    # 5. Generate Combinations for the Next Draw
    # Uses scores_data (for next draw), patt_data (from df_hist),
    # arm_rules_hist (pre-calculated for the relevant history),
    # winning_seg_pcts (calculated above based on current scores and historical wins), and weights.
    final_recs_list, final_rec_strs_list = generate_combinations(
        scores_data,
        patt_data,
        arm_rules_hist, # Use the passed-in ARM rules
        winning_seg_pcts,
        weights_conf_param
    )

    # 6. Consolidate analysis results summary
    analysis_results_summary = {
        'frequency_omission': freq_om_data,
        'patterns': patt_data,
        # Optionally include ARM rules if needed for external review, but they are large
        # 'associations_used_for_recommendation': arm_rules_hist.to_dict('records') if arm_rules_hist is not None and not arm_rules_hist.empty else [],
    }

    logger.debug(f"analyze_and_recommend: Generated {len(final_recs_list)} recommendations.")

    return final_recs_list, final_rec_strs_list, analysis_results_summary, ml_models, scores_data, winning_seg_pcts
# --- END OF INSERTED FUNCTION ---

def get_prize_level(red_hits: int, blue_hit: bool) -> Optional[str]:
    """
    根据红球命中数和蓝球命中情况确定奖级。
    Args:
        red_hits (int): 红球命中数。
        blue_hit (bool): 蓝球是否命中。
    Returns:
        Optional[str]: 奖级名称（例如 "一等奖"），如果未中奖则为None。
    """
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

def backtest(df: pd.DataFrame, ml_lags_list: List[int], weights_config: Dict,
             arm_rules_for_backtest: pd.DataFrame,
             backtest_periods_to_eval: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    执行回测以评估策略性能。
    回测采用滚动窗口方式：每次使用当前期之前的所有数据进行分析和模型训练，然后预测下一期，并与实际开奖结果对比。
    Args:
        df (pd.DataFrame): 完整的历史数据DataFrame。
        ml_lags_list (List[int]): 机器学习模型使用的滞后特征阶数。
        weights_config (Dict): 当前使用的权重配置。
        arm_rules_for_backtest (pd.DataFrame): 完整的历史关联规则，用于回测中的组合生成。
        backtest_periods_to_eval (int): 要回测的期数。
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: (回测结果DataFrame, 扩展统计信息字典)。
    """
    # 确定回测所需的最少历史数据量
    max_hist_lag = max(ml_lags_list) if ml_lags_list else 0
    # 训练ML模型所需的最小期数 + 1 (预测目标期)
    min_initial_train_periods = max_hist_lag + 1 + MIN_POSITIVE_SAMPLES_FOR_ML
    
    if len(df) < min_initial_train_periods + 1: # +1 for the first prediction target itself
        logger.warning(f"回测: 数据不足({len(df)})，需要至少 {min_initial_train_periods + 1} 期 (训练所需 + 1预测期)。")
        return pd.DataFrame(), {}

    # 确定回测循环的起始和结束索引
    # 预测目标期从 min_initial_train_periods 索引开始
    first_prediction_target_idx = min_initial_train_periods
    last_prediction_target_idx = len(df) - 1

    if first_prediction_target_idx > last_prediction_target_idx:
        logger.warning("回测: 无足够后续数据进行预测评估循环。")
        return pd.DataFrame(), {}

    # 实际回测的起始索引：确保从足够早的期数开始，以满足 backtest_periods_to_eval 的要求
    # 理论上应该从 df.index[first_prediction_target_idx] 开始，到 df.index[last_prediction_target_idx] 结束
    # 且确保回测期数等于 backtest_periods_to_eval
    
    # Calculate the actual start index for the loop to run `backtest_periods_to_eval` iterations
    # The first target period's index is `min_initial_train_periods`
    # So if `backtest_periods_to_eval` is 100, and last index is `N-1`, we want to start at `N-1 - 100 + 1`
    # But also ensure it's not before `min_initial_train_periods`
    loop_start_idx = max(first_prediction_target_idx, (len(df) - backtest_periods_to_eval))
    total_periods_to_loop = (len(df) - 1) - loop_start_idx + 1 # Actual number of periods to test

    results_list = [] # 存储每期每个组合的回测结果
    red_cols_list = [f'red{i+1}' for i in range(6)]
    
    # 标记是否为Optuna运行（Optuna运行时需要抑制日志）
    is_opt_run_flag = backtest_periods_to_eval == OPTIMIZATION_BACKTEST_PERIODS
    
    prize_counts = Counter() # 各奖级命中次数
    best_hit_per_period = [] # 每期最佳命中情况
    periods_with_any_blue_hit = set() # 至少命中蓝球的期数
    num_combinations_generated_per_run = weights_config.get('NUM_COMBINATIONS_TO_GENERATE', 10)

    backtest_start_time = time.perf_counter()

    winning_red_ball_ranks_percentiles = [] # 中奖红球在分数排名中的百分位
    winning_blue_ball_ranks_percentiles = [] # 中奖蓝球在分数排名中的百分位

    for df_idx_for_prediction_target in range(loop_start_idx, last_prediction_target_idx + 1):
        current_loop_iteration = df_idx_for_prediction_target - loop_start_idx + 1

        # 非Optuna运行时，打印回测进度
        if not is_opt_run_flag and (current_loop_iteration == 1 or current_loop_iteration % 10 == 0 or current_loop_iteration == total_periods_to_loop):
            original_console_level_bt_progress = global_console_handler.level
            original_console_formatter_bt_progress = global_console_handler.formatter
            set_console_verbosity(logging.INFO, use_simple_formatter=True) # 简洁模式显示进度
            logger.info(f"  回测进度: {current_loop_iteration} / {total_periods_to_loop}")
            
            elapsed_time = time.perf_counter() - backtest_start_time
            if current_loop_iteration > 0:
                avg_time_per_period = elapsed_time / current_loop_iteration
                remaining_periods = total_periods_to_loop - current_loop_iteration
                
                set_console_verbosity(logging.INFO, use_simple_formatter=False) # 详细模式显示时间预估
                logger.info(f"    回测单期平均耗时: {avg_time_per_period:.4f} 秒。")
                if remaining_periods > 0:
                    estimated_remaining_time = avg_time_per_period * remaining_periods
                    hours, rem = divmod(estimated_remaining_time, 3600)
                    minutes, seconds = divmod(rem, 60)
                    logger.info(f"    预估剩余完成时间: {int(hours):02d}小时 {int(minutes):02d}分钟 {int(seconds):02d}秒。")
                else:
                    logger.info(f"    回测已完成 {current_loop_iteration}/{total_periods_to_loop} 期。")
            
            global_console_handler.setLevel(original_console_level_bt_progress)
            global_console_handler.setFormatter(original_console_formatter_bt_progress)

        # 准备当前训练数据 (当前期之前的全部历史数据)
        current_train_data = df.iloc[:df_idx_for_prediction_target].copy()
        if len(current_train_data) < min_initial_train_periods:
            logger.debug(f"跳过周期 {df.loc[df_idx_for_prediction_target, '期号']}，因训练数据不足 ({len(current_train_data)})。")
            best_hit_per_period.append({ # Add a placeholder for skipped periods
                'period': df.loc[df_idx_for_prediction_target, '期号'],
                'max_red_hits': -1, # Indicates data issue
                'blue_hit_in_period': False,
                'error': f"Not enough training data ({len(current_train_data)} < {min_initial_train_periods})"
            })
            continue

        # 获取当前预测目标期的实际开奖结果
        actual_outcome_row = df.loc[df_idx_for_prediction_target]
        current_period_actual_id = actual_outcome_row['期号']
        try:
            actual_red_list = actual_outcome_row[red_cols_list].astype(int).tolist()
            actual_red_set = set(actual_red_list)
            actual_blue_val = int(actual_outcome_row['blue'])
            # 再次验证实际开奖号码的有效性
            if not (len(actual_red_set) == 6 and all(min(RED_BALL_RANGE)<=r_val<=max(RED_BALL_RANGE) for r_val in actual_red_set) and min(BLUE_BALL_RANGE)<=actual_blue_val<=max(BLUE_BALL_RANGE)):
                raise ValueError("实际开奖号码超出范围或数量不正确")
        except Exception as e_actual:
            logger.warning(f"回测: 获取期号 {current_period_actual_id} 实际结果失败或无效: {e_actual}. 问题行: {actual_outcome_row.to_dict()}")
            best_hit_per_period.append({
                'period': current_period_actual_id, 'max_red_hits': -1, 'blue_hit_in_period': False, 'error': str(e_actual)
            })
            continue

        # Add diagnostic checks for analyze_and_recommend
        if 'analyze_and_recommend' not in sys.modules[__name__].__dict__:
            logger.critical(f"CRITICAL ERROR: 'analyze_and_recommend' is not in global scope for period {current_period_actual_id}.")
            raise RuntimeError("analyze_and_recommend function not found in global scope. Please check module loading.")
        elif not callable(sys.modules[__name__].__dict__['analyze_and_recommend']):
            logger.critical(f"CRITICAL ERROR: 'analyze_and_recommend' exists but is not callable for period {current_period_actual_id}.")
            raise RuntimeError("analyze_and_recommend exists but is not a callable function. Please check module loading.")


        # 在Optuna运行时抑制 analyze_and_recommend 的输出
        original_logger_level_loop = logger.level
        original_console_level_loop = global_console_handler.level
        
        scores_for_current_period = {}
        
        if is_opt_run_flag:
            logger.setLevel(logging.CRITICAL) # Optuna运行时只记录严重错误
            set_console_verbosity(logging.CRITICAL)
            # 使用标准的 with 语句来处理 SuppressOutput
            with SuppressOutput(suppress_stdout=True, capture_stderr=True):
                predicted_combos_list, _, analysis_res_bt, _, scores_for_current_period, _ = analyze_and_recommend(
                    current_train_data, ml_lags_list, weights_config, arm_rules_for_backtest, train_ml=True)
            # with 语句结束时，__exit__ 会自动调用，恢复 sys.stdout/stderr
            logger.setLevel(original_logger_level_loop) # 恢复日志级别
            set_console_verbosity(original_console_level_loop) # 恢复控制台显示级别
        else:
            # 如果不是Optuna运行，则正常调用 analyze_and_recommend
            predicted_combos_list, _, analysis_res_bt, _, scores_for_current_period, _ = analyze_and_recommend(
                current_train_data, ml_lags_list, weights_config, arm_rules_for_backtest, train_ml=True)
        
        # 记录中奖红球和蓝球在当前评分中的排名百分位
        if scores_for_current_period:
            red_scores_dict = scores_for_current_period.get('red_scores')
            if red_scores_dict:
                # 按分数降序排序，并计算每个球的排名
                # Filter out NaN or non-numeric scores before sorting
                valid_red_scores_items = [(k,v) for k,v in red_scores_dict.items() if pd.notna(v) and np.isfinite(v)]
                if valid_red_scores_items:
                    sorted_reds_by_score = sorted(valid_red_scores_items, key=lambda item: item[1], reverse=True)
                    red_rank_map = {ball: rank + 1 for rank, (ball, score) in enumerate(sorted_reds_by_score)}
                    num_total_red_options = len(RED_BALL_RANGE)
                    for winning_red in actual_red_set:
                        rank = red_rank_map.get(winning_red)
                        if rank is not None:
                            percentile_rank = (rank / num_total_red_options) * 100 # 越小越好 (排名越靠前)
                            winning_red_ball_ranks_percentiles.append(percentile_rank)
                        else:
                            logger.debug(f"回测期 {current_period_actual_id}: 中奖红球 {winning_red} 未在评分中找到。")
                else: logger.debug(f"回测期 {current_period_actual_id}: 红球分数字典为空或无效，无法计算排名。")

            blue_scores_dict = scores_for_current_period.get('blue_scores')
            if blue_scores_dict:
                valid_blue_scores_items = [(k,v) for k,v in blue_scores_dict.items() if pd.notna(v) and np.isfinite(v)]
                if valid_blue_scores_items:
                    sorted_blues_by_score = sorted(valid_blue_scores_items, key=lambda item: item[1], reverse=True)
                    blue_rank_map = {ball: rank + 1 for rank, (ball, score) in enumerate(sorted_blues_by_score)}
                    num_total_blue_options = len(BLUE_BALL_RANGE)
                    rank = blue_rank_map.get(actual_blue_val)
                    if rank is not None:
                        percentile_rank = (rank / num_total_blue_options) * 100
                        winning_blue_ball_ranks_percentiles.append(percentile_rank)
                    else:
                        logger.debug(f"回测期 {current_period_actual_id}: 中奖蓝球 {actual_blue_val} 未在评分中找到。")
                else: logger.debug(f"回测期 {current_period_actual_id}: 蓝球分数字典为空或无效，无法计算排名。")

        period_max_red_hits = 0 # 当前期预测组合中的最大红球命中数
        period_blue_hit_achieved_this_draw = False # 当前期是否有任何组合命中蓝球

        if predicted_combos_list:
            for combo_dict_info in predicted_combos_list:
                pred_r_set = set(combo_dict_info['combination']['red'])
                pred_b_val = combo_dict_info['combination']['blue']
                
                red_h = len(pred_r_set.intersection(actual_red_set)) # 红球命中数
                blue_h = (pred_b_val == actual_blue_val) # 蓝球是否命中

                results_list.append({
                    'period': current_period_actual_id,
                    'predicted_red': sorted(list(pred_r_set)), 'predicted_blue': pred_b_val,
                    'actual_red': sorted(actual_red_list), 'actual_blue': actual_blue_val, # Use sorted list for consistency
                    'red_hits': red_h,
                    'blue_hit': blue_h,
                    'combination_score': combo_dict_info.get('score', 0.0)
                })
                
                prize = get_prize_level(red_h, blue_h)
                if prize:
                    prize_counts[prize] += 1
                
                if blue_h:
                    periods_with_any_blue_hit.add(current_period_actual_id)
                    period_blue_hit_achieved_this_draw = True
                if red_h > period_max_red_hits:
                    period_max_red_hits = red_h
            
            best_hit_per_period.append({
                'period': current_period_actual_id,
                'max_red_hits': period_max_red_hits,
                'blue_hit_in_period': period_blue_hit_achieved_this_draw
            })
        else:
            # 如果没有预测出任何组合
            best_hit_per_period.append({
                'period': current_period_actual_id, 'max_red_hits': 0, 'blue_hit_in_period': False, 'error': 'No combos predicted'
            })
            logger.debug(f"回测: 期号 {current_period_actual_id} 未预测任何组合。")


    if not results_list and not winning_red_ball_ranks_percentiles and not winning_blue_ball_ranks_percentiles:
        logger.warning("回测：未产生任何有效结果或分数排名数据。")
        return pd.DataFrame(), {}
    
    results_df_final = pd.DataFrame(results_list) if results_list else pd.DataFrame()
    # 为回测结果DataFrame添加属性，记录回测的期号范围
    if '期号' in df.columns and loop_start_idx < len(df) and last_prediction_target_idx < len(df):
        try:
            results_df_final.attrs['start_period_id'] = df.loc[loop_start_idx, '期号']
            results_df_final.attrs['end_period_id'] = df.loc[last_prediction_target_idx, '期号']
            results_df_final.attrs['num_periods_tested'] = total_periods_to_loop
        except KeyError:
             logger.warning("回测: 由于索引问题，无法设置开始/结束周期属性。")
    
    extended_stats = {
        'prize_counts': dict(prize_counts),
        'best_hit_per_period_df': pd.DataFrame(best_hit_per_period) if best_hit_per_period else pd.DataFrame(),
        'total_combinations_evaluated': len(results_df_final),
        'num_combinations_per_draw_tested': num_combinations_generated_per_run,
        'periods_with_any_blue_hit_count': len(periods_with_any_blue_hit),
        'winning_red_ball_score_percentiles': winning_red_ball_ranks_percentiles,
        'winning_blue_ball_score_percentiles': winning_blue_ball_ranks_percentiles,
    }
    logger.debug(f"回测执行完毕。回测期数: {total_periods_to_loop}。")
    return results_df_final, extended_stats

# 全局变量用于Optuna回调的计时
start_time_optuna_global = 0
estimated_time_logged_global = False

# 修改 optuna_time_estimation_callback 函数签名，接收 total_trials_param
def optuna_time_estimation_callback(study, trial, total_trials_param):
    """
    Optuna回调函数，用于在优化过程中报告进度和预估剩余时间。
    Args:
        study (optuna.study.Study): Optuna研究对象。
        trial (optuna.trial.FrozenTrial): 当前试验对象。
        total_trials_param (int): 传入的总试验次数。
    """
    global start_time_optuna_global, estimated_time_logged_global
    
    if start_time_optuna_global == 0:
        start_time_optuna_global = time.time()
        estimated_time_logged_global = False

    current_trial_num = trial.number + 1
    total_trials = total_trials_param # <-- 使用传入的参数

    # 每隔10%的进度或特定trial次数时打印进度
    if current_trial_num == 1 or current_trial_num % (total_trials // 10 if total_trials // 10 > 0 else 1) == 0 or current_trial_num == total_trials:
        elapsed_time = time.time() - start_time_optuna_global
        avg_time_per_trial = elapsed_time / current_trial_num
        
        remaining_trials = total_trials - current_trial_num
        estimated_remaining_time = avg_time_per_trial * remaining_trials

        hours, rem = divmod(estimated_remaining_time, 3600)
        minutes, seconds = divmod(rem, 60)
        
        # 临时调整控制台日志级别和格式，以保证进度信息输出
        original_console_level_opt = global_console_handler.level
        original_console_formatter_opt = global_console_handler.formatter
        set_console_verbosity(logging.INFO, use_simple_formatter=True)
        
        logger.info(f"Optuna进度: {current_trial_num}/{total_trials} 完成。 平均每试验耗时: {avg_time_per_trial:.2f}s。")
        logger.info(f"  预估剩余时间: {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s。")
        
        # 恢复日志级别和格式
        global_console_handler.setLevel(original_console_level_opt)
        global_console_handler.setFormatter(original_console_formatter_opt)

    # 打印Optuna Trial的详细信息（仅在DEBUG级别下）
    logger.debug(f"Optuna试验 {trial.number} 完成。 Value: {trial.value:.4f}")
    if trial.value is not None and np.isfinite(trial.value) and trial.value != float('inf') and trial.value != float('-inf'):
        # 仅在值有效时才尝试剪枝
        pass # Optuna 会在内部处理剪枝逻辑，此处无需手动实现


def objective(trial: optuna.trial.Trial, df_for_optimization: pd.DataFrame, fixed_ml_lags: List[int]) -> float:
    """
    Optuna的目标函数，用于优化权重。
    Args:
        trial (optuna.trial.Trial): Optuna试验对象，用于建议参数。
        df_for_optimization (pd.DataFrame): 用于优化的历史数据DataFrame。
        fixed_ml_lags (List[int]): 固定的ML滞后特征阶数。
    Returns:
        float: 目标函数值（负的性能分数，Optuna目标是最小化此值）。
    """
    # 建议各种参数的取值范围
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
        
        'REVERSE_THINKING_ITERATIONS': trial.suggest_int('REVERSE_THINKING_ITERATIONS', 0, 3),
        'REVERSE_THINKING_RED_BALLS_TO_REMOVE_PER_ITER': trial.suggest_int('REVERSE_THINKING_RED_BALLS_TO_REMOVE_PER_ITER', 0, 10),

        'OPTUNA_PRIZE_6_WEIGHT': trial.suggest_float('OPTUNA_PRIZE_6_WEIGHT', 0.0, 0.5),
        'OPTUNA_PRIZE_5_WEIGHT': trial.suggest_float('OPTUNA_PRIZE_5_WEIGHT', 0.1, 1.0),
        'OPTUNA_PRIZE_4_WEIGHT': trial.suggest_float('OPTUNA_PRIZE_4_WEIGHT', 0.5, 2.0),
        'OPTUNA_PRIZE_3_WEIGHT': trial.suggest_float('OPTUNA_PRIZE_3_WEIGHT', 1.0, 5.0),
        'OPTUNA_PRIZE_2_WEIGHT': trial.suggest_float('OPTUNA_PRIZE_2_WEIGHT', 2.0, 10.0),
        'OPTUNA_PRIZE_1_WEIGHT': trial.suggest_float('OPTUNA_PRIZE_1_WEIGHT', 5.0, 20.0),
        'OPTUNA_BLUE_HIT_RATE_WEIGHT': trial.suggest_float('OPTUNA_BLUE_HIT_RATE_WEIGHT', 5.0, 20.0),
        'OPTUNA_RED_HITS_WEIGHT': trial.suggest_float('OPTUNA_RED_HITS_WEIGHT', 0.5, 5.0),
    }
    
    # 归一化候选池比例，确保 High + Medium <= 1.0
    prop_h_trial = weights_to_eval['CANDIDATE_POOL_PROPORTIONS_HIGH']
    prop_m_trial = weights_to_eval['CANDIDATE_POOL_PROPORTIONS_MEDIUM']
    if prop_h_trial + prop_m_trial > 1.0 + 1e-9: # 允许一点浮点误差
        total_prop = prop_h_trial + prop_m_trial
        if total_prop > 1e-9 : # 避免除以零
             weights_to_eval['CANDIDATE_POOL_PROPORTIONS_HIGH'] = prop_h_trial / total_prop
             weights_to_eval['CANDIDATE_POOL_PROPORTIONS_MEDIUM'] = prop_m_trial / total_prop
        else: # 极端情况，两者都接近0，则设为默认值
             weights_to_eval['CANDIDATE_POOL_PROPORTIONS_HIGH'] = DEFAULT_WEIGHTS['CANDIDATE_POOL_PROPORTIONS_HIGH']
             weights_to_eval['CANDIDATE_POOL_PROPORTIONS_MEDIUM'] = DEFAULT_WEIGHTS['CANDIDATE_POOL_PROPORTIONS_MEDIUM']

    # 为当前的Optuna试验生成关联规则 (抑制其日志输出)
    with SuppressOutput(suppress_stdout=True, capture_stderr=True):
        arm_rules_for_this_trial = analyze_associations(df_for_optimization.copy(), weights_to_eval)
    logger.debug(f"Optuna试验 {trial.number}: 找到 {len(arm_rules_for_this_trial)} 条ARM规则。")

    # 执行回测
    backtest_results_df, extended_bt_stats = backtest(
        df_for_optimization.copy(), fixed_ml_lags, weights_to_eval,
        arm_rules_for_this_trial,
        OPTIMIZATION_BACKTEST_PERIODS # 使用为Optuna设定的较短回测期数
    )

    if backtest_results_df.empty or len(backtest_results_df) == 0:
        # If backtest produced no results, it's a very bad trial
        logger.debug(f"Optuna试验 {trial.number}: 回测结果为空，返回无穷大。")
        return float('inf')

    # Calculate weighted average red hits (typically higher weight to more hits, e.g., x^1.5)
    # Ensure 'red_hits' column is numeric and filter out 'bad_result' or -1 items
    if 'red_hits' in backtest_results_df.columns:
        valid_red_hits = pd.to_numeric(backtest_results_df['red_hits'], errors='coerce')
        valid_red_hits = valid_red_hits.dropna() # Remove NaNs introduced by coerce
        valid_red_hits = valid_red_hits[valid_red_hits >= 0].astype(float) # Exclude error values, ensure float
        if not valid_red_hits.empty:
            avg_weighted_red_hits = (valid_red_hits ** 1.5).mean() * weights_to_eval['OPTUNA_RED_HITS_WEIGHT']
        else:
            avg_weighted_red_hits = 0.0
    else:
        avg_weighted_red_hits = 0.0
        logger.warning(f"Optuna试验 {trial.number}: 回测结果中缺少 'red_hits' 列。")

    # Calculate blue hit rate score
    num_periods_in_backtest = backtest_results_df['period'].nunique()
    blue_hit_rate_score = 0.0
    blue_hit_rate_per_period = 0.0

    if num_periods_in_backtest > 0:
        blue_hit_periods_count = extended_bt_stats.get('periods_with_any_blue_hit_count', 0)
        blue_hit_rate_per_period = blue_hit_periods_count / num_periods_in_backtest
        blue_hit_rate_score = blue_hit_rate_per_period * weights_to_eval['OPTUNA_BLUE_HIT_RATE_WEIGHT']

    # Calculate prize level score (normalized by total combinations)
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
    
    total_combinations_in_backtest = extended_bt_stats.get('total_combinations_evaluated', 1)
    prize_score_rate = prize_score / total_combinations_in_backtest if total_combinations_in_backtest > 0 else 0

    # Composite performance score (higher is better)
    performance_score = avg_weighted_red_hits + blue_hit_rate_score + prize_score_rate

    # Add penalty term: If blue hit rate is very low and red hit performance is also poor, give an extra penalty
    if blue_hit_rate_per_period < 0.01 and avg_weighted_red_hits < 0.1:
        performance_score -= 5.0 # Significant penalty

    # Optuna minimizes objective function, so return negative performance score
    return -performance_score

if __name__ == "__main__":
    # 配置日志文件
    log_filename = os.path.join(SCRIPT_DIR, f"ssq_analysis_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(detailed_formatter)
    # 文件日志级别，如果需要看详细的调试信息（如 generate_combinations 内部的debug信息），可以设为logging.DEBUG
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    set_console_verbosity(logging.INFO, use_simple_formatter=True) # 默认控制台显示简洁模式信息

    logger.info(f"--- 双色球分析报告 ---")
    logger.info(f"运行日期: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"日志文件: {log_filename}")
    logger.info(f"红球候选池分数阈值: High > {CANDIDATE_POOL_SCORE_THRESHOLDS['High']}, Medium > {CANDIDATE_POOL_SCORE_THRESHOLDS['Medium']}")
    logger.info(f"ML 特征滞后阶数: {ML_LAG_FEATURES}")
    logger.info(f"ML 交互特征对: {ML_INTERACTION_PAIRS}, 自交互: {ML_INTERACTION_SELF}")
    logger.info("-" * 30)

    # ---- Optuna 优化开关及权重处理逻辑 ----
    logger.info(f"Optuna优化开关 ENABLE_OPTUNA_OPTIMIZATION: {ENABLE_OPTUNA_OPTIMIZATION}")

    if ENABLE_OPTUNA_OPTIMIZATION:
        logger.info(f"\n>>> Optuna优化已启用。将进行权重优化，并使用优化后的权重进行后续分析。")
        
        # 为Optuna准备数据
        main_df_for_opt = None
        # 临时调整控制台日志级别为详细模式，以便查看数据加载和处理的详细过程
        original_console_level_opt_data = global_console_handler.level
        set_console_verbosity(logging.INFO, use_simple_formatter=False)
        
        if os.path.exists(PROCESSED_CSV_PATH):
            df_proc_opt = load_data(PROCESSED_CSV_PATH)
            # 检查已处理文件是否包含必要的列
            required_cols_opt = [f'red{i+1}' for i in range(6)] + ['blue', '期号', 'red_sum', 'red_span', 'red_odd_count', 'blue_is_odd', 'blue_is_large', 'blue_is_prime', 'red_consecutive_pairs', 'red_repeat_count'] + [f'red_{zone}_count' for zone in RED_ZONES.keys()]
            if df_proc_opt is not None and not df_proc_opt.empty and all(c in df_proc_opt.columns for c in required_cols_opt):
                main_df_for_opt = df_proc_opt
                logger.info(f"成功加载已处理数据用于Optuna优化: {PROCESSED_CSV_PATH}")
            else:
                logger.warning(f"已处理数据 {PROCESSED_CSV_PATH} 不完整或无效，将尝试重新处理原始数据。")
        
        if main_df_for_opt is None: # 如果没有成功加载已处理数据
            logger.info(f"为Optuna优化处理原始数据: {CSV_FILE_PATH}")
            df_raw_opt = load_data(CSV_FILE_PATH)
            if df_raw_opt is not None and not df_raw_opt.empty:
                df_clean_opt = clean_and_structure(df_raw_opt)
                if df_clean_opt is not None and not df_clean_opt.empty:
                    main_df_for_opt = feature_engineer(df_clean_opt)
                    # Optuna使用的数据不需要重复保存 processed_csv, 主流程会做
                else: logger.error("Optuna数据清洗失败。")
            else: logger.error("Optuna原始数据加载失败。")
        
        # 恢复控制台日志模式
        set_console_verbosity(original_console_level_opt_data, use_simple_formatter=True)

        # 检查数据量是否满足Optuna优化要求
        # 最少数据量 = 最大滞后 + 1 (预测目标期) + 最小正样本数 + Optuna回测期数
        min_data_for_opt = (max(ML_LAG_FEATURES) if ML_LAG_FEATURES else 0) + 1 + MIN_POSITIVE_SAMPLES_FOR_ML + OPTIMIZATION_BACKTEST_PERIODS
        
        if main_df_for_opt is None or main_df_for_opt.empty:
            logger.error("Optuna优化所需数据准备失败，无法继续优化。将使用默认权重。")
            CURRENT_WEIGHTS = DEFAULT_WEIGHTS.copy()
            save_weights_to_file(WEIGHTS_CONFIG_FILE, CURRENT_WEIGHTS)
        elif len(main_df_for_opt) < min_data_for_opt:
            logger.warning(f"数据不足 ({len(main_df_for_opt)}期) 进行权重优化 (需要至少 {min_data_for_opt}期)。将使用默认权重。")
            CURRENT_WEIGHTS = DEFAULT_WEIGHTS.copy()
            save_weights_to_file(WEIGHTS_CONFIG_FILE, CURRENT_WEIGHTS)
        else:
            # --- 执行Optuna优化 ---
            optuna.logging.set_verbosity(optuna.logging.WARNING) # Optuna自身的日志级别设置为警告，避免其打印过多信息

            # 创建Optuna研究
            optuna_study = optuna.create_study(direction='minimize')
            
            # 设置Optuna超时（可选）
            optuna_timeout_setting = None # 例如：3600 秒 (1小时)
            timeout_message = "不超时" if optuna_timeout_setting is None else f"超时: {optuna_timeout_setting}s"
            logger.info(f"Optuna优化试验次数: {OPTIMIZATION_TRIALS}, {timeout_message}")

            # 启动Optuna优化，并使用自定义回调函数
            start_time_optuna_global = time.time() # 记录开始时间
            estimated_time_logged_global = False # 重置标志
            try:
                optuna_study.optimize(
                    lambda trial_obj: objective(trial_obj, main_df_for_opt.copy(), ML_LAG_FEATURES),
                    n_trials=OPTIMIZATION_TRIALS,
                    timeout=optuna_timeout_setting,
                    n_jobs=1, # Optuna的n_jobs参数，这里设置为1，让其内部的ProcessPoolExecutor处理并行
                    # 修正回调函数调用方式，传递 OPTIMIZATION_TRIALS
                    callbacks=[lambda study, trial: optuna_time_estimation_callback(study, trial, OPTIMIZATION_TRIALS)]
                )
            except optuna.exceptions.OptunaError as e_optuna:
                 logger.error(f"Optuna优化过程中发生错误或被中断: {e_optuna}")
            except KeyboardInterrupt:
                 logger.warning("Optuna优化被用户中断。")

            logger.info(f"\n>>> Optuna优化完成。")
            if optuna_study.best_trial:
                logger.info(f"  最佳目标函数值 (负的性能分): {optuna_study.best_value:.4f}")
                best_params_from_optuna = optuna_study.best_params
                
                # 将Optuna找到的最佳参数更新到 CURRENT_WEIGHTS 中
                temp_updated_weights = DEFAULT_WEIGHTS.copy()
                for key, value in best_params_from_optuna.items():
                     if key in temp_updated_weights:
                         # 根据默认值的类型进行安全转换
                         if isinstance(temp_updated_weights[key], int) and isinstance(value, (int, float)):
                             temp_updated_weights[key] = int(round(value))
                         elif isinstance(temp_updated_weights[key], float) and isinstance(value, (int,float)):
                             temp_updated_weights[key] = float(value)
                         elif isinstance(temp_updated_weights[key], bool) and isinstance(value, bool):
                              temp_updated_weights[key] = value
                         # 对于其他类型（如列表），如果Optuna优化它们，也需要相应的类型检查和赋值
                
                # 再次归一化 CANDIDATE_POOL_PROPORTIONS
                prop_h_opt = temp_updated_weights.get('CANDIDATE_POOL_PROPORTIONS_HIGH', DEFAULT_WEIGHTS['CANDIDATE_POOL_PROPORTIONS_HIGH'])
                prop_m_opt = temp_updated_weights.get('CANDIDATE_POOL_PROPORTIONS_MEDIUM', DEFAULT_WEIGHTS['CANDIDATE_POOL_PROPORTIONS_MEDIUM'])
                if prop_h_opt + prop_m_opt > 1.0 + 1e-9:
                    total_prop_opt = prop_h_opt + prop_m_opt
                    if total_prop_opt > 1e-9:
                        temp_updated_weights['CANDIDATE_POOL_PROPORTIONS_HIGH'] = prop_h_opt / total_prop_opt
                        temp_updated_weights['CANDIDATE_POOL_PROPORTIONS_MEDIUM'] = prop_m_opt / total_prop_opt
                    else:
                        temp_updated_weights['CANDIDATE_POOL_PROPORTIONS_HIGH'] = DEFAULT_WEIGHTS['CANDIDATE_POOL_PROPORTIONS_HIGH']
                        temp_updated_weights['CANDIDATE_POOL_PROPORTIONS_MEDIUM'] = DEFAULT_WEIGHTS['CANDIDATE_POOL_PROPORTIONS_MEDIUM']
                
                CURRENT_WEIGHTS = temp_updated_weights
                logger.info(f"  使用优化后的参数更新当前权重。")
                logger.info(f"  最佳参数 (部分):")
                # 仅显示部分参数，避免输出过长
                params_to_display = {k: v for k, v in list(optuna_study.best_params.items())[:7]} # 显示前7个
                for k_disp, v_disp in params_to_display.items():
                    logger.info(f"    {k_disp}: {v_disp}")

            else:
                logger.warning("  Optuna未找到最佳试验 (可能由于提前中断或无有效试验)。将使用默认权重。")
                CURRENT_WEIGHTS = DEFAULT_WEIGHTS.copy()
            
            save_weights_to_file(WEIGHTS_CONFIG_FILE, CURRENT_WEIGHTS)
            logger.info(f"优化后的权重已记录到: {WEIGHTS_CONFIG_FILE}")
    else: # ENABLE_OPTUNA_OPTIMIZATION is False
        logger.info(f"\n>>> Optuna优化已禁用。尝试从 {WEIGHTS_CONFIG_FILE} 加载权重。")
        loaded_weights_from_file, load_status = load_weights_from_file(WEIGHTS_CONFIG_FILE, DEFAULT_WEIGHTS)
        CURRENT_WEIGHTS = loaded_weights_from_file
        
        if load_status == 'loaded_active_config':
            logger.info(f"成功从 {WEIGHTS_CONFIG_FILE} 加载权重。")
        elif load_status == 'defaults_used_new_config_saved':
            logger.info(f"{WEIGHTS_CONFIG_FILE} 未找到，已使用默认权重并保存到该文件。")
        elif load_status == 'defaults_used_config_error':
            logger.warning(f"{WEIGHTS_CONFIG_FILE} 加载失败，已使用默认权重。请检查文件格式。")
    
    # ---- 权重处理逻辑结束 ----

    # --- 主流程数据加载 ---
    main_df = None # 初始化 main_df
    # 临时调整控制台日志级别为详细模式，以便查看数据加载和处理的详细过程
    original_console_level_main_data = global_console_handler.level
    set_console_verbosity(logging.INFO, use_simple_formatter=False)
    
    if os.path.exists(PROCESSED_CSV_PATH):
        df_proc = load_data(PROCESSED_CSV_PATH)
        # 检查已处理文件是否包含必要的列
        required_cols = [f'red{i+1}' for i in range(6)] + ['blue', '期号', 'red_sum', 'red_span', 'red_odd_count', 'blue_is_odd', 'blue_is_large', 'blue_is_prime', 'red_consecutive_pairs', 'red_repeat_count'] + [f'red_{zone}_count' for zone in RED_ZONES.keys()]
        if df_proc is not None and not df_proc.empty and all(c in df_proc.columns for c in required_cols):
            main_df = df_proc
            logger.info(f"成功加载已处理数据用于主分析: {PROCESSED_CSV_PATH}")
        else:
            logger.warning(f"已处理数据 {PROCESSED_CSV_PATH} 不完整或无效，将尝试重新处理原始数据。")
    
    if main_df is None: # 如果没有成功加载已处理数据
        logger.info(f"为主分析处理原始数据: {CSV_FILE_PATH}")
        df_raw_main = load_data(CSV_FILE_PATH)
        if df_raw_main is not None and not df_raw_main.empty:
            df_clean_main = clean_and_structure(df_raw_main)
            if df_clean_main is not None and not df_clean_main.empty:
                main_df = feature_engineer(df_clean_main)
                if main_df is not None and not main_df.empty:
                    try:
                        # 确保保存前所有列都是可序列化的类型
                        main_df_to_save = main_df.copy()
                        # 布尔类型转换为int，确保CSV兼容性
                        for col in ['blue_is_odd', 'blue_is_large', 'blue_is_prime']:
                            if col in main_df_to_save.columns:
                                main_df_to_save[col] = main_df_to_save[col].astype(int)
                        main_df_to_save.to_csv(PROCESSED_CSV_PATH, index=False)
                        logger.info(f"已处理数据已保存到: {PROCESSED_CSV_PATH}")
                    except Exception as e_csv_save:
                        logger.warning(f"保存已处理数据失败: {e_csv_save}")
                else: logger.error("特征工程失败。")
            else: logger.error("数据清洗失败。")
        else: logger.error("原始数据加载失败。")
    
    # 恢复控制台日志模式
    set_console_verbosity(original_console_level_main_data, use_simple_formatter=True)

    if main_df is None or main_df.empty:
        logger.error("主分析数据准备失败，无法继续。"); sys.exit(1)
    
    # 数据类型最终确认和清理，确保用于计算和模型训练的数据是干净且类型正确的
    for r_col_m in [f'red{i+1}' for i in range(6)]:
        if r_col_m in main_df.columns:
            main_df[r_col_m] = pd.to_numeric(main_df[r_col_m], errors='coerce')
    if 'blue' in main_df.columns:
        main_df['blue'] = pd.to_numeric(main_df['blue'], errors='coerce')
    
    main_df.dropna(subset=([f'red{i+1}' for i in range(6)] + ['blue']), inplace=True)
    if main_df.empty:
        logger.error("数据经过最终清理后为空，无法继续。"); sys.exit(1)

    for r_col_m in [f'red{i+1}' for i in range(6)]:
        if r_col_m in main_df.columns:
            main_df[r_col_m] = main_df[r_col_m].astype(int)
    if 'blue' in main_df.columns:
        main_df['blue'] = main_df['blue'].astype(int)

    # --- 使用最终确定的 CURRENT_WEIGHTS 进行后续分析 ---
    logger.info(f"\n>>> 当前用于分析的权重 (部分展示):")
    keys_to_display = ['NUM_COMBINATIONS_TO_GENERATE', 'ML_PROB_SCORE_WEIGHT_RED',
                       'FINAL_COMBO_REVERSE_ENABLED', 'FINAL_COMBO_REVERSE_REMOVE_TOP_PERCENT',
                       'OMISSION_SCORE_WEIGHT', 'ARM_COMBINATION_BONUS_WEIGHT', 'DIVERSITY_MIN_DIFFERENT_REDS']
    for key_to_show in keys_to_display:
        logger.info(f"  {key_to_show}: {CURRENT_WEIGHTS.get(key_to_show, 'N/A')}")

    logger.info("使用当前权重分析关联规则 (基于全部历史数据)...")
    # 传递副本以避免 analyze_associations 函数修改原始DataFrame
    full_history_arm_rules = analyze_associations(main_df.copy(), CURRENT_WEIGHTS)
    
    # --- 数据概况与统计分析 ---
    min_p_val, max_p_val, total_p_val = main_df['期号'].min(), main_df['期号'].max(), len(main_df)
    last_draw_dt = main_df['日期'].iloc[-1] if '日期' in main_df.columns and not main_df.empty else "未知"
    last_draw_period = main_df['期号'].iloc[-1] if not main_df.empty else "未知"
    
    set_console_verbosity(logging.INFO, use_simple_formatter=False) # 详细模式显示统计概况
    logger.info(f"\n{'='*15} 数据概况 {'='*15}")
    logger.info(f"  数据范围: {min_p_val} - {max_p_val} (共 {total_p_val} 期)")
    logger.info(f"  最后开奖: {last_draw_dt} (期号: {last_draw_period})")

    # 完整分析和回测所需的最小数据量
    min_periods_for_full_run = (max(ML_LAG_FEATURES) if ML_LAG_FEATURES else 0) + 1 + MIN_POSITIVE_SAMPLES_FOR_ML + BACKTEST_PERIODS_COUNT
    if total_p_val < min_periods_for_full_run:
        logger.error(f"数据不足 ({total_p_val}期) 进行完整分析和回测报告 (需 {min_periods_for_full_run}期)。")
        logger.info(f"\n--- 分析报告结束 (因数据不足提前终止，详情请查阅日志文件: {log_filename}) ---")
        sys.exit(0) # 正常退出，但提示数据不足
    
    logger.info(f"\n{'='*10} 完整历史统计分析 {'='*10}")
    # 再次使用详细模式显示统计分析
    full_freq_d = analyze_frequency_omission(main_df, CURRENT_WEIGHTS)
    full_patt_d = analyze_patterns(main_df, CURRENT_WEIGHTS)
    
    logger.info(f"  热门红球 (Top 5): {[int(x) for x in full_freq_d.get('hot_reds', [])[:5]]}")
    logger.info(f"  冷门红球 (Bottom 5): {[int(x) for x in full_freq_d.get('cold_reds', [])[-5:]]}")
    # Ensure recent_N_freq_red has values for sorting
    recent_freq_items = [(int(k),v) for k,v in full_freq_d.get('recent_N_freq_red', {}).items() if v is not None and v > 0]
    if recent_freq_items:
        logger.info(f"  最近 {RECENT_FREQ_WINDOW} 期热门红球: " + str(sorted(recent_freq_items, key=lambda x: x[1], reverse=True)[:5]))
    else:
        logger.info(f"  最近 {RECENT_FREQ_WINDOW} 期无热门红球数据。")
    logger.info(f"  最常见红球奇偶比: {full_patt_d.get('most_common_odd_even_count')}")
    if full_history_arm_rules is not None and not full_history_arm_rules.empty:
        logger.info(f"  发现 {len(full_history_arm_rules)} 条关联规则 (Top 3 LIFT): \n{full_history_arm_rules.head(3).to_string(index=False)}")
    else: logger.info("  未找到显著关联规则.")
    
    # --- 回测摘要 ---
    logger.info(f"\n{'='*15} 回测摘要 {'='*15}")
    set_console_verbosity(logging.INFO, use_simple_formatter=True) # 回测进度用简洁模式
    # 传递副本，避免回测函数修改原始DataFrame
    backtest_res_df, extended_bt_stats = backtest(main_df.copy(), ML_LAG_FEATURES, CURRENT_WEIGHTS, full_history_arm_rules, BACKTEST_PERIODS_COUNT)
    set_console_verbosity(logging.INFO, use_simple_formatter=False) # 回测结果用详细模式

    if not backtest_res_df.empty or extended_bt_stats.get('winning_red_ball_score_percentiles') or extended_bt_stats.get('winning_blue_ball_score_percentiles'):
        s_p_f = backtest_res_df.attrs.get('start_period_id', 'N/A')
        e_p_f = backtest_res_df.attrs.get('end_period_id', 'N/A')
        num_tested_periods = backtest_res_df.attrs.get('num_periods_tested', 'N/A')
        logger.info(f"  回测期范围: {s_p_f} 至 {e_p_f} (共测试 {num_tested_periods} 期)")
        
        if not backtest_res_df.empty:
            logger.info(f"  每期生成组合数: {extended_bt_stats.get('num_combinations_per_draw_tested', 'N/A')}")
            logger.info(f"  总评估组合数: {extended_bt_stats.get('total_combinations_evaluated', 'N/A')}")
            logger.info(f"  --- 整体命中表现 ---")
            
            # 确保 'red_hits' 列是数值型，并处理其中的非数值项（如 -1, 'No combos predicted'）
            valid_red_hits_for_avg = pd.to_numeric(backtest_res_df['red_hits'], errors='coerce').dropna()
            valid_red_hits_for_avg = valid_red_hits_for_avg[valid_red_hits_for_avg >= 0].astype(int) # 排除错误值
            
            if not valid_red_hits_for_avg.empty:
                logger.info(f"    每个组合平均红球命中: {valid_red_hits_for_avg.mean():.3f}")
                logger.info(f"    每个组合加权(x^1.5)平均红球命中: {(valid_red_hits_for_avg**1.5).mean():.3f}")
            else:
                logger.info(f"    无有效红球命中数据进行平均计算。")

            # Check if 'blue_hit' column exists and is boolean/numeric
            if 'blue_hit' in backtest_res_df.columns:
                blue_hit_overall_rate = backtest_res_df['blue_hit'].mean() * 100
                logger.info(f"    蓝球命中率 (每个组合): {blue_hit_overall_rate:.2f}%")
            else:
                logger.info("    无蓝球命中数据进行平均计算。")

            periods_any_blue_hit_count = extended_bt_stats.get('periods_with_any_blue_hit_count', 0)
            if isinstance(num_tested_periods, int) and num_tested_periods > 0:
                logger.info(f"    至少一个组合命中蓝球的期数占比: {periods_any_blue_hit_count / num_tested_periods:.2%}")

            logger.info(f"  --- 红球命中数分布 (按组合) ---")
            if not valid_red_hits_for_avg.empty:
                hit_counts_dist = valid_red_hits_for_avg.value_counts(normalize=True).sort_index() * 100
                for hit_num, pct in hit_counts_dist.items(): logger.info(f"    命中 {hit_num} 红球: {pct:.2f}%")
            else:
                logger.info("    无有效红球命中数据进行分布统计。")

            logger.info(f"  --- 中奖等级统计 (按组合) ---")
            prize_dist = extended_bt_stats.get('prize_counts', {})
            if prize_dist:
                prize_order = {"一等奖": 1, "二等奖": 2, "三等奖": 3, "四等奖": 4, "五等奖": 5, "六等奖": 6}
                sorted_prize_dist = sorted(prize_dist.items(), key=lambda item: prize_order.get(item[0], 99))
                for prize_level, count in sorted_prize_dist: logger.info(f"    {prize_level}: {count} 次")
            else: logger.info("    未命中任何奖级。")

            best_hits_df = extended_bt_stats.get('best_hit_per_period_df')
            if best_hits_df is not None and not best_hits_df.empty:
                logger.info(f"  --- 每期最佳红球命中数分布 ---")
                # 过滤掉 -1 等错误值
                valid_max_red_hits = best_hits_df['max_red_hits'][best_hits_df['max_red_hits'] >= 0]
                if not valid_max_red_hits.empty:
                    best_red_dist = valid_max_red_hits.value_counts(normalize=True).sort_index() * 100
                    for hit_num, pct in best_red_dist.items():
                         logger.info(f"    最佳命中 {hit_num} 红球的期数占比: {pct:.2f}%")
                else:
                    logger.info("    无有效每期最佳红球命中数据。")

                if 'blue_hit_in_period' in best_hits_df.columns and isinstance(num_tested_periods, int) and num_tested_periods > 0:
                     periods_with_best_blue_hit = best_hits_df['blue_hit_in_period'].sum()
                     logger.info(f"    至少一个组合命中蓝球的期数占比 (来自best_hit_per_period): {periods_with_best_blue_hit / num_tested_periods:.2%}")
        
        logger.info(f"\n  --- 中奖号码评分位置分析 (百分位排名) ---")
        red_percentiles = extended_bt_stats.get('winning_red_ball_score_percentiles', [])
        blue_percentiles = extended_bt_stats.get('winning_blue_ball_score_percentiles', [])
        if red_percentiles:
            # Ensure red_percentiles is not empty before calculating mean/median
            if red_percentiles:
                avg_red_percentile = np.mean(red_percentiles)
                median_red_percentile = np.median(red_percentiles)
            else:
                avg_red_percentile = np.nan
                median_red_percentile = np.nan

            logger.info(f"    中奖红球:")
            logger.info(f"      平均百分位排名: {avg_red_percentile:.2f}% (越小越好，表示分数越高)")
            logger.info(f"      中位数百分位排名: {median_red_percentile:.2f}%")
            
            # 计算分位数
            quantiles_to_check_red = [10, 25, 50, 75, 90]
            for q_val in quantiles_to_check_red:
                if len(red_percentiles) >= (q_val / 100.0) * len(red_percentiles) and len(red_percentiles) >= 1: # 确保数据量足够计算分位数
                    logger.info(f"      {q_val}分位数处的百分位排名: {np.percentile(red_percentiles, q_val):.2f}%")
            
            top_10_pct_count_red = sum(1 for p in red_percentiles if p <= 10)
            top_25_pct_count_red = sum(1 for p in red_percentiles if p <= 25)
            if len(red_percentiles) > 0:
                logger.info(f"      落在分数排名前10%的中奖红球比例: {top_10_pct_count_red / len(red_percentiles) * 100:.2f}%")
                logger.info(f"      落在分数排名前25%的中奖红球比例: {top_25_pct_count_red / len(red_percentiles) * 100:.2f}%")
        else: logger.info("    未能收集到中奖红球的评分排名数据。")
        
        if blue_percentiles:
            # Ensure blue_percentiles is not empty
            if blue_percentiles:
                avg_blue_percentile = np.mean(blue_percentiles)
                median_blue_percentile = np.median(blue_percentiles)
            else:
                avg_blue_percentile = np.nan
                median_blue_percentile = np.nan

            logger.info(f"    中奖蓝球:")
            logger.info(f"      平均百分位排名: {avg_blue_percentile:.2f}%"); logger.info(f"      中位数百分位排名: {median_blue_percentile:.2f}%")
            
            quantiles_to_check_blue = [10, 25, 50, 75, 90]
            for q_val in quantiles_to_check_blue:
                if len(blue_percentiles) >= (q_val / 100.0) * len(blue_percentiles) and len(blue_percentiles) >= 1:
                    logger.info(f"      {q_val}分位数处的百分位排名: {np.percentile(blue_percentiles, q_val):.2f}%")
            
            num_blue_options = len(BLUE_BALL_RANGE)
            # 计算前1、前2、前4名对应的百分位阈值
            blue_ranks_pcts = {rank: (rank / num_blue_options) * 100 for rank in [1,2,4]}
            for rank_num, pct_thresh in blue_ranks_pcts.items():
                count_top_rank = sum(1 for p in blue_percentiles if p <= pct_thresh + 1e-9)
                if len(blue_percentiles) > 0:
                    logger.info(f"      落在分数排名前 {pct_thresh:.1f}% (前{rank_num}名) 的中奖蓝球比例: {count_top_rank / len(blue_percentiles) * 100:.2f}%")
        else: logger.info("    未能收集到中奖蓝球的评分排名数据。")
    else:
        logger.info("  最终回测未产生有效结果（无组合评估或分数排名数据）。")

    # --- 最终推荐号码 ---
    logger.info(f"\n{'='*12} 最终推荐号码 {'='*12}")
    set_console_verbosity(logging.INFO, use_simple_formatter=True) # 推荐号码用简洁模式
    # 传递副本，避免函数修改原始DataFrame
    final_recs_list, final_rec_strs_list, _, final_models_for_rec, final_scores_dict, final_win_seg_pcts = analyze_and_recommend(
        main_df.copy(), ML_LAG_FEATURES, CURRENT_WEIGHTS, full_history_arm_rules, train_ml=True
    )
    for line_str in final_rec_strs_list: logger.info(line_str)

    # --- 基于最终分数的历史分析和复式参考 ---
    set_console_verbosity(logging.INFO, use_simple_formatter=False) # 详细模式显示分数段分析和复式参考
    logger.info(f"\n{'='*8} 中奖红球分数段历史分析 (基于最终分数) {'='*8}")
    if final_scores_dict.get('red_scores'):
        disp_cts, disp_pcts_vals = analyze_winning_red_ball_score_segments(main_df, final_scores_dict['red_scores'], SCORE_SEGMENT_BOUNDARIES, SCORE_SEGMENT_LABELS)
        tot_win_reds_d = sum(disp_cts.values())
        if tot_win_reds_d > 0:
            logger.info(f"  历史中奖红球分数段分布 (总计 {tot_win_reds_d} 个红球实例):")
            # Ensure sorting key works for non-numeric labels too (e.g., 'Low')
            # Define a custom sort key for the segment labels
            def segment_sort_key(label):
                if '-' in label and label.split('-')[0].isdigit():
                    return int(label.split('-')[0])
                # Assign arbitrary numeric values for 'High', 'Medium', 'Low' if they are standalone
                elif 'High' in label: return 1000
                elif 'Medium' in label: return 500
                elif 'Low' in label: return 0
                return sys.maxsize # For any other unknown labels
            
            for seg_name in sorted(disp_cts.keys(), key=segment_sort_key):
                logger.info(f"    分数段 {seg_name}: {disp_cts.get(seg_name,0)} 个 ({disp_pcts_vals.get(seg_name,0.0):.2f}%)")
        else: logger.info("  历史中奖红球分数段无有效数据。")
    else: logger.info("  无法进行分数段分析（无红球得分）。")

    logger.info(f"\n{'='*14} 7+7 复式参考 {'='*14}")
    r_s_77_f = final_scores_dict.get('red_scores', {}); b_s_77_f = final_scores_dict.get('blue_scores', {})
    if r_s_77_f and len(r_s_77_f)>=7 and b_s_77_f and len(b_s_77_f)>=7:
        # 获取红蓝球分数最高的前7个，并排序显示
        top_7r_f = sorted([int(n_val) for n_val,_ in sorted(r_s_77_f.items(), key=lambda i_item:i_item[1], reverse=True)[:7]])
        top_7b_f = sorted([int(n_val) for n_val,_ in sorted(b_s_77_f.items(), key=lambda i_item:i_item[1], reverse=True)[:7]])
        logger.info(f"  推荐7红球: {top_7r_f}")
        logger.info(f"  推荐7蓝球: {top_7b_f}")
    else: logger.info("  评分号码不足以选择7+7。")

    logger.info(f"\n--- 分析报告结束 (详情请查阅日志文件: {log_filename}) ---")

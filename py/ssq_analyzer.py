# =============================================================================
# 双色球整合策略分析脚本 (最终修正版)
#
# 功能:
# 1. [已修正] 历史回测: 对最近50期数据进行逻辑严谨的策略回测，并输出详细报告。
# 2. 机器学习评分: 为每个号码训练独立的LGBM模型，并结合频率、遗漏值进行综合评分。
# 3. 规则过滤: 从ML评分选出的“大底”号码中，应用超过15条专家规则进行精选。
# 4. 反向排他: 使用一个大型随机库，排除掉过于“平庸”的组合。
# 5. 交互式输出: 智能判断是否需要用户确认，以显示全部筛选结果。
# 6. 高级推荐: 从最终的精华组合中，生成高覆盖度的7红球大底和随机单式推荐。
# 7. [新增] 动态路径: 采用更灵活的父目录结构来管理数据和报告文件。
# 8. [新增] 报告固化: 自动将屏幕上显示的最终报告保存为带时间戳的txt文件。
#
# 作者: Gemini AI
# 版本: 5.1 (2025-11-13) - 修正了回测逻辑, 并集成了动态路径与报告输出功能
# =============================================================================

# --- 核心库导入 ---
import pandas as pd
import numpy as np
import lightgbm as lgb
from itertools import combinations
from collections import Counter
import datetime
import warnings
from tqdm import tqdm
import os
import json
import random
import time
import threading
import sys

# --- 平台特定模块导入，用于实现非阻塞的键盘输入监听 ---
try:
    import msvcrt # 用于Windows平台的非阻塞输入
except ImportError:
    import select # 用于Linux/Mac平台的非阻塞输入

# 忽略所有警告信息，使输出更整洁
warnings.filterwarnings('ignore')

# --- 1. 全局可调参数与路径设置 ---

# --- 核心策略参数 ---
POOL_SIZE_RED = 16          # 红球大底号码池的大小 (建议15-18)。值越大，候选组合越多，计算时间越长。
NUM_BLUE_BALLS = 7          # 最终推荐的蓝球个数。
REJECTION_LIB_SIZE = 500000  # 随机抛弃库的大小 (建议10000-100000)。值越大，排他性越强，但生成库的时间越长。

# --- 回测与输出参数 ---
BACKTEST_PERIODS = 50       # 执行回测的最近期数。
INTERACTIVE_THRESHOLD = 100 # 当通过检验的组合数超过此值时，启动交互式倒计时，询问用户是否全部显示。
COUNTDOWN_SECONDS = 10      # 交互式输入的倒计时秒数。
NUM_RECOMMENDATIONS = 10    # 报告中最终推荐的单式注数。

# --- 文件路径设置 (已采纳 ssq2 的动态路径方案) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# 假设脚本在子目录中，数据和报告位于上一级目录
root_dir = os.path.dirname(script_dir) 
# 构造历史数据CSV文件的绝对路径
CSV_PATH = os.path.join(root_dir, 'shuangseqiu.csv')
# 构造机器学习优化参数文件的绝对路径
PARAMS_JSON_PATH = os.path.join(root_dir, 'best_params.json')
# 新增: 报告输出目录
REPORT_DIR = os.path.join(root_dir, 'report')

# --- 2. 辅助函数库 ---

def load_and_preprocess_data(filepath=CSV_PATH):
    """
    从CSV文件加载并预处理双色球历史数据。
    
    Args:
        filepath (str): CSV文件路径。
        
    Returns:
        DataFrame: 处理好并按期号升序排列的数据。如果加载或处理失败，返回None。
    """
    try:
        # 读取CSV，假设有表头，并直接指定列名
        df = pd.read_csv(filepath, header=0)
        df.columns = ['期号', '日期', '红球', '蓝球']
    except Exception as e:
        # 如果文件读取失败，打印错误并返回None
        print(f"错误: 无法加载数据文件 '{filepath}': {e}")
        return None
    
    # --- 数据清洗和类型转换 ---
    # 将'红球'列的字符串（如'1,2,3,4,5,6'）转换为排序后的整数列表（如[1, 2, 3, 4, 5, 6]）
    df['红球'] = df['红球'].apply(lambda x: sorted([int(num) for num in str(x).split(',')]))
    # 将'蓝球'列转换为整数类型
    df['蓝球'] = df['蓝球'].astype(int)
    
    # 按'期号'升序排列，并重置索引，确保数据按时间顺序排列
    return df.sort_values('期号').reset_index(drop=True)

def feature_engineer(df):
    """
    为数据集进行特征工程，基于历史数据计算出各种可能影响下一期结果的统计指标。
    
    Args:
        df (DataFrame): 输入的包含'红球'和'蓝球'列的数据。
        
    Returns:
        DataFrame: 增加了18个新特征列的数据。
    """
    # 和值: 6个红球号码之和
    df['red_sum'] = df['红球'].apply(sum)
    # 跨度: 6个红球中最大号码与最小号码的差
    df['red_span'] = df['红球'].apply(lambda x: max(x) - min(x))
    # 奇数个数: 6个红球中奇数的数量
    df['odd_count'] = df['红球'].apply(lambda x: sum(1 for i in x if i % 2 != 0))
    # 蓝球滞后1期: 上一期的蓝球号码
    df['blue_lag1'] = df['蓝球'].shift(1)
    # 小区(1-11)号码个数
    df['red_zone_small'] = df['红球'].apply(lambda x: sum(1 for i in x if 1 <= i <= 11))
    # 中区(12-22)号码个数
    df['red_zone_medium'] = df['红球'].apply(lambda x: sum(1 for i in x if 12 <= i <= 22))
    # 大区(23-33)号码个数
    df['red_zone_large'] = df['红球'].apply(lambda x: sum(1 for i in x if 23 <= i <= 33))
    # 大数(>16)个数
    df['red_big_count'] = df['红球'].apply(lambda x: sum(1 for i in x if i > 16))
    # 红球质数集合 (预先定义以提高效率)
    RED_PRIME_NUMBERS = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
    # 质数个数
    df['red_prime_count'] = df['红球'].apply(lambda x: sum(1 for i in x if i in RED_PRIME_NUMBERS))
    # 和尾: 和值的个位数
    df['red_sum_tail'] = df['red_sum'].apply(lambda x: x % 10)
    
    def count_consecutive_groups(nums):
        """计算一组号码中的连号组数 (例如 [1,2, 4,5] 有2组连号)"""
        groups = 0
        in_group = False
        for i in range(len(nums) - 1):
            if nums[i+1] - nums[i] == 1:
                if not in_group:
                    groups += 1
                    in_group = True
            else:
                in_group = False
        return groups
    # 连号组数
    df['red_consecutive_groups'] = df['红球'].apply(count_consecutive_groups)
    # AC值: 号码间两两之差的绝对值的唯一数量
    df['red_ac_value'] = df['红球'].apply(lambda nums: len(set(abs(n1-n2) for n1, n2 in combinations(nums, 2))))
    # 尾数唯一值个数: 6个号码的个位数有多少种
    df['red_tail_uniques'] = df['红球'].apply(lambda x: len(set(n % 10 for n in x)))
    
    # --- 涉及多期数据的移动平均(MA)和滞后(Lag)特征 ---
    window_size = 5 # 定义移动平均的窗口大小
    # 和值滞后1期
    df['red_sum_lag1'] = df['red_sum'].shift(1)
    # 奇数个数滞后1期
    df['odd_count_lag1'] = df['odd_count'].shift(1)
    # 和值5期移动平均 (使用shift(1)确保不包含当期数据)
    df['red_sum_ma5'] = df['red_sum'].shift(1).rolling(window=window_size).mean()
    # 奇数个数5期移动平均
    df['odd_count_ma5'] = df['odd_count'].shift(1).rolling(window=window_size).mean()
    # 蓝球5期移动平均
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
    for i in range(1, 34):
        # 查找号码i最后一次出现的索引位置
        last_occurrence = df[df['红球'].apply(lambda x: i in x)].index.max()
        # 如果号码从未出现过(last_occurrence为NaN)，则遗漏值为总期数
        if pd.isna(last_occurrence):
            omission = total_draws
        else:
            # 否则，遗漏值为总期数 - 最后出现位置的索引 - 1
            omission = total_draws - last_occurrence - 1
        red_omission[i] = omission
    return red_omission

def get_weighted_frequency(series, decay_factor):
    """
    计算时间衰减加权频率。越近的期数权重越高。
    
    Args:
        series (pd.Series): 一个包含号码列表的Series (例如df['红球'])。
        decay_factor (float): 衰减因子，越接近1，时间权重衰减越慢。
        
    Returns:
        pd.Series: 每个号码的加权频率。
    """
    N = len(series)
    # 创建一个权重数组，最近的期数权重最高 (decay_factor^0=1)，最远的最低
    weights = np.array([decay_factor ** (N - i - 1) for i in range(N)])
    weighted_counts = {}
    # 遍历每一期的号码列表
    for i, sublist in enumerate(series):
        # 为该期的每个号码加上对应的权重
        for ball in sublist:
            weighted_counts[ball] = weighted_counts.get(ball, 0) + weights[i]
    return pd.Series(weighted_counts)

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
    # .iloc[[-1]] 确保返回的是DataFrame而不是Series
    last_features = df_history.iloc[[-1]][feature_columns].copy()
    # 如果最新特征中有空值（通常是由于移动平均窗口不足），用历史均值填充
    for col in last_features.columns:
        if last_features[col].isnull().any():
            last_features[col].fillna(df_history[col].mean(), inplace=True)
    
    # 2. 红球评分
    # 计算红球的时间衰减加权频率
    red_weighted_freq = get_weighted_frequency(df_history['红球'], params['decay_factor'])
    # 计算红球的当前遗漏值
    red_omission = get_omission(df_history)
    # 使用ML模型预测每个红球下一期出现的概率
    red_ml_probs = {ball: ml_models_red[ball].predict_proba(last_features)[:, 1][0] for ball in range(1, 34)}
    
    red_scores = {}
    # 归一化处理，避免量纲影响
    max_red_freq = red_weighted_freq.max() or 1 # or 1 防止除以0
    max_red_omission = max(red_omission.values()) or 1
    
    for ball in range(1, 34):
        # 归一化频率
        norm_freq = red_weighted_freq.get(ball, 0) / max_red_freq
        # 归一化遗漏值
        norm_omission = red_omission.get(ball, 0) / max_red_omission
        # 综合评分 = 频率分 * 权重 + 遗漏分 * 权重 + ML预测分 * 权重
        red_scores[ball] = (norm_freq * params['weight_freq'] + 
                            norm_omission * params['weight_omission'] + 
                            red_ml_probs[ball] * params['weight_ml'])
    
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
        # 蓝球综合评分
        blue_scores[ball] = (norm_blue_freq * params['weight_blue_freq'] + 
                             blue_ml_probs[ball] * params['weight_blue_ml'])
        
    return red_scores, blue_scores


# --- 规则过滤函数库 (每个函数都是一条独立的过滤规则) ---
# r: 代表一个已排序的6红球组合, e.g., (1, 5, 10, 12, 23, 31)

PRIMES_IN_33 = {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31} # 预计算质数集合 (注意：1在数学上不是质数，但彩票分析中常被包含)
def is_prime(n): return n in PRIMES_IN_33 # 判断是否是质数的辅助函数

def calculate_ac_value(r): 
    """计算AC值（算术复杂度），并进行一个自定义调整 (-5)"""
    standard_ac = len(set(abs(n1 - n2) for n1, n2 in combinations(r, 2)))
    return standard_ac - 5 # 减5是一个自定义调整

def filter_highly_regular(r): 
    """过滤掉高度规律的组合 (例如等差数列 '2 4 6 8 10 12')"""
    # 如果所有相邻数字的差值都一样，则该组合只有一个差值，集合长度为1，被过滤
    return len(set(r[i+1] - r[i] for i in range(len(r) - 1))) > 1

def filter_sum_value(r): 
    """过滤和值过高或过低的组合。大部分中奖号码的和值集中在中间区域。"""
    return 70 <= sum(r) <= 160

def filter_span(r): 
    """过滤跨度过小的组合。跨度指最大号与最小号之差。"""
    return (r[-1] - r[0]) >= 15

def filter_consecutive_numbers(r):
    """过滤连号过多的组合。例如，不允许出现3组连号或4连号及以上。"""
    groups, max_c, current_c = 0, 0, 1 # 连号组数, 最大连号长度, 当前连号长度
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
    """过滤掉所有号码都集中在同一个区域的组合。"""
    all_in_small = all(b <= 11 for b in r)
    all_in_medium = all(12 <= b <= 22 for b in r)
    all_in_large = all(b >= 23 for b in r)
    return not (all_in_small or all_in_medium or all_in_large)

def filter_ac_value(r): 
    """过滤AC值不在理想范围的组合。"""
    # 这里的 6-10 是基于自定义AC值，相当于标准AC值的 11-15
    return 6 <= calculate_ac_value(r) <= 10

def filter_prime_composite_ratio(r): 
    """过滤质数个数极端（过多或过少）的组合。"""
    # 排除质数个数为 0, 1, 5, 6 的组合
    return sum(1 for ball in r if is_prime(ball)) not in [0, 1, 5, 6]

def filter_big_small_ratio(r): 
    """过滤大小比极端（大数或小数过多）的组合。以16为界。"""
    # 排除小数(<=16)个数为 0, 1, 5, 6 的组合
    return sum(1 for ball in r if ball <= 16) not in [0, 1, 5, 6]

def filter_recent_overlap(r, last_10_draws_sets):
    """过滤与最近10期内任一期号码重合数达到4个或以上的组合。"""
    candidate_set = set(r)
    for draw_set in last_10_draws_sets:
        if len(candidate_set.intersection(draw_set)) >= 4:
            return False # 如果重合过多，过滤
    return True

def filter_all_cold(r, omission_values): 
    """过滤掉全部由冷号（遗漏值大于15）组成的组合。"""
    return not all(omission_values.get(ball, 0) > 15 for ball in r)

def filter_odd_even_ratio(r): 
    """过滤奇偶比极端（全奇、全偶或接近全奇全偶）的组合。"""
    # 排除偶数个数为 0, 1, 5, 6 的组合
    return sum(1 for n in r if n % 2 == 0) not in [0, 1, 5, 6]

def filter_modulo3_roads(r): 
    """过滤除3余数。要求0路、1路、2路号码都必须出现。"""
    # {n % 3 for n in r} 会得到一个包含所有余数的集合，如{0, 1, 2}
    return len({n % 3 for n in r}) == 3

def filter_ending_digits(r):
    """过滤尾数分布。不允许出现3个以上同尾号，或尾数种类过少。"""
    tails = [n % 10 for n in r]
    counts = Counter(tails) # 统计每个尾数出现的次数
    # 如果任一尾数出现次数>=3，或不同尾数少于3种，则过滤
    return not (max(counts.values()) >= 3 or len(counts) <= 2)

def filter_head_tail_range(r): 
    """过滤首尾号码范围。龙头(第一个号)不宜过大，凤尾(最后一个号)不宜过小。"""
    return not (r[0] > 10 or r[-1] < 25)

def filter_sum_of_tails(r): 
    """过滤尾数和值。所有号码的个位数之和应在一个合理范围。"""
    return 15 <= sum(n % 10 for n in r) <= 45

def filter_related_numbers(r, last_draw_set):
    """过滤与上期号码关联度低的组合。要求至少包含1个重号或边号。"""
    combo_set = set(r)
    # 重号：与上期相同的号码
    repeats = combo_set.intersection(last_draw_set)
    # 边号：与上期号码加减1的号码
    adjacents = combo_set.intersection({n - 1 for n in last_draw_set} | {n + 1 for n in last_draw_set})
    # 如果重号和边号的总数大于0，则保留
    return (len(repeats) + len(adjacents)) > 0

def filter_diagonal_consecutive(r, last_draw_set, last_2_draw_set):
    """过滤斜连号。例如，上上期有10，上期有11，本期组合中不应出现12。"""
    for n in r:
        if (n - 1) in last_draw_set and (n - 2) in last_2_draw_set:
            return False # 发现斜连号，过滤
    return True

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
    
    # 将列表转为集合以获得O(1)的查找速度
    passed_combos_set = set(passed_combos_tuples)
    
    # 以每个通过的6红球组合作为“种子”
    unique_seeds = passed_combos_set
    
    best_7_red_combos = {} # 用于存储找到的7红球组合及其覆盖度
    
    # 遍历每个种子
    for seed_combo in tqdm(unique_seeds, desc="生成7红球大底", leave=False, ncols=80):
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
                    coverage += 1
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
            if msvcrt.kbhit(): # 如果有键盘敲击
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

def run_full_backtest(full_df, params, feature_columns, num_periods):
    """
    [逻辑已修正] 对最近N期执行逻辑严谨的策略回测。
    
    核心修正:
    此函数现在会在回测的每一步循环中，仅使用当前步骤之前的数据来重新训练一套全新的机器学习模型。
    这避免了“用未来的数据预测过去”的逻辑错误，保证了回测结果的有效性。
    
    性能警告:
    由于需要反复训练模型(回测期数 * 49次)，此函数运行会非常缓慢，这是保证准确性的必要开销。
    """
    print("\n" + "="*70)
    print(f"        [新功能] 最近 {num_periods} 期完整策略回测报告 (逻辑严谨版)")
    print("="*70)
    
    # 检查是否有足够的数据进行回测 (需要回测期数 + 至少50期用于模型训练)
    if len(full_df) < num_periods + 50:
        print(f"历史数据不足 {num_periods + 50} 期，无法执行回测。跳过此步骤。")
        return

    # 初始化统计变量
    prize_counts = Counter() # 用于统计各奖项的命中次数
    # 双色球奖金规则字典: (命中红球数, 命中蓝球数): 奖金
    PRIZE_RULES = {(6, 1): 5000000, (6, 0): 100000, (5, 1): 3000, (5, 0): 200, (4, 1): 200, (4, 0): 10, (3, 1): 10, (2, 1): 5, (1, 1): 5, (0, 1): 5}
    total_cost, total_winnings = 0, 0
    total_passed_combos = 0

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
            
            # --- 核心修正部分: 在每次循环内部，仅使用当前的历史数据重新训练模型 ---
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
                    lgb_clf_r = lgb.LGBMClassifier(random_state=42, verbose=-1); lgb_clf_r.fit(X_r, y_r)
                    local_ml_models_red[ball_r] = lgb_clf_r
            
            # 训练蓝球模型
            for ball_b in range(1, 17):
                training_data_for_step[f'blue_{ball_b}_next'] = training_data_for_step['蓝球'].apply(lambda x: 1 if x == ball_b else 0).shift(-1)
                df_temp_b = training_data_for_step.dropna(subset=feature_columns + [f'blue_{ball_b}_next'])
                X_b, y_b = df_temp_b[feature_columns], df_temp_b[f'blue_{ball_b}_next']
                if not X_b.empty:
                    lgb_clf_b = lgb.LGBMClassifier(random_state=42, verbose=-1); lgb_clf_b.fit(X_b, y_b)
                    local_ml_models_blue[ball_b] = lgb_clf_b
            
            if len(local_ml_models_red) != 33 or len(local_ml_models_blue) != 16: # 如果模型训练不完整，跳过
                pbar.update(1)
                continue
            # --- 核心修正部分结束 ---

            # 2. 使用刚刚训练好的【局部模型】进行评分和筛选
            red_scores, blue_scores = run_strategy_and_get_scores(history_df_for_step, params, local_ml_models_red, local_ml_models_blue, feature_columns)
            
            red_pool = [item[0] for item in sorted(red_scores.items(), key=lambda x: (-x[1], x[0]))[:POOL_SIZE_RED]]
            recommended_blue = sorted(blue_scores, key=blue_scores.get, reverse=True)[0]
            
            rejection_set = {tuple(sorted(random.sample(range(1, 34), 6))) for _ in range(REJECTION_LIB_SIZE)}
            
            omission = get_omission(history_df_for_step)
            last_10 = [set(d) for d in history_df_for_step.iloc[-10:]['红球'].tolist()]
            last_1 = last_10[-1]; last_2 = last_10[-2]
            
            potential_combos = list(combinations(sorted(red_pool), 6))
            passed_combos = []
            for r in potential_combos:
                if (filter_highly_regular(r) and filter_sum_value(r) and filter_span(r) and
                    filter_consecutive_numbers(r) and filter_zones(r) and filter_ac_value(r) and
                    filter_big_small_ratio(r) and filter_recent_overlap(r, last_10) and
                    filter_all_cold(r, omission) and filter_odd_even_ratio(r) and
                    filter_modulo3_roads(r) and filter_ending_digits(r) and
                    filter_head_tail_range(r) and filter_sum_of_tails(r) and
                    filter_related_numbers(r, last_1) and filter_diagonal_consecutive(r, last_1, last_2) and
                    (r not in rejection_set)):
                    passed_combos.append(r)
            
            # 3. 计算当期成本和收益
            if passed_combos:
                period_cost = len(passed_combos) * 2
                period_winnings = 0
                for combo in passed_combos:
                    red_hits = len(set(combo) & actual_red_set)
                    blue_hit = 1 if recommended_blue == actual_blue else 0
                    hit_key = (red_hits, blue_hit)
                    prize = PRIZE_RULES.get(hit_key, 0)
                    if prize > 0:
                        period_winnings += prize
                        prize_counts[hit_key] += 1
                total_cost += period_cost
                total_winnings += period_winnings

            total_passed_combos += len(passed_combos)
            pbar.update(1)

    # 4. 打印回测报告
    print("\n--- 回测结果摘要 ---")
    print(f"回测期数: {num_periods} 期")
    print(f"平均每期生成组合数: {total_passed_combos / num_periods:.2f} 组")
    print(f"总投入: {total_cost} 元")
    print(f"总回报: {total_winnings} 元")
    print(f"净利润: {total_winnings - total_cost} 元")
    print("\n--- 命中详情 ---")
    if sum(prize_counts.values()) == 0:
        print("  在回测期间未命中任何奖项。")
    else:
        # 按奖金从高到低排序显示
        sorted_prizes = sorted(prize_counts.items(), key=lambda item: PRIZE_RULES.get(item[0], 0), reverse=True)
        for (r_hit, b_hit), count in sorted_prizes:
            print(f"  命中 {r_hit}+{b_hit} : {count} 次")
    print("-" * 70 + "\n")


# --- 5. 主执行逻辑 ---
if __name__ == '__main__':
    # 打印脚本标题
    print("="*70)
    print("         双色球整合策略分析脚本 (逻辑修正版)")
    print("="*70)

    # --- [阶段 1/8] 加载与特征工程 ---
    print("\n[阶段 1/8] 正在加载和处理历史数据...")
    full_df = load_and_preprocess_data()
    # 如果数据加载失败或数据量太少，则退出程序
    if full_df is None or len(full_df) < 50:
        exit("错误: 历史数据加载失败或数据量过少（至少需要50期），程序终止。")
    full_df = feature_engineer(full_df)
    FEATURE_COLUMNS = [col for col in full_df.columns if col not in ['期号', '日期', '红球', '蓝球']]
    print("数据加载与特征工程完成。")

    # --- [阶段 2/8] 执行严谨的历史回测 ---
    # 尝试加载外部JSON文件中的参数，如果失败则使用默认值
    try:
        with open(PARAMS_JSON_PATH, 'r') as f:
            params = json.load(f)
    except FileNotFoundError:
        print(f"警告: 未找到参数文件 {PARAMS_JSON_PATH}，将使用默认参数。")
        params = {'decay_factor': 0.995, 'weight_freq': 0.3, 'weight_omission': 0.4, 'weight_ml': 0.3, 
                  'weight_blue_freq': 0.5, 'weight_blue_ml': 0.5}
    run_full_backtest(full_df, params, FEATURE_COLUMNS, BACKTEST_PERIODS)
    
    # --- [阶段 3/8] 训练最终预测模型 ---
    print("\n[阶段 3/8] 正在使用全部历史数据，训练用于最终预测的模型...")
    final_ml_models_red, final_ml_models_blue = {}, {}
    ml_training_df = full_df.iloc[5:].copy()
    
    # 训练33个独立的红球模型
    for i in tqdm(range(1, 34), desc="训练最终红球模型", ncols=80):
        ml_training_df[f'red_{i}_next'] = ml_training_df['红球'].apply(lambda x: 1 if i in x else 0).shift(-1)
        df_temp = ml_training_df.dropna(subset=FEATURE_COLUMNS + [f'red_{i}_next'])
        X, y = df_temp[FEATURE_COLUMNS], df_temp[f'red_{i}_next']
        if not X.empty:
            lgb_clf = lgb.LGBMClassifier(random_state=42, verbose=-1)
            lgb_clf.fit(X, y)
            final_ml_models_red[i] = lgb_clf
            
    # 训练16个独立的蓝球模型
    for i in tqdm(range(1, 17), desc="训练最终蓝球模型", ncols=80):
        ml_training_df[f'blue_{i}_next'] = ml_training_df['蓝球'].apply(lambda x: 1 if x == i else 0).shift(-1)
        df_temp = ml_training_df.dropna(subset=FEATURE_COLUMNS + [f'blue_{i}_next'])
        X, y = df_temp[FEATURE_COLUMNS], df_temp[f'blue_{i}_next']
        if not X.empty:
            lgb_clf = lgb.LGBMClassifier(random_state=42, verbose=-1)
            lgb_clf.fit(X, y)
            final_ml_models_blue[i] = lgb_clf
    
    # --- [阶段 4/8] 执行对下一期的预测 ---
    print(f"\n[阶段 4/8] 正在为下一期号码进行机器学习评分...")
    red_scores, blue_scores = run_strategy_and_get_scores(full_df, params, final_ml_models_red, final_ml_models_blue, FEATURE_COLUMNS)
    # 按评分从高到低排序
    red_scores_list = sorted(list(red_scores.items()), key=lambda item: (-item[1], item[0]))
    # 选出评分最高的N个红球作为大底
    red_pool = [item[0] for item in red_scores_list[:POOL_SIZE_RED]]
    # 选出评分最高的M个蓝球
    recommended_blues = sorted(blue_scores, key=blue_scores.get, reverse=True)[:NUM_BLUE_BALLS]
    print(f"已根据ML评分选出 {POOL_SIZE_RED} 个红球大底: {sorted(red_pool)}")

    # --- [阶段 5/8] 规则过滤 ---
    print(f"\n[阶段 5/8] 正在从大底中生成组合并应用所有规则进行过滤...")
    # 提前生成反向排他库
    rejection_set = {tuple(sorted(random.sample(range(1, 34), 6))) for _ in tqdm(range(REJECTION_LIB_SIZE), desc="生成随机抛弃库", ncols=80)}
    # 提前计算过滤所需的历史数据
    omission_values = get_omission(full_df)
    last_10_draws_sets = [set(d) for d in full_df.iloc[-10:]['红球'].tolist()]
    last_draw_set = last_10_draws_sets[-1]
    last_2_draw_set = last_10_draws_sets[-2]
    
    # 从红球大底中生成所有可能的6球组合
    potential_combos = list(combinations(sorted(red_pool), 6))
    passed_combos_tuples = []
    # 遍历所有潜在组合，应用过滤规则
    for r in tqdm(potential_combos, desc="规则过滤进度", ncols=80):
        if (filter_highly_regular(r) and filter_sum_value(r) and filter_span(r) and
            filter_consecutive_numbers(r) and filter_zones(r) and filter_ac_value(r) and
            filter_big_small_ratio(r) and filter_recent_overlap(r, last_10_draws_sets) and
            filter_all_cold(r, omission_values) and filter_odd_even_ratio(r) and
            filter_modulo3_roads(r) and filter_ending_digits(r) and filter_head_tail_range(r) and
            filter_sum_of_tails(r) and filter_related_numbers(r, last_draw_set) and
            filter_diagonal_consecutive(r, last_draw_set, last_2_draw_set) and
            (r not in rejection_set)): # 最后检查是否在抛弃库中
            passed_combos_tuples.append(r)
    print(f"过滤完成！共有 {len(passed_combos_tuples)} 组号码通过所有规则检验。")

    # --- [阶段 6/8] 交互式输出 ---
    # 如果通过的组合数在 0 和阈值之间，直接全部打印
    if 0 < len(passed_combos_tuples) < INTERACTIVE_THRESHOLD:
        print(f"\n通过检验的组合数量为 {len(passed_combos_tuples)} (低于{INTERACTIVE_THRESHOLD})，全部输出如下：")
        for i, combo in enumerate(passed_combos_tuples, 1):
            print(f"  组合 {i:>2}: {' '.join(f'{n:02d}' for n in combo)}")
    # 如果组合数过多，启动交互式倒计时
    elif len(passed_combos_tuples) >= INTERACTIVE_THRESHOLD:
        input_thread = threading.Thread(target=get_user_input_with_timeout, args=(COUNTDOWN_SECONDS,))
        input_thread.start() # 启动子线程等待输入
        input_thread.join() # 主线程等待子线程结束
        if user_input_flag == 'y': # 如果用户确认
            print("\n根据您的确认，输出所有通过检验的组合：")
            for i, combo in enumerate(passed_combos_tuples, 1):
                print(f"  组合 {i:>3}: {' '.join(f'{n:02d}' for n in combo)}")

    # --- [阶段 7/8] 高级推荐 ---
    print(f"\n[阶段 7/8] 正在从最终组合中，生成高重合度的7红球大底...")
    best_7_reds = find_best_7_red_combinations(passed_combos_tuples, red_pool)

    # --- [阶段 8/8] 最终报告与输出 (已采纳 ssq2 的报告生成方案) ---
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("                          最终推荐报告")
    report_lines.append("="*70)
    
    latest_issue = str(full_df.iloc[-1]['期号'])
    target_issue = int(latest_issue) + 1 if latest_issue.isdigit() else f"{latest_issue}_Next"
    
    report_lines.append(f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"数据基于期号: {latest_issue}")
    report_lines.append(f"预测目标期号: {target_issue}\n")
    
    report_lines.append(f"--- 推荐蓝球 (Top {NUM_BLUE_BALLS}) ---")
    report_lines.append(f"  {recommended_blues}\n")
    
    report_lines.append(f"--- 高重合度 7 红球大底推荐 ---")
    if not best_7_reds:
        report_lines.append("  未能生成7红球大底 (原因: 通过规则的组合数量不足)。")
    else:
        report_lines.append("  (覆盖度: 指该7红球大底能命中多少个通过规则的6红球组合)")
        for combo, coverage in best_7_reds[:5]:
            combo_str = ' '.join(f'{n:02d}' for n in combo)
            report_lines.append(f"  组合: {combo_str} (覆盖度: {coverage})")
            
    report_lines.append(f"\n--- 单式红球组合推荐 (随机 {NUM_RECOMMENDATIONS} 组) ---")
    if not passed_combos_tuples:
        report_lines.append("  未找到符合所有规则的单式组合。")
    else:
        final_selection = random.sample(passed_combos_tuples, min(len(passed_combos_tuples), NUM_RECOMMENDATIONS))
        for i, combo in enumerate(final_selection, 1):
            combo_str = ' '.join(f'{n:02d}' for n in combo)
            report_lines.append(f"  组合 {i:>2}: {combo_str}")
            
    report_lines.append("\n" + "="*70)
    report_lines.append("报告结束。祝您好运！")
    report_lines.append("="*70)
    
    # 整合报告为单一字符串并打印
    final_report_string = "\n".join(report_lines)
    print("\n" + final_report_string)

    # 尝试将报告写入文件
    try:
        os.makedirs(REPORT_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ssq_analysis_output_{timestamp}.txt"
        filepath = os.path.join(REPORT_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(final_report_string)
        print(f"\n报告已成功保存到文件: {filepath}")
    except Exception as e:
        print(f"\n写入报告文件失败: {e}")

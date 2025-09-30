import pandas as pd
import numpy as np
import lightgbm as lgb
from itertools import combinations, product
from joblib import Parallel, delayed
import datetime
import warnings
from tqdm import tqdm
import os

warnings.filterwarnings('ignore')

# --- 动态路径设置 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
CSV_PATH = os.path.join(root_dir, 'shuangseqiu.csv')
REPORT_DIR = os.path.join(root_dir, 'report')


# --- 1. 数据加载与基础特征工程 ---

def load_and_preprocess_data(filepath=CSV_PATH):
    """加载并预处理指定格式的CSV数据，默认第一行为表头并跳过。"""
    try:
        df = pd.read_csv(filepath, header=0)
        df.columns = ['期号', '日期', '红球', '蓝球']
    except Exception as e:
        print(f"读取CSV文件 {filepath} 失败: {e}")
        return None
    df['红球'] = df['红球'].apply(lambda x: sorted([int(num) for num in str(x).split(',')]))
    df['蓝球'] = df['蓝球'].astype(int)
    df = df.sort_values('期号').reset_index(drop=True)
    return df

# --- MODIFICATION: 替换为包含更丰富特征的 feature_engineer 函数 ---
# 在函数外部先定义好质数列表，避免重复计算
RED_PRIME_NUMBERS = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}

def feature_engineer(df):
    """为ML模型创建扩展特征集。"""
    # --- 原始特征 ---
    df['red_sum'] = df['红球'].apply(sum)
    df['red_span'] = df['红球'].apply(lambda x: max(x) - min(x))
    df['odd_count'] = df['红球'].apply(lambda x: sum(1 for i in x if i % 2 != 0))
    df['blue_lag1'] = df['蓝球'].shift(1)

    # --- 新增特征 ---

    # 1. 分区特征 (小: 1-11, 中: 12-22, 大: 23-33)
    df['red_zone_small'] = df['红球'].apply(lambda x: sum(1 for i in x if 1 <= i <= 11))
    df['red_zone_medium'] = df['红球'].apply(lambda x: sum(1 for i in x if 12 <= i <= 22))
    df['red_zone_large'] = df['红球'].apply(lambda x: sum(1 for i in x if 23 <= i <= 33))

    # 2. 大小比特征 (大数 > 16)
    df['red_big_count'] = df['红球'].apply(lambda x: sum(1 for i in x if i > 16))

    # 3. 质合比特征
    df['red_prime_count'] = df['红球'].apply(lambda x: sum(1 for i in x if i in RED_PRIME_NUMBERS))

    # 4. 和尾特征
    df['red_sum_tail'] = df['red_sum'].apply(lambda x: x % 10)

    # 5. 连续性特征 (统计连号组数)
    def count_consecutive_groups(nums):
        if not nums:
            return 0
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
    df['red_consecutive_groups'] = df['红球'].apply(count_consecutive_groups)

    # 6. AC值 (Arithmetical Complexity)
    def calculate_ac(nums):
        return len(set(abs(n1 - n2) for n1, n2 in combinations(nums, 2)))
    df['red_ac_value'] = df['红球'].apply(calculate_ac)

    # 7. 尾数特征 (统计不同尾数的个数)
    df['red_tail_uniques'] = df['红球'].apply(lambda x: len(set(n % 10 for n in x)))

    # 8. 更多滞后与移动平均特征
    window_size = 5 # 注意：这个值要和主程序中的 WINDOW_SIZE 保持一致
    df['red_sum_lag1'] = df['red_sum'].shift(1)
    df['odd_count_lag1'] = df['odd_count'].shift(1)
    df['red_sum_ma5'] = df['red_sum'].shift(1).rolling(window=window_size).mean()
    df['odd_count_ma5'] = df['odd_count'].shift(1).rolling(window=window_size).mean()
    df['blue_ma5'] = df['蓝球'].shift(1).rolling(window=window_size).mean()

    return df


# --- 2. 核心策略与评分函数 ---

def get_weighted_frequency(series, decay_factor):
    N = len(series)
    weights = np.array([decay_factor ** (N - i - 1) for i in range(N)])
    weighted_counts = {}
    for i, sublist in enumerate(series):
        for ball in sublist:
            weighted_counts[ball] = weighted_counts.get(ball, 0) + weights[i]
    return pd.Series(weighted_counts)

def get_omission(df):
    red_omission = {}
    for i in range(1, 34):
        last_occurrence = df[df['红球'].apply(lambda x: i in x)].index.max()
        omission = len(df) - last_occurrence - 1 if not pd.isna(last_occurrence) else len(df)
        red_omission[i] = omission
    return red_omission

# --- MODIFICATION: 更新 run_strategy_and_get_scores 以使用全部特征 ---
def run_strategy_and_get_scores(df_history, params, ml_models_red, ml_models_blue, feature_columns):
    # 使用传入的特征列表
    last_features = df_history.iloc[[-1]][feature_columns].copy()

    # 填充可能因数据不足产生的NaN值 (例如回测初期)
    for col in last_features.columns:
        if last_features[col].isnull().any():
            # 使用历史数据该列的均值进行填充
            fill_value = df_history[col].mean()
            last_features[col].fillna(fill_value, inplace=True)

    red_weighted_freq = get_weighted_frequency(df_history['红球'], params['decay_factor'])
    red_omission = get_omission(df_history)
    red_ml_probs = {ball: ml_models_red[ball].predict_proba(last_features)[:, 1][0] for ball in range(1, 34)}
    
    red_scores = {}
    for ball in range(1, 34):
        norm_freq = red_weighted_freq.get(ball, 0) / (red_weighted_freq.max() or 1)
        norm_omission = red_omission.get(ball, 0) / (max(red_omission.values()) or 1)
        red_scores[ball] = (norm_freq * params['weight_freq'] +
                            norm_omission * params['weight_omission'] +
                            red_ml_probs[ball] * params['weight_ml'])
    
    recent_hot_draws = df_history.tail(params['hot_lookback'])
    hot_counts = pd.Series([ball for sublist in recent_hot_draws['红球'] for ball in sublist]).value_counts()
    for ball in hot_counts[hot_counts >= params['hot_threshold']].index:
        red_scores[ball] *= params['hot_bonus']
        
    recent_cold_draws = df_history.tail(params['cold_lookback'])
    cold_numbers = set(range(1, 34)) - set([ball for sublist in recent_cold_draws['红球'] for ball in sublist])
    for ball in cold_numbers:
        red_scores[ball] *= params['cold_bonus']
        
    for ball in df_history.iloc[-1]['红球']:
        red_scores[ball] *= params['repeat_bonus']
        
    blue_weighted_freq = get_weighted_frequency(df_history['蓝球'].apply(lambda x: [x]), params['decay_factor'])
    blue_ml_probs = {ball: ml_models_blue[ball].predict_proba(last_features)[:, 1][0] for ball in range(1, 17)}
    
    blue_scores = {}
    for ball in range(1, 17):
        norm_blue_freq = blue_weighted_freq.get(ball, 0) / (blue_weighted_freq.max() or 1)
        blue_scores[ball] = (norm_blue_freq * params['weight_blue_freq'] + 
                             blue_ml_probs[ball] * params['weight_blue_ml'])
    return red_scores, blue_scores


# --- 3. 组合生成与回测评估 ---

def generate_recommendations(red_scores, blue_scores, num_single=10):
    sorted_reds_by_score = sorted(red_scores, key=red_scores.get, reverse=True)
    sorted_blues_by_score = sorted(blue_scores, key=blue_scores.get, reverse=True)
    
    duplex_reds = sorted_reds_by_score[:7]
    duplex_blues = sorted_blues_by_score[:7]
    
    core_balls = sorted_reds_by_score[:4]
    pool_balls = sorted_reds_by_score[4:12]
    
    single_combinations = []
    if len(pool_balls) >= 2:
        for tow_combo in combinations(pool_balls, 2):
            single_combo = sorted(core_balls + list(tow_combo))
            single_combinations.append(single_combo)
            if len(single_combinations) >= num_single:
                break
    
    return single_combinations, (sorted(duplex_reds), sorted(duplex_blues))

# --- MODIFICATION: 更新 run_backtest 的函数调用 ---
def run_backtest(params, full_df, ml_models_red, ml_models_blue, backtest_range, prize_rules, cost_per_ticket, feature_columns):
    total_profit = 0
    prize_counts = {key: 0 for key in prize_rules.keys()}
    
    start_index, end_index = backtest_range
    for i in range(start_index, end_index):
        df_history = full_df.iloc[:i]
        actual_draw = full_df.iloc[i]
        
        # 将 feature_columns 传递给评分函数
        red_scores, blue_scores = run_strategy_and_get_scores(df_history, params, ml_models_red, ml_models_blue, feature_columns)
        
        recommended_reds = sorted(red_scores, key=red_scores.get, reverse=True)[:6]
        recommended_blue = sorted(blue_scores, key=blue_scores.get, reverse=True)[0]
        
        red_hits = len(set(recommended_reds) & set(actual_draw['红球']))
        blue_hit = 1 if recommended_blue == actual_draw['蓝球'] else 0
        
        hit_key = (red_hits, blue_hit)
        prize = prize_rules.get(hit_key, 0)
        
        if prize > 0:
            prize_counts[hit_key] += 1
        
        total_profit += (prize - cost_per_ticket)
        
    return total_profit, prize_counts


# --- 4. 主执行与优化框架 ---

if __name__ == '__main__':
    SKIP_OPTIMIZATION = False #False True
    COST_PER_TICKET = 2
    PRIZE_RULES = {
        (6, 1): 5000000, (6, 0): 100000, (5, 1): 3000, (5, 0): 200,
        (4, 1): 200, (4, 0): 10, (3, 1): 10, (2, 1): 5, (1, 1): 5, (0, 1): 5
    }
    PRIZE_NAMES = {
        (6, 1): "一等奖", (6, 0): "二等奖", (5, 1): "三等奖", 
        (5, 0): "四等奖", (4, 1): "四等奖", (4, 0): "五等奖", 
        (3, 1): "五等奖", (2, 1): "六等奖", (1, 1): "六等奖", (0, 1): "六等奖"
    }

    # --- MODIFICATION: 定义全局的特征列表和窗口大小 ---
    WINDOW_SIZE = 5
    FEATURE_COLUMNS = [
        'red_sum', 'red_span', 'odd_count', 'blue_lag1',
        'red_zone_small', 'red_zone_medium', 'red_zone_large',
        'red_big_count', 'red_prime_count', 'red_sum_tail',
        'red_consecutive_groups', 'red_ac_value', 'red_tail_uniques',
        'red_sum_lag1', 'odd_count_lag1',
        'red_sum_ma5', 'odd_count_ma5', 'blue_ma5'
    ]

    print("正在加载数据...")
    full_df = load_and_preprocess_data()
    if full_df is None: exit()
    print("正在进行特征工程...")
    full_df = feature_engineer(full_df)
    
    print("正在预训练机器学习模型...")
    ml_models_red, ml_models_blue = {}, {}
    # --- MODIFICATION: 调整训练数据的起始点，以跳过包含NaN的行 ---
    ml_training_df = full_df.iloc[WINDOW_SIZE:].copy()

    for i in tqdm(range(1, 34), desc="训练红球模型"):
        ml_training_df[f'red_{i}_next'] = ml_training_df['红球'].apply(lambda x: 1 if i in x else 0).shift(-1)
        # 确保所有特征和目标列都没有NaN值
        df_temp = ml_training_df.dropna(subset=FEATURE_COLUMNS + [f'red_{i}_next'])
        # --- MODIFICATION: 使用完整的特征列表 ---
        X, y = df_temp[FEATURE_COLUMNS], df_temp[f'red_{i}_next']
        if not X.empty and not y.empty:
            lgb_clf = lgb.LGBMClassifier(random_state=42, verbose=-1); lgb_clf.fit(X, y)
            ml_models_red[i] = lgb_clf

    for i in tqdm(range(1, 17), desc="训练蓝球模型"):
        ml_training_df[f'blue_{i}_next'] = ml_training_df['蓝球'].apply(lambda x: 1 if x == i else 0).shift(-1)
        df_temp = ml_training_df.dropna(subset=FEATURE_COLUMNS + [f'blue_{i}_next'])
        # --- MODIFICATION: 使用完整的特征列表 ---
        X, y = df_temp[FEATURE_COLUMNS], df_temp[f'blue_{i}_next']
        if not X.empty and not y.empty:
            lgb_clf = lgb.LGBMClassifier(random_state=42, verbose=-1); lgb_clf.fit(X, y)
            ml_models_blue[i] = lgb_clf
        
    best_params, best_profit = {}, 0
    best_prize_counts = {}
    # 回测起始点也应大于WINDOW_SIZE，以确保有足够数据计算特征
    backtest_start_index = max(1000, WINDOW_SIZE + 1)
    backtest_end_index = len(full_df) - 1

    if SKIP_OPTIMIZATION:
        print("\n" + "="*50 + "\n【模式】: 跳过优化，使用预设的最优参数\n" + "="*50)
        best_params = {
            'decay_factor': 0.995, 'weight_freq': 0.3, 'weight_omission': 0.4, 'weight_ml': 0.3,
            'hot_lookback': 15, 'hot_threshold': 3, 'hot_bonus': 1.2,
            'cold_lookback': 20, 'cold_bonus': 1.05, 'repeat_bonus': 1.05,
            'weight_blue_freq': 0.5, 'weight_blue_ml': 0.5
        }
        print("\n正在为预设参数计算历史回测利润和中奖详情...")
        # --- MODIFICATION: 传递 feature_columns 参数 ---
        best_profit, best_prize_counts = run_backtest(
            best_params, full_df, ml_models_red, ml_models_blue, 
            (backtest_start_index, backtest_end_index), 
            PRIZE_RULES, COST_PER_TICKET, FEATURE_COLUMNS
        )
        print("计算完成。")
    else:
        print("\n" + "="*50 + "\n【模式】: 完整优化，并行搜索最优参数\n" + "="*50)
        param_space = {
            'decay_factor': [0.995, 0.999], 'weight_freq': [0.3], 'weight_omission': [0.4], 'weight_ml': [0.3],
            'hot_lookback': [10, 15], 'hot_threshold': [2, 3], 'hot_bonus': [1.1, 1.2],
            'cold_lookback': [20, 30], 'cold_bonus': [1.05, 1.1], 'repeat_bonus': [1.05, 1.15],
            'weight_blue_freq': [0.4, 0.5, 0.6], 'weight_blue_ml': [0.6, 0.5, 0.4]
        }
        keys, values = zip(*param_space.items())
        all_param_combinations = [dict(zip(keys, v)) for v in product(*values) if round(v[list(keys).index('weight_blue_freq')] + v[list(keys).index('weight_blue_ml')], 2) == 1.0]
        print(f"总计需要测试 {len(all_param_combinations)} 种参数组合。")
        
        # --- MODIFICATION: 在并行任务中传递 feature_columns ---
        results = Parallel(n_jobs=-1)(
            delayed(run_backtest)(
                p, full_df, ml_models_red, ml_models_blue, 
                (backtest_start_index, backtest_end_index), 
                PRIZE_RULES, COST_PER_TICKET, FEATURE_COLUMNS
            )
            for p in tqdm(all_param_combinations, desc="优化进度")
        )
        
        best_profit = -float('inf')
        for params, (profit, counts) in zip(all_param_combinations, results):
            if profit > best_profit:
                best_profit, best_params, best_prize_counts = profit, params, counts

    print("\n--- 正在生成最终推荐报告 ---")
    # --- MODIFICATION: 传递 feature_columns 参数 ---
    final_red_scores, final_blue_scores = run_strategy_and_get_scores(full_df, best_params, ml_models_red, ml_models_blue, FEATURE_COLUMNS)
    single_combos, duplex_combo = generate_recommendations(final_red_scores, final_blue_scores, num_single=10)

    report_lines = []
    report_lines.append("="*60)
    report_lines.append("          双色球策略分析与推荐报告 (增强特征版)")
    report_lines.append("="*60)
    
    basis_issue = full_df.iloc[-1]['期号']
    try:
        target_issue = int(basis_issue) + 1
    except ValueError:
        target_issue = f"{basis_issue}_Next"
    report_lines.append("\n--- 0. 报告元数据 ---")
    report_lines.append(f"Data_Basis_Issue: {basis_issue}")
    report_lines.append(f"Prediction_Target_Issue: {target_issue}")
    
    report_lines.append(f"\n报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n--- 1. 策略参数 ---")
    report_lines.append(f"模式: {'使用预设的最优参数 (测试模式)' if SKIP_OPTIMIZATION else '基于历史数据动态寻优'}")
    for key, val in best_params.items():
        report_lines.append(f"  - {key:<20}: {val}")
    report_lines.append(f"\n该策略在历史回测 ({backtest_end_index - backtest_start_index}期) 中的模拟总利润为: {best_profit:.2f} 元")
    report_lines.append("中奖详情如下：")
    aggregated_counts = {name: 0 for name in set(PRIZE_NAMES.values())}
    for (red, blue), count in best_prize_counts.items():
        if count > 0:
            prize_name = PRIZE_NAMES.get((red, blue))
            if prize_name:
                aggregated_counts[prize_name] += count
    if sum(aggregated_counts.values()) == 0:
        report_lines.append("  - 未中任何奖项。")
    else:
        sorted_prize_names = ["一等奖", "二等奖", "三等奖", "四等奖", "五等奖", "六等奖"]
        for prize_name in sorted_prize_names:
            count = aggregated_counts.get(prize_name, 0)
            if count > 0:
                report_lines.append(f"  - {prize_name:<5}: {count} 次")
    report_lines.append("\n--- 2. 推荐组合 ---")
    top_blue_for_single = duplex_combo[1][0] if duplex_combo[1] else "N/A"
    report_lines.append("\n【单式推荐 (10组)】")
    if single_combos:
        for i, combo in enumerate(single_combos, 1):
            report_lines.append(f"  组合 {i:>2}: 红球 {str(combo):<24} 蓝球 [{top_blue_for_single:02d}]")
    else:
        report_lines.append("  - 未能生成足够的单式组合。")

    report_lines.append("\n【7+7 复式推荐 (1组)】")
    report_lines.append(f"  红球: {duplex_combo[0]}")
    report_lines.append(f"  蓝球: {duplex_combo[1]}")
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

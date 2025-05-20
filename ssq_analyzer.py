import pandas as pd
import numpy as np
from collections import Counter, OrderedDict # OrderedDict for prize sorting
import itertools
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

# --- 配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'shuangseqiu.csv')
PROCESSED_CSV_PATH = os.path.join(SCRIPT_DIR, 'shuangseqiu_processed.csv')
WEIGHTS_CONFIG_FILE = os.path.join(SCRIPT_DIR, 'weights_config.json')

RED_BALL_RANGE = range(1, 34)
BLUE_BALL_RANGE = range(1, 17)
RED_ZONES = {'Zone1': (1, 11), 'Zone2': (12, 22), 'Zone3': (23, 33)}

ML_LAG_FEATURES = [1, 3, 5, 10]
BACKTEST_PERIODS_COUNT = 100
OPTIMIZATION_BACKTEST_PERIODS = 30
OPTIMIZATION_TRIALS = 50
RECENT_FREQ_WINDOW = 20

CANDIDATE_POOL_SCORE_THRESHOLDS = {'High': 70, 'Medium': 40}
CANDIDATE_POOL_SEGMENT_NAMES = ['High', 'Medium', 'Low']

DEFAULT_WEIGHTS = {
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
}
CURRENT_WEIGHTS = DEFAULT_WEIGHTS.copy()

SCORE_SEGMENT_BOUNDARIES = [0, 25, 50, 75, 100]
SCORE_SEGMENT_LABELS = [f'{SCORE_SEGMENT_BOUNDARIES[i]+1}-{SCORE_SEGMENT_BOUNDARIES[i+1]}'
                        for i in range(len(SCORE_SEGMENT_BOUNDARIES)-1)]
SCORE_SEGMENT_LABELS[0] = f'{SCORE_SEGMENT_BOUNDARIES[0]}-{SCORE_SEGMENT_BOUNDARIES[1]}'
if len(SCORE_SEGMENT_LABELS) != len(SCORE_SEGMENT_BOUNDARIES) - 1:
     raise ValueError("分数段标签数量与边界数量不匹配，请检查配置。")

LGBM_PARAMS = {'objective': 'binary', 'metric': 'binary_logloss', 'n_estimators': 100, 'learning_rate': 0.04, 'feature_fraction': 0.7, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'lambda_l1': 0.15, 'lambda_l2': 0.15, 'num_leaves': 15, 'min_child_samples': 15, 'verbose': -1, 'n_jobs': 1, 'seed': 42, 'boosting_type': 'gbdt'}
LOGISTIC_REG_PARAMS = {'penalty': 'l2', 'C': 0.1, 'solver': 'lbfgs', 'random_state': 42, 'max_iter': 5000, 'tol': 1e-3}
SVC_PARAMS = {'C': 0.1, 'kernel': 'rbf', 'gamma': 'scale', 'probability': True, 'random_state': 42, 'cache_size': 200, 'max_iter': 25000, 'tol': 1e-3}
XGB_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'n_estimators': 100, 'learning_rate': 0.04, 'max_depth': 3, 'subsample': 0.7, 'colsample_bytree': 0.7, 'gamma': 0.1, 'lambda': 0.15, 'alpha': 0.15, 'seed': 42, 'n_jobs': 1}
MIN_POSITIVE_SAMPLES_FOR_ML = 25

console_formatter = logging.Formatter('%(message)s')
detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ssq_analyzer')
logger.setLevel(logging.DEBUG) # Set base level to DEBUG for logger, handlers control output
logger.propagate = False

# Global console handler, its level will be changed dynamically
global_console_handler = logging.StreamHandler(sys.stdout)
global_console_handler.setFormatter(console_formatter) # Default to simple
logger.addHandler(global_console_handler) # Add it once

def set_console_verbosity(level=logging.INFO, use_simple_formatter=False):
    global_console_handler.setLevel(level)
    if use_simple_formatter:
        global_console_handler.setFormatter(console_formatter)
    else:
        global_console_handler.setFormatter(detailed_formatter)

class SuppressOutput:
    def __init__(self, suppress_stdout=True, capture_stderr=True):
        self.suppress_stdout = suppress_stdout; self.capture_stderr = capture_stderr
        self.old_stdout = None; self.old_stderr = None; self.stderr_io = None
    def __enter__(self):
        if self.suppress_stdout: self.old_stdout = sys.stdout; sys.stdout = open(os.devnull, 'w')
        if self.capture_stderr: self.old_stderr = sys.stderr; self.stderr_io = io.StringIO(); sys.stderr = self.stderr_io
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.capture_stderr and self.old_stderr:
            sys.stderr = self.old_stderr
            captured_stderr_content = self.stderr_io.getvalue()
            if captured_stderr_content.strip(): logger.warning(f"捕获到的标准错误输出:\n{captured_stderr_content.strip()}")
        if self.suppress_stdout and self.old_stdout:
            if sys.stdout and not sys.stdout.closed: sys.stdout.close()
            sys.stdout = self.old_stdout
        return False

def load_weights_from_file(filepath: str, defaults: Dict) -> Dict:
    try:
        with open(filepath, 'r') as f:
            loaded_weights = json.load(f)
        merged_weights = defaults.copy()
        for key in defaults:
            if key in loaded_weights:
                if isinstance(defaults[key], (int, float)) and isinstance(loaded_weights[key], (int, float)):
                    merged_weights[key] = type(defaults[key])(loaded_weights[key])
                elif isinstance(defaults[key], str) and isinstance(loaded_weights[key], str):
                     merged_weights[key] = loaded_weights[key]

        for key_default in defaults:
            if key_default not in merged_weights:
                logger.info(f"权重文件 {filepath} 缺少键 '{key_default}'。将使用该键的默认值。")
                merged_weights[key_default] = defaults[key_default]

        prop_h = merged_weights.get('CANDIDATE_POOL_PROPORTIONS_HIGH', 0.5)
        prop_m = merged_weights.get('CANDIDATE_POOL_PROPORTIONS_MEDIUM', 0.3)
        if not (0 <= prop_h <= 1 and 0 <= prop_m <= 1 and (prop_h + prop_m) <= 1):
            logger.warning(f"CANDIDATE_POOL_PROPORTIONS 无效 (H:{prop_h}, M:{prop_m})。恢复为默认值。")
            merged_weights['CANDIDATE_POOL_PROPORTIONS_HIGH'] = defaults['CANDIDATE_POOL_PROPORTIONS_HIGH']
            merged_weights['CANDIDATE_POOL_PROPORTIONS_MEDIUM'] = defaults['CANDIDATE_POOL_PROPORTIONS_MEDIUM']

        logger.info(f"权重已从 {filepath} 成功加载并合并。")
        return merged_weights
    except FileNotFoundError:
        logger.info(f"权重配置文件 {filepath} 未找到。")
        return defaults.copy()
    except json.JSONDecodeError:
        logger.error(f"权重文件 {filepath} 格式错误。")
        return defaults.copy()
    except Exception as e:
        logger.error(f"加载权重时发生未知错误: {e}。")
        return defaults.copy()

def save_weights_to_file(filepath: str, weights_to_save: Dict):
    try:
        with open(filepath, 'w') as f:
            json.dump(weights_to_save, f, indent=4)
        logger.info(f"权重已成功保存到 {filepath}")
    except Exception as e:
        logger.error(f"保存权重时出错: {e}")

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    try:
        encodings = ['utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                return df
            except UnicodeDecodeError:
                continue
        logger.error(f"无法使用任何尝试的编码打开文件 {file_path}。")
        return None
    except FileNotFoundError: logger.error(f"错误: {file_path} 找不到"); return None
    except pd.errors.EmptyDataError: logger.error(f"错误: {file_path} 为空"); return None
    except Exception as e: logger.error(f"加载 {file_path} 出错: {e}"); return None

def clean_and_structure(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty: return None
    df.dropna(subset=['期号', '红球', '蓝球'], inplace=True)
    if df.empty: return None
    try:
        df['期号'] = pd.to_numeric(df['期号'], errors='coerce').astype('Int64')
        df.dropna(subset=['期号'], inplace=True)
        df['期号'] = df['期号'].astype(int)
        df.sort_values(by='期号', ascending=True, inplace=True); df.reset_index(drop=True, inplace=True)
    except Exception: return None
    if df.empty: return None
    parsed_rows = []
    for _, row in df.iterrows():
        try:
            rs, bv, pv = str(row.get('红球','')), row.get('蓝球'), row.get('期号')
            if not rs or pd.isna(bv) or pd.isna(pv): continue
            bn = int(bv);
            if not (1 <= bn <= 16): continue
            reds_str = rs.split(',')
            if len(reds_str) != 6: continue
            reds_int = sorted([int(x) for x in reds_str if 1 <= int(x) <= 33])
            if len(reds_int) != 6: continue
            rd = {'期号': int(pv)};
            if '日期' in row and pd.notna(row['日期']): rd['日期'] = str(row['日期']).strip()
            for i in range(6):
                rd[f'red{i+1}'] = reds_int[i]
                rd[f'red_pos{i+1}'] = reds_int[i]
            rd['blue'] = bn; parsed_rows.append(rd)
        except Exception: continue
    return pd.DataFrame(parsed_rows).sort_values(by='期号').reset_index(drop=True) if parsed_rows else None

def feature_engineer(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty: return None
    red_cols = [f'red{i+1}' for i in range(6)]
    if not all(c in df.columns for c in red_cols + ['blue', '期号']): return None
    df_fe = df.copy()
    for r_col in red_cols: df_fe[r_col] = pd.to_numeric(df_fe[r_col], errors='coerce')
    df_fe.dropna(subset=red_cols, inplace=True)

    df_fe['red_sum'] = df_fe[red_cols].sum(axis=1)
    df_fe['red_span'] = df_fe[red_cols].max(axis=1) - df_fe[red_cols].min(axis=1)

    if pd.api.types.is_numeric_dtype(df_fe[red_cols].values.dtype):
         df_fe['red_odd_count'] = df_fe[red_cols].apply(lambda r: sum(int(x) % 2 != 0 for x in r), axis=1)
         for zone, (start, end) in RED_ZONES.items():
             df_fe[f'red_{zone}_count'] = df_fe[red_cols].apply(lambda r: sum(start <= int(x) <= end for x in r), axis=1)
         df_fe['current_reds_str'] = df_fe[red_cols].astype(int).astype(str).agg(','.join, axis=1)
         df_fe['prev_reds_str'] = df_fe['current_reds_str'].shift(1)
         df_fe['red_repeat_count'] = df_fe.apply(lambda r: len(set(int(x) for x in r['prev_reds_str'].split(',')) & set(int(x) for x in r['current_reds_str'].split(','))) if pd.notna(r['prev_reds_str']) and pd.notna(r['current_reds_str']) else 0, axis=1)
         df_fe.drop(columns=['current_reds_str', 'prev_reds_str'], inplace=True, errors='ignore')
    else:
        df_fe['red_odd_count'] = np.nan; df_fe['red_repeat_count'] = np.nan
        for zone in RED_ZONES: df_fe[f'red_{zone}_count'] = np.nan

    red_pos_cols = [f'red_pos{i+1}' for i in range(6)]
    if not df_fe.empty and all(c in df_fe.columns for c in red_pos_cols) and pd.api.types.is_numeric_dtype(df_fe[red_pos_cols].values.dtype):
        df_fe['red_consecutive_pairs'] = df_fe.apply(lambda r: sum(1 for i in range(5) if int(r[red_pos_cols[i]]) + 1 == int(r[red_pos_cols[i+1]])), axis=1)
    else: df_fe['red_consecutive_pairs'] = np.nan

    if 'blue' in df_fe.columns and pd.api.types.is_numeric_dtype(df_fe['blue']):
        df_fe['blue'] = pd.to_numeric(df_fe['blue'], errors='coerce').dropna().astype(int)
        df_fe['blue_is_odd'] = df_fe['blue'] % 2 != 0
        df_fe['blue_is_large'] = df_fe['blue'] > 8
        primes = {2, 3, 5, 7, 11, 13}; df_fe['blue_is_prime'] = df_fe['blue'].apply(lambda x: x in primes if pd.notna(x) else False)
    else: df_fe['blue_is_odd'] = np.nan; df_fe['blue_is_large'] = np.nan; df_fe['blue_is_prime'] = np.nan
    return df_fe

def analyze_frequency_omission(df: pd.DataFrame, weights_config: Dict) -> dict:
    if df is None or df.empty: return {}
    red_cols = [f'red{i+1}' for i in range(6)]
    most_recent_idx = len(df) - 1
    if most_recent_idx < 0: return {}

    num_red_cols = [c for c in red_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    num_blue_col = 'blue' if 'blue' in df.columns and pd.api.types.is_numeric_dtype(df['blue']) else None
    if not num_red_cols and not num_blue_col: return {}

    all_reds_flat = df[num_red_cols].values.flatten() if num_red_cols else np.array([])
    red_freq = Counter(all_reds_flat[~np.isnan(all_reds_flat)].astype(int))
    blue_freq = Counter(df[num_blue_col].dropna().astype(int)) if num_blue_col else Counter()

    current_omission = {}
    max_historical_omission_red = {num: 0 for num in RED_BALL_RANGE}
    recent_N_freq_red = {num: 0 for num in RED_BALL_RANGE}

    if num_red_cols:
        for num in RED_BALL_RANGE:
            appearances = (df[num_red_cols] == num).any(axis=1)
            app_indices = df.index[appearances]

            if not app_indices.empty:
                current_omission[num] = most_recent_idx - app_indices.max()
                max_o = app_indices[0]
                for i in range(len(app_indices) - 1):
                    max_o = max(max_o, app_indices[i+1] - app_indices[i] - 1)
                max_o = max(max_o, most_recent_idx - app_indices.max())
                max_historical_omission_red[num] = max_o
            else:
                current_omission[num] = len(df)
                max_historical_omission_red[num] = len(df)

        recent_df_slice = df.tail(RECENT_FREQ_WINDOW)
        if not recent_df_slice.empty:
            recent_reds_flat = recent_df_slice[num_red_cols].values.flatten()
            recent_freq_counts = Counter(recent_reds_flat[~np.isnan(recent_reds_flat)].astype(int))
            for num in RED_BALL_RANGE:
                recent_N_freq_red[num] = recent_freq_counts.get(num, 0)

    if num_blue_col:
        for num in BLUE_BALL_RANGE:
            app_indices = df.index[df[num_blue_col] == num]
            latest_idx = app_indices.max() if not app_indices.empty else -1
            current_omission[f'blue_{num}'] = len(df) if latest_idx == -1 else most_recent_idx - latest_idx

    avg_interval = {num: len(df) / (red_freq.get(num, 0) + 1e-9) for num in RED_BALL_RANGE}
    for num in BLUE_BALL_RANGE: avg_interval[f'blue_{num}'] = len(df) / (blue_freq.get(num, 0) + 1e-9)

    red_items = sorted(red_freq.items(), key=lambda item: item[1], reverse=True)
    blue_items = sorted(blue_freq.items(), key=lambda item: item[1], reverse=True)
    hot_reds = [n for n, _ in red_items[:max(0, int(len(RED_BALL_RANGE) * 0.2))]]
    cold_reds = [n for n, _ in red_items[min(len(red_items)-1, int(len(RED_BALL_RANGE) * 0.8)):] if n not in hot_reds]
    hot_blues = [n for n, _ in blue_items[:max(0, int(len(BLUE_BALL_RANGE) * 0.3))]]
    cold_blues = [n for n, _ in blue_items[min(len(blue_items)-1, int(len(BLUE_BALL_RANGE) * 0.7)):] if n not in hot_blues]

    return {'red_freq': red_freq, 'blue_freq': blue_freq, 'current_omission': current_omission,
            'average_interval': avg_interval, 'hot_reds': hot_reds, 'cold_reds': cold_reds,
            'hot_blues': hot_blues, 'cold_blues': cold_blues,
            'max_historical_omission_red': max_historical_omission_red,
            'recent_N_freq_red': recent_N_freq_red}

def analyze_patterns(df: pd.DataFrame, weights_config: Dict) -> dict:
    if df is None or df.empty: return {}
    res = {}
    def safe_mode(series): return int(series.mode().iloc[0]) if not series.empty and not series.mode().empty else None

    for col, name in [('red_sum', 'sum'), ('red_span', 'span')]:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and not df[col].empty:
            res[f'{name}_stats'] = df[col].describe().to_dict()
            res[f'most_common_{name}'] = safe_mode(df[col])
    if 'red_odd_count' in df.columns and pd.api.types.is_numeric_dtype(df['red_odd_count']) and not df['red_odd_count'].empty:
        res['most_common_odd_even_count'] = safe_mode(df['red_odd_count'].dropna())
    zone_cols = [f'red_{zone}_count' for zone in RED_ZONES.keys()]
    if all(c in df.columns and pd.api.types.is_numeric_dtype(df[c]) for c in zone_cols) and not df.empty:
        zc_df = df[zone_cols].dropna().astype(int)
        if not zc_df.empty:
            dist_counts = zc_df.apply(tuple, axis=1).value_counts()
            res['most_common_zone_distribution'] = dist_counts.index[0] if not dist_counts.empty else None
    for col_name, data_key in [('blue_is_odd', 'blue_odd_counts'), ('blue_is_large', 'blue_large_counts')]:
        if col_name in df.columns and not df[col_name].dropna().empty:
            counts = df[col_name].dropna().astype(bool).value_counts()
            res[data_key] = {bool(k): int(v) for k, v in counts.items()}
    return res

def analyze_associations(df: pd.DataFrame, weights_config: Dict) -> pd.DataFrame:
    min_s = weights_config.get('ARM_MIN_SUPPORT', 0.008) # Use .get for safety if key missing
    min_c = weights_config.get('ARM_MIN_CONFIDENCE', 0.35)
    min_l = weights_config.get('ARM_MIN_LIFT', 1.1)

    if df is None or df.empty or len(df) < 2: return pd.DataFrame()
    red_cols = [f'red{i+1}' for i in range(6)]
    if not all(c in df.columns and pd.api.types.is_numeric_dtype(df[c]) for c in red_cols): return pd.DataFrame()
    tx_df = df.dropna(subset=red_cols).copy()
    if tx_df.empty: return pd.DataFrame()
    try:
        tx_df[red_cols] = tx_df[red_cols].astype(int)
        txs = tx_df[red_cols].astype(str).values.tolist()
    except ValueError: return pd.DataFrame()
    if not txs: return pd.DataFrame()
    te = TransactionEncoder();
    try:
        te_ary = te.fit_transform(txs)
    except Exception: return pd.DataFrame()
    df_oh = pd.DataFrame(te_ary, columns=te.columns_)
    if df_oh.empty: return pd.DataFrame()
    try:
        actual_min_support = max(2/len(df_oh) if len(df_oh)>0 else min_s, min_s)
        f_items = apriori(df_oh, min_support=actual_min_support, use_colnames=True)
        if f_items.empty: return pd.DataFrame()
        rules = association_rules(f_items, metric="lift", min_threshold=min_l)
        if 'confidence' in rules.columns and isinstance(rules['confidence'], pd.Series):
            return rules[rules['confidence'] >= min_c].sort_values(by='lift', ascending=False)
        else:
            logger.debug("ARM: 'confidence' column issue in rules dataframe during filtering.")
            return rules.sort_values(by='lift', ascending=False) if 'lift' in rules.columns else pd.DataFrame()
    except Exception as e_apriori:
        logger.debug(f"Apriori/AssociationRules failed: {e_apriori}")
        return pd.DataFrame()

def get_score_segment(score: float, boundaries: List[int], labels: List[str]) -> str:
    if score is None or pd.isna(score): return "未知"
    tolerance = 1e-9
    if score < boundaries[0] - tolerance: return labels[0] if labels else "未知"
    if score > boundaries[-1] + tolerance: return labels[-1] if labels else "未知"
    for i in range(len(boundaries) - 1):
        if i == 0:
             if boundaries[i] <= score <= boundaries[i+1]: return labels[i]
        elif boundaries[i] < score <= boundaries[i+1]: return labels[i]
    return "未知"

def analyze_winning_red_ball_score_segments(df: pd.DataFrame, red_ball_scores: dict, score_boundaries: List[int], score_labels: List[str]) -> Tuple[Dict[str, int], Dict[str, float]]:
    seg_counts = {label: 0 for label in score_labels}; total_win_reds = 0
    red_cols = [f'red{i+1}' for i in range(6)]
    if df is None or df.empty or not red_ball_scores or not all(c in df.columns for c in red_cols):
        return seg_counts, {label: 0.0 for label in score_labels}

    for _, row in df.iterrows():
        win_reds = []
        valid_row = True
        for c in red_cols:
            val = row.get(c)
            if pd.isna(val): valid_row = False; break
            try:
                num_val = int(float(val))
                if num_val not in RED_BALL_RANGE: valid_row = False; break
                win_reds.append(num_val)
            except (ValueError, TypeError): valid_row = False; break
        if not valid_row or len(win_reds) != 6: continue

        for ball in win_reds:
            score = red_ball_scores.get(ball)
            if score is not None and pd.notna(score) and isinstance(score, (int, float)):
                segment = get_score_segment(score, score_boundaries, score_labels)
                if segment in seg_counts and segment != "未知":
                    seg_counts[segment] += 1; total_win_reds += 1

    seg_pcts = {seg: (cnt / total_win_reds) * 100 if total_win_reds > 0 else 0.0 for seg, cnt in seg_counts.items()}
    return seg_counts, seg_pcts

def create_lagged_features(df: pd.DataFrame, lags: List[int]) -> Optional[pd.DataFrame]:
    if df is None or df.empty or not lags: return None
    base_cols_candidates = ['red_sum', 'red_span', 'red_odd_count', 'red_consecutive_pairs', 'red_repeat_count'] + \
                           [f'red_{zone}_count' for zone in RED_ZONES.keys()] + \
                           ['blue', 'blue_is_odd', 'blue_is_large', 'blue_is_prime']
    df_temp = df.copy()
    existing_lag_cols = []
    for col in base_cols_candidates:
        if col in df_temp.columns:
            if pd.api.types.is_bool_dtype(df_temp[col].dtype):
                df_temp[col] = df_temp[col].astype(int)
                existing_lag_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df_temp[col].dtype):
                existing_lag_cols.append(col)
            else:
                try:
                    df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
                    if pd.api.types.is_numeric_dtype(df_temp[col].dtype): existing_lag_cols.append(col)
                except Exception: pass
    if not existing_lag_cols: return None
    df_lagged = df_temp[existing_lag_cols].copy()
    for lag_val in lags:
        if lag_val > 0:
            for col in existing_lag_cols: df_lagged[f'{col}_lag{lag_val}'] = df_lagged[col].shift(lag_val)
    df_lagged.dropna(inplace=True)
    if df_lagged.empty: return None
    feature_cols = [col for col in df_lagged.columns if any(f'_lag{lag_val}' in col for lag_val in lags)]
    return df_lagged[feature_cols] if feature_cols else None

def train_single_model(model_type, ball_type_str, ball_number, X, y, params, min_pos_samples,
                       lgbm_ref, svc_ref, scaler_ref, pipe_ref, logreg_ref, xgb_ref):
    if y.sum() < min_pos_samples or len(y.unique()) < 2: return None, None
    model_key = f'{model_type}_{ball_number}'
    model_params = params.copy()

    positive_count = y.sum()
    negative_count = len(y) - positive_count
    scale_pos_weight_val = negative_count / (positive_count + 1e-9) if positive_count > 0 else 1.0
    class_weight_val = 'balanced' if positive_count > 0 and negative_count > 0 else None

    model = None
    try:
        if model_type == 'lgbm':
            model_params['scale_pos_weight'] = scale_pos_weight_val
            model = lgbm_ref(**model_params)
            model.fit(X, y)
        elif model_type == 'xgb':
            model_params['scale_pos_weight'] = scale_pos_weight_val
            model = xgb_ref(**model_params)
            model.fit(X, y)
        elif model_type == 'logreg':
            if class_weight_val: model_params['class_weight'] = class_weight_val
            model_params.pop('scale_pos_weight', None)
            model = pipe_ref([('scaler', scaler_ref()), ('logreg', logreg_ref(**model_params))])
            model.fit(X, y)
        elif model_type == 'svc':
            if class_weight_val: model_params['class_weight'] = class_weight_val
            model_params.pop('scale_pos_weight', None)
            svc_actual_params = model_params.copy()
            svc_actual_params['probability'] = True
            model = pipe_ref([('scaler', scaler_ref()), ('svc', svc_ref(**svc_actual_params))])
            model.fit(X, y)
            svc_estimator = model.named_steps.get('svc')
            if not (svc_estimator and hasattr(svc_estimator, 'probability') and svc_estimator.probability):
                logger.debug(f"SVC for {ball_type_str} {ball_number} did not enable probability correctly.")
                model = None
        return model, model_key
    except Exception as e_train:
        logger.debug(f"Training {model_type} for {ball_type_str} {ball_number} failed: {e_train}")
        return None, None

def train_prediction_models(df_train_raw: pd.DataFrame, ml_lags_list: List[int], weights_config: Dict) -> Optional[dict]:
    X = create_lagged_features(df_train_raw.copy(), ml_lags_list)
    if X is None or X.empty: logger.warning("ML Train: Lagged features are empty."); return None

    target_df = df_train_raw.loc[X.index].copy()
    if target_df.empty: logger.warning("ML Train: Target DF is empty."); return None

    red_cols = [f'red{i+1}' for i in range(6)]
    if not all(c in target_df.columns for c in red_cols + ['blue']):
        logger.error("ML Train: Missing ball columns in target_df."); return None
    try:
        for col in red_cols + ['blue']: target_df[col] = pd.to_numeric(target_df[col], errors='coerce').astype(int)
    except (ValueError, TypeError): logger.error("ML Train: Failed to convert ball columns to int."); return None

    trained_models = {'red': {}, 'blue': {}, 'feature_cols': X.columns.tolist()}
    min_pos = MIN_POSITIVE_SAMPLES_FOR_ML

    futures_map = {}
    num_cpus = os.cpu_count()
    max_workers = num_cpus if num_cpus and num_cpus > 1 else 1 # Ensure at least 1 worker

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for ball_num in RED_BALL_RANGE:
            y_red = target_df[red_cols].apply(lambda row: ball_num in row.values, axis=1).astype(int)
            for mt, mp in [('lgbm', LGBM_PARAMS), ('xgb', XGB_PARAMS), ('logreg', LOGISTIC_REG_PARAMS), ('svc', SVC_PARAMS)]:
                f = executor.submit(train_single_model, mt, '红', ball_num, X, y_red, mp, min_pos,
                                    LGBMClassifier, SVC, StandardScaler, Pipeline, LogisticRegression, xgb.XGBClassifier)
                futures_map[f] = ('red', mt, ball_num)
        for ball_num in BLUE_BALL_RANGE:
            y_blue = (target_df['blue'] == ball_num).astype(int)
            for mt, mp in [('lgbm', LGBM_PARAMS), ('xgb', XGB_PARAMS), ('logreg', LOGISTIC_REG_PARAMS), ('svc', SVC_PARAMS)]:
                f = executor.submit(train_single_model, mt, '蓝', ball_num, X, y_blue, mp, min_pos,
                                    LGBMClassifier, SVC, StandardScaler, Pipeline, LogisticRegression, xgb.XGBClassifier)
                futures_map[f] = ('blue', mt, ball_num)

    models_trained_count = 0
    for future in concurrent.futures.as_completed(futures_map):
        ball_type_str, model_type, ball_number = futures_map[future]
        try:
            model, model_key = future.result()
            if model and model_key:
                trained_models[ball_type_str][model_key] = model
                models_trained_count +=1
        except Exception as e_future:
            logger.warning(f"Exception retrieving result for {ball_type_str} {ball_number} {model_type}: {e_future}")

    logger.debug(f"ML模型训练完成。成功训练 {models_trained_count} 个模型。")
    return trained_models if models_trained_count > 0 else None

def predict_next_draw_probabilities(df_historical: pd.DataFrame, trained_models: Optional[dict], ml_lags_list: List[int], weights_config: Dict) -> Dict:
    probs = {'red': {}, 'blue': {}}
    if not trained_models or df_historical is None or df_historical.empty: return probs

    feat_cols = trained_models.get('feature_cols')
    if not feat_cols: logger.warning("ML Predict: No feature_cols in trained_models."); return probs

    max_hist_lag = max(ml_lags_list) if ml_lags_list else 0
    if len(df_historical) < max_hist_lag + 1:
        logger.warning(f"ML Predict: Not enough historical data ({len(df_historical)}) for lag ({max_hist_lag})."); return probs

    predict_X = create_lagged_features(df_historical.tail(max_hist_lag + 1).copy(), ml_lags_list)
    if predict_X is None or predict_X.empty:
        logger.warning("ML Predict: create_lagged_features returned empty for prediction."); return probs
    if len(predict_X) != 1:
        logger.error(f"ML Predict: Expected 1 row for prediction features, got {len(predict_X)}."); return probs

    try:
        predict_X = predict_X.reindex(columns=feat_cols, fill_value=0)
        for col in predict_X.columns: predict_X[col] = pd.to_numeric(predict_X[col], errors='coerce')
        predict_X.fillna(0, inplace=True)
        if predict_X.isnull().values.any(): logger.error("ML Predict: NaN in prediction features after processing."); return probs
    except Exception as e_pred_preprocess:
        logger.error(f"ML Predict: Error preprocessing prediction features: {e_pred_preprocess}."); return probs

    for ball_type_key, ball_val_range, models_sub_dict in [('red', RED_BALL_RANGE, trained_models.get('red', {})),
                                                           ('blue', BLUE_BALL_RANGE, trained_models.get('blue', {}))]:
        if not models_sub_dict: continue
        for ball_val in ball_val_range:
            ball_preds = []
            for model_variant in ['lgbm', 'xgb', 'logreg', 'svc']:
                model_instance = models_sub_dict.get(f'{model_variant}_{ball_val}')
                if model_instance and hasattr(model_instance, 'predict_proba'):
                    try:
                        proba = model_instance.predict_proba(predict_X)[0][1]
                        ball_preds.append(proba)
                    except Exception as e_proba:
                        logger.debug(f"ML Predict: Failed predict_proba for {ball_type_key} {ball_val} {model_variant}: {e_proba}")
            if ball_preds: probs[ball_type_key][ball_val] = np.mean(ball_preds)
    return probs

def calculate_scores(freq_omission_data: dict, pattern_analysis_data: dict, predicted_probabilities: dict, weights_config: Dict) -> dict:
    r_scores, b_scores = {}, {}
    r_freq = freq_omission_data.get('red_freq', {}); b_freq = freq_omission_data.get('blue_freq', {})
    omission = freq_omission_data.get('current_omission', {}); avg_int = freq_omission_data.get('average_interval', {})

    max_hist_omission_r = freq_omission_data.get('max_historical_omission_red', {})
    recent_N_freq_r = freq_omission_data.get('recent_N_freq_red', {})

    r_freq_series = pd.Series(r_freq).reindex(list(RED_BALL_RANGE), fill_value=0)
    r_freq_rank = r_freq_series.rank(method='min', ascending=False)
    b_freq_series = pd.Series(b_freq).reindex(list(BLUE_BALL_RANGE), fill_value=0)
    b_freq_rank = b_freq_series.rank(method='min', ascending=False)

    r_pred_probs = predicted_probabilities.get('red', {}); b_pred_probs = predicted_probabilities.get('blue', {})
    max_r_rank, max_b_rank = len(RED_BALL_RANGE), len(BLUE_BALL_RANGE)

    recent_freq_values = [v for v in recent_N_freq_r.values() if v is not None]
    min_rec_freq, max_rec_freq = min(recent_freq_values) if recent_freq_values else 0, max(recent_freq_values) if recent_freq_values else 0

    for num in RED_BALL_RANGE:
        freq_s = max(0, (max_r_rank - (r_freq_rank.get(num, max_r_rank+1)-1))/max_r_rank * weights_config['FREQ_SCORE_WEIGHT'])
        dev = omission.get(num, max_r_rank*2) - avg_int.get(num, max_r_rank*2)
        omit_s = max(0, weights_config['OMISSION_SCORE_WEIGHT'] * np.exp(-0.005 * dev**2))

        max_o = max_hist_omission_r.get(num, 0)
        cur_o = omission.get(num, 0)
        max_omit_ratio_s = 0
        if max_o > 0:
            ratio_o = cur_o / max_o
            max_omit_ratio_s = max(0, min(1.0, ratio_o)) * weights_config['MAX_OMISSION_RATIO_SCORE_WEIGHT_RED']
            if ratio_o > 1.2: max_omit_ratio_s *= 1.2
            if ratio_o < 0.2: max_omit_ratio_s *= 0.5
        else:
            max_omit_ratio_s = weights_config['MAX_OMISSION_RATIO_SCORE_WEIGHT_RED'] if cur_o > 0 else 0

        rec_f = recent_N_freq_r.get(num, 0)
        norm_rec_f_score = 0
        if max_rec_freq > min_rec_freq:
            norm_rec_f_score = (rec_f - min_rec_freq) / (max_rec_freq - min_rec_freq)
        elif max_rec_freq > 0 :
             norm_rec_f_score = 0.5 if rec_f > 0 else 0
        recent_freq_s = max(0, norm_rec_f_score * weights_config['RECENT_FREQ_SCORE_WEIGHT_RED'])

        ml_s = max(0, r_pred_probs.get(num, 0.0) * weights_config['ML_PROB_SCORE_WEIGHT_RED'])

        r_scores[num] = freq_s + omit_s + ml_s + max_omit_ratio_s + recent_freq_s

    for num in BLUE_BALL_RANGE:
        freq_s = max(0, (max_b_rank - (b_freq_rank.get(num, max_b_rank+1)-1))/max_b_rank * weights_config['BLUE_FREQ_SCORE_WEIGHT'])
        dev = omission.get(f'blue_{num}', max_b_rank*2) - avg_int.get(f'blue_{num}', max_b_rank*2)
        omit_s = max(0, weights_config['BLUE_OMISSION_SCORE_WEIGHT'] * np.exp(-0.01 * dev**2))
        ml_s = max(0, b_pred_probs.get(num, 0.0) * weights_config['ML_PROB_SCORE_WEIGHT_BLUE'])
        b_scores[num] = freq_s + omit_s + ml_s

    all_s_vals = [s for s in list(r_scores.values()) + list(b_scores.values()) if np.isfinite(s)]
    if all_s_vals:
        min_s_val, max_s_val = min(all_s_vals), max(all_s_vals)
        if (max_s_val - min_s_val) > 1e-9:
            r_scores = {n: max(0,min(100,(s-min_s_val)/(max_s_val-min_s_val)*100)) if np.isfinite(s) else 0 for n,s in r_scores.items()}
            b_scores = {n: max(0,min(100,(s-min_s_val)/(max_s_val-min_s_val)*100)) if np.isfinite(s) else 0 for n,s in b_scores.items()}
        else:
            r_scores = {n:50.0 for n in RED_BALL_RANGE}; b_scores = {n:50.0 for n in BLUE_BALL_RANGE}
    else:
        r_scores = {n:0.0 for n in RED_BALL_RANGE}; b_scores = {n:0.0 for n in BLUE_BALL_RANGE}
    return {'red_scores': r_scores, 'blue_scores': b_scores}

def generate_combinations(scores_data: dict, pattern_analysis_data: dict, association_rules_df: pd.DataFrame,
                          winning_segment_percentages: Dict[str, float], weights_config: Dict) -> tuple[List[Dict], list[str]]:
    num_combinations_to_generate = weights_config.get('NUM_COMBINATIONS_TO_GENERATE', 10)
    target_red_pool_size = weights_config.get('TOP_N_RED_FOR_CANDIDATE', 18)
    top_n_blue = weights_config.get('TOP_N_BLUE_FOR_CANDIDATE', 8)

    min_different_reds = weights_config.get('DIVERSITY_MIN_DIFFERENT_REDS', 3)
    max_common_reds_allowed = 6 - min_different_reds
    diversity_max_attempts = weights_config.get('DIVERSITY_SELECTION_MAX_ATTEMPTS', 20)

    prop_h = weights_config.get('CANDIDATE_POOL_PROPORTIONS_HIGH', 0.5)
    prop_m = weights_config.get('CANDIDATE_POOL_PROPORTIONS_MEDIUM', 0.3)
    prop_l = max(0, 1.0 - prop_h - prop_m)
    segment_proportions = [prop_h, prop_m, prop_l]
    min_per_segment = weights_config.get('CANDIDATE_POOL_MIN_PER_SEGMENT', 2)

    r_scores = scores_data.get('red_scores', {})
    b_scores = scores_data.get('blue_scores', {})

    r_cand_pool = []
    if r_scores:
        segmented_balls_dict = {name: [] for name in CANDIDATE_POOL_SEGMENT_NAMES}
        for ball_num, score_val in r_scores.items():
            if score_val > CANDIDATE_POOL_SCORE_THRESHOLDS['High']:
                segmented_balls_dict['High'].append(ball_num)
            elif score_val > CANDIDATE_POOL_SCORE_THRESHOLDS['Medium']:
                segmented_balls_dict['Medium'].append(ball_num)
            else:
                segmented_balls_dict['Low'].append(ball_num)

        for seg_name in CANDIDATE_POOL_SEGMENT_NAMES:
            segment_balls_with_scores = {b: r_scores.get(b, 0) for b in segmented_balls_dict[seg_name]}
            segmented_balls_dict[seg_name] = [b for b, _ in sorted(segment_balls_with_scores.items(), key=lambda x: x[1], reverse=True)]

        temp_pool_set = set()
        num_to_pick_segments = [
            max(min_per_segment, int(round(prop * target_red_pool_size))) for prop in segment_proportions
        ]

        for i, seg_name in enumerate(CANDIDATE_POOL_SEGMENT_NAMES):
            balls_from_segment = segmented_balls_dict[seg_name]
            num_to_add = num_to_pick_segments[i]
            added_count = 0
            for ball in balls_from_segment:
                if len(temp_pool_set) >= target_red_pool_size: break
                if ball not in temp_pool_set and added_count < num_to_add:
                    temp_pool_set.add(ball)
                    added_count += 1
            if len(temp_pool_set) >= target_red_pool_size: break
        
        r_cand_pool = list(temp_pool_set)

        if len(r_cand_pool) < target_red_pool_size:
            all_sorted_reds_overall = [n for n, _ in sorted(r_scores.items(), key=lambda item: item[1], reverse=True)]
            for ball in all_sorted_reds_overall:
                if len(r_cand_pool) >= target_red_pool_size: break
                if ball not in r_cand_pool:
                    r_cand_pool.append(ball)
    
    if len(r_cand_pool) < 6:
        logger.debug(f"Red candidate pool has only {len(r_cand_pool)} balls after segmented selection. Expanding.")
        current_pool_set = set(r_cand_pool)
        all_sorted_reds_overall = [n for n, _ in sorted(r_scores.items(), key=lambda item: item[1], reverse=True)]
        for ball in all_sorted_reds_overall:
            if len(r_cand_pool) >= 6: break 
            if ball not in current_pool_set:
                r_cand_pool.append(ball)
                current_pool_set.add(ball)
        if len(r_cand_pool) < 6:
            r_cand_pool.extend(b for b in RED_BALL_RANGE if b not in r_cand_pool)
            r_cand_pool = list(set(r_cand_pool))[:6] 

    b_cand_pool = [n for n, _ in sorted(b_scores.items(), key=lambda item: item[1], reverse=True)[:top_n_blue]]
    if len(b_cand_pool) < 1: b_cand_pool = list(BLUE_BALL_RANGE)
    
    large_pool_size = max(num_combinations_to_generate * 100, 200) 
    max_attempts_pool = large_pool_size * 20 

    win_seg_pcts = winning_segment_percentages
    valid_seg_pcts = win_seg_pcts and all(lbl in win_seg_pcts for lbl in SCORE_SEGMENT_LABELS) and sum(win_seg_pcts.values()) > 1e-6
    seg_factors = {lbl:1.0 for lbl in SCORE_SEGMENT_LABELS}
    if valid_seg_pcts:
        seg_factors_temp = {lbl:(pct/100.0)+0.05 for lbl,pct in win_seg_pcts.items()}
        tot_factor_sum = sum(seg_factors_temp.values())
        if tot_factor_sum > 1e-9: seg_factors = {lbl:f_val/tot_factor_sum for lbl,f_val in seg_factors_temp.items()}

    r_probs_raw = {}
    if not r_cand_pool: 
        logger.error("Critical: r_cand_pool is empty before probability calculation. Defaulting to range.")
        r_cand_pool = random.sample(list(RED_BALL_RANGE), k=min(target_red_pool_size, len(RED_BALL_RANGE)))

    for n in r_cand_pool:
        seg = get_score_segment(r_scores.get(n,0), SCORE_SEGMENT_BOUNDARIES, SCORE_SEGMENT_LABELS)
        r_probs_raw[n] = (r_scores.get(n,0)+1.0) * seg_factors.get(seg, 1.0/len(seg_factors) if seg_factors else 1.0)

    r_cand_pool_for_probs = [ball for ball in r_cand_pool if ball in r_probs_raw and r_probs_raw[ball] > 0] 
    
    if not r_cand_pool_for_probs: 
        logger.debug("No red balls with positive raw probability. Using uniform from r_cand_pool or range.")
        r_cand_pool_for_probs = r_cand_pool if r_cand_pool else random.sample(list(RED_BALL_RANGE), k=6)
        r_probs_arr = np.ones(len(r_cand_pool_for_probs)) / len(r_cand_pool_for_probs) if r_cand_pool_for_probs else np.array([])
    else:
        r_probs_arr = np.array([r_probs_raw.get(n,0) for n in r_cand_pool_for_probs])
        tot_r_prob_raw = np.sum(r_probs_arr) 
        if tot_r_prob_raw > 1e-9 :
            r_probs_arr = r_probs_arr / tot_r_prob_raw
            if len(r_probs_arr) > 1 : r_probs_arr[-1]=max(0,1.0-np.sum(r_probs_arr[:-1]))
            elif len(r_probs_arr) == 1: r_probs_arr[0] = 1.0
        else: 
            r_probs_arr = np.ones(len(r_cand_pool_for_probs))/len(r_cand_pool_for_probs)

    b_weights_arr = np.array([b_scores.get(n,0)+1.0 for n in b_cand_pool])
    b_probs_arr = np.zeros(len(b_cand_pool)) 
    if not b_cand_pool: 
        b_cand_pool = random.sample(list(BLUE_BALL_RANGE), k=min(top_n_blue, len(BLUE_BALL_RANGE)))
        if not b_cand_pool: b_cand_pool = [1] 
        b_weights_arr = np.array([b_scores.get(n,0)+1.0 for n in b_cand_pool])

    if np.sum(b_weights_arr) > 1e-9 and len(b_cand_pool) > 0:
        b_probs_arr = b_weights_arr / np.sum(b_weights_arr)
        if len(b_probs_arr) > 1: b_probs_arr[-1] = max(0, 1.0 - np.sum(b_probs_arr[:-1]))
        elif len(b_probs_arr) == 1: b_probs_arr[0] = 1.0
    elif len(b_cand_pool) > 0: b_probs_arr = np.ones(len(b_cand_pool)) / len(b_cand_pool)
    else: 
        b_probs_arr = np.array([1.0])
        b_cand_pool = [1]

    sample_size_red = 6
    replace_red_sampling = False
    use_fallback_sampling_flag = False 

    if len(r_cand_pool_for_probs) < sample_size_red:
        logger.debug(f"Red candidate pool for probability sampling ({len(r_cand_pool_for_probs)}) is smaller than sample size ({sample_size_red}).")
        if len(r_cand_pool_for_probs) == 0: 
             use_fallback_sampling_flag = True 
        else: 
            sample_size_red = len(r_cand_pool_for_probs) 
            replace_red_sampling = False 
    
    if not use_fallback_sampling_flag:
        use_fallback_sampling_flag = not (len(b_cand_pool)>=1) or \
                                (len(r_probs_arr) != len(r_cand_pool_for_probs) or not (np.isclose(np.sum(r_probs_arr),1.0) if len(r_probs_arr)>0 else True)) or \
                                (len(b_probs_arr) != len(b_cand_pool) or not (np.isclose(np.sum(b_probs_arr),1.0) if len(b_probs_arr)>0 else True)) or \
                                (len(r_probs_arr) > 0 and np.any(r_probs_arr < 0)) or \
                                (len(b_probs_arr) > 0 and np.any(b_probs_arr < 0))

    if use_fallback_sampling_flag: logger.debug("Using fallback (random) sampling in generate_combinations.")

    gen_pool = []
    attempts = 0
    while len(gen_pool) < large_pool_size and attempts < max_attempts_pool:
        attempts +=1
        try:
            if use_fallback_sampling_flag:
                safe_r_pool = r_cand_pool if len(r_cand_pool) >= 6 else list(RED_BALL_RANGE)
                r_balls_s = sorted(random.sample(safe_r_pool, 6))
                safe_b_pool = b_cand_pool if len(b_cand_pool) >=1 else list(BLUE_BALL_RANGE)
                b_ball_s = random.choice(safe_b_pool)
            else:
                r_balls_s = sorted(np.random.choice(r_cand_pool_for_probs, size=sample_size_red, replace=replace_red_sampling, p=r_probs_arr).tolist())
                if len(r_balls_s) < 6:
                    remaining_needed = 6 - len(r_balls_s)
                    fill_pool_candidates = [b for b in r_cand_pool if b not in r_balls_s] 
                    if len(fill_pool_candidates) < remaining_needed: 
                        fill_pool_candidates.extend([b for b in RED_BALL_RANGE if b not in r_balls_s and b not in fill_pool_candidates])
                    
                    if len(fill_pool_candidates) >= remaining_needed:
                         r_balls_s.extend(random.sample(fill_pool_candidates, remaining_needed))
                         r_balls_s = sorted(list(set(r_balls_s))) 
                    else: 
                        logger.debug(f"Could not form 6 red balls for combo, got {len(r_balls_s)}. Skipping attempt.")
                        continue
                if len(r_balls_s) != 6 : 
                    logger.debug(f"Skipping combo due to red ball count mismatch after fill: {len(r_balls_s)}")
                    continue
                b_ball_s = np.random.choice(b_cand_pool, size=1, p=b_probs_arr).tolist()[0]

            combo = {'red': r_balls_s, 'blue': b_ball_s}
            if len(set(combo['red'])) == 6 and combo['blue'] is not None:
                 if combo not in gen_pool: gen_pool.append(combo)
            else:
                 logger.debug(f"Invalid combo generated and skipped: {combo}")

        except ValueError as e_val:
            use_fallback_sampling_flag = True 
            if attempts <= 5: logger.debug(f"Probabilistic sampling failed in generate_combinations ({e_val}), switching to fallback for this run.")
        except Exception as e_gen:
            logger.debug(f"Exception during combination generation attempt: {e_gen}")
            continue
            
    if not gen_pool: return [], ["推荐组合:", "无法生成推荐组合。"]
    
    scored_combos = []
    patt_data = pattern_analysis_data
    hist_odd_cnt = patt_data.get('most_common_odd_even_count')
    hist_zone_dist = patt_data.get('most_common_zone_distribution')
    blue_l_counts = patt_data.get('blue_large_counts',{})
    hist_blue_large = blue_l_counts.get(True,0) > blue_l_counts.get(False,0) if blue_l_counts else None
    blue_o_counts = patt_data.get('blue_odd_counts',{})
    hist_blue_odd = blue_o_counts.get(True,0) > blue_o_counts.get(False,0) if blue_o_counts else None

    arm_rules_processed = pd.DataFrame()
    if association_rules_df is not None and not association_rules_df.empty:
        arm_rules_processed = association_rules_df.copy() 
        if not arm_rules_processed.empty:
            try: 
                arm_rules_processed['antecedents_set'] = arm_rules_processed['antecedents'].apply(lambda x: set(map(int, x)))
                arm_rules_processed['consequents_set'] = arm_rules_processed['consequents'].apply(lambda x: set(map(int, x)))
            except (TypeError, ValueError) as e_arm_conv:
                logger.warning(f"Error converting ARM rule items to int sets ({e_arm_conv}). ARM bonus may not be applied correctly.")
                arm_rules_processed = pd.DataFrame() 

    for combo_item in gen_pool:
        r_list, b_val = combo_item['red'], combo_item['blue']
        base_s = sum(r_scores.get(ball_num,0) for ball_num in r_list) + b_scores.get(b_val,0)
        bonus_s = 0
        if hist_odd_cnt is not None and sum(x%2!=0 for x in r_list)==hist_odd_cnt: bonus_s += weights_config['COMBINATION_ODD_COUNT_MATCH_BONUS']
        if hist_blue_odd is not None and (b_val%2!=0)==hist_blue_odd: bonus_s += weights_config['COMBINATION_BLUE_ODD_MATCH_BONUS']
        if hist_zone_dist:
            zones_count = [0,0,0]
            for ball_num_in_combo in r_list:
                if RED_ZONES['Zone1'][0]<=ball_num_in_combo<=RED_ZONES['Zone1'][1]: zones_count[0]+=1
                elif RED_ZONES['Zone2'][0]<=ball_num_in_combo<=RED_ZONES['Zone2'][1]: zones_count[1]+=1
                elif RED_ZONES['Zone3'][0]<=ball_num_in_combo<=RED_ZONES['Zone3'][1]: zones_count[2]+=1
            if tuple(zones_count)==hist_zone_dist: bonus_s += weights_config['COMBINATION_ZONE_MATCH_BONUS']
        if hist_blue_large is not None and (b_val>8)==hist_blue_large: bonus_s += weights_config['COMBINATION_BLUE_SIZE_MATCH_BONUS']

        arm_specific_bonus = 0
        combo_red_set = set(r_list)
        if not arm_rules_processed.empty and 'antecedents_set' in arm_rules_processed.columns:
            for _, rule in arm_rules_processed.iterrows():
                if isinstance(rule.get('antecedents_set'), set) and isinstance(rule.get('consequents_set'), set):
                    if rule['antecedents_set'].issubset(combo_red_set) and rule['consequents_set'].issubset(combo_red_set):
                        lift_bonus = (rule.get('lift', 1.0) - 1.0) * weights_config.get('ARM_BONUS_LIFT_FACTOR', 0.2) 
                        conf_bonus = rule.get('confidence', 0.0) * weights_config.get('ARM_BONUS_CONF_FACTOR', 0.1)
                        current_rule_bonus = (lift_bonus + conf_bonus) * weights_config['ARM_COMBINATION_BONUS_WEIGHT']
                        arm_specific_bonus += current_rule_bonus
            arm_specific_bonus = min(arm_specific_bonus, weights_config['ARM_COMBINATION_BONUS_WEIGHT'] * 2.0) 

        bonus_s += arm_specific_bonus
        scored_combos.append({
            'combination': combo_item, 
            'score': base_s + bonus_s,
            'red_tuple': tuple(sorted(r_list)) 
        })
        
    final_recs_data = []
    if not scored_combos: 
        return [], ["推荐组合:", "无法生成推荐组合 (评分后为空)。"]

    sorted_scored_combos = sorted(scored_combos, key=lambda x: x['score'], reverse=True)

    if sorted_scored_combos:
        final_recs_data.append(sorted_scored_combos.pop(0)) 
    
    attempts_for_diversity = 0
    candidate_idx = 0
    while len(final_recs_data) < num_combinations_to_generate and candidate_idx < len(sorted_scored_combos):
        if attempts_for_diversity > diversity_max_attempts * num_combinations_to_generate : 
            logger.debug(f"多样性选择达到最大尝试次数，当前已选 {len(final_recs_data)} 组合。")
            break 
            
        candidate_combo_dict = sorted_scored_combos[candidate_idx]
        candidate_red_set = set(candidate_combo_dict['red_tuple'])
        is_diverse_enough = True
        
        for existing_rec_dict in final_recs_data:
            existing_red_set = set(existing_rec_dict['red_tuple'])
            common_reds = len(candidate_red_set.intersection(existing_red_set))
            if common_reds > max_common_reds_allowed:
                is_diverse_enough = False
                break
        
        if is_diverse_enough:
            final_recs_data.append(candidate_combo_dict)
        
        candidate_idx += 1
        attempts_for_diversity +=1

    if len(final_recs_data) < num_combinations_to_generate:
        logger.debug(f"多样性选择后组合数不足 ({len(final_recs_data)})，将从剩余高分组合中补充。")
        final_recs_red_tuples = {rec['red_tuple'] for rec in final_recs_data}
        
        needed_more = num_combinations_to_generate - len(final_recs_data)
        added_count = 0
        for combo_dict in sorted(scored_combos, key=lambda x: x['score'], reverse=True): 
            if added_count >= needed_more:
                break
            if combo_dict['red_tuple'] not in final_recs_red_tuples:
                 is_already_present = any(fc['combination'] == combo_dict['combination'] for fc in final_recs_data)
                 if not is_already_present:
                    final_recs_data.append(combo_dict)
                    final_recs_red_tuples.add(combo_dict['red_tuple']) 
                    added_count += 1
    
    final_recs_data = sorted(final_recs_data, key=lambda x: x['score'], reverse=True)[:num_combinations_to_generate]

    output_strs = [f"  组合 {i+1}: 红球 {sorted(rec['combination']['red'])} 蓝球 {rec['combination']['blue']} (综合分: {rec['score']:.2f})"
                   for i,rec in enumerate(final_recs_data)] if final_recs_data else ["  无法生成推荐组合。"]
    return final_recs_data, ["推荐组合 (Top {}):".format(len(final_recs_data))] + output_strs


def analyze_and_recommend(
    df_historical: pd.DataFrame, ml_lags_list: List[int], weights_config: Dict,
    association_rules_df_main: pd.DataFrame,
    train_ml: bool = True, existing_models: Optional[Dict] = None
) -> tuple[List[Dict], list[str], dict, Optional[Dict], Dict, Dict[str, float]]:
    recs_data, recs_strs = [], []
    analysis_res = {}; current_models = None; scores_res = {}; win_seg_pcts = {}

    if df_historical is None or df_historical.empty:
        return recs_data, recs_strs, analysis_res, current_models, scores_res, win_seg_pcts

    freq_om_data = analyze_frequency_omission(df_historical, weights_config)
    patt_an_data = analyze_patterns(df_historical, weights_config)
    analysis_res = {'freq_omission': freq_om_data, 'patterns': patt_an_data, 'associations': association_rules_df_main}

    pred_probs = {}
    min_ml_periods = (max(ml_lags_list) if ml_lags_list else 0) + 1 + MIN_POSITIVE_SAMPLES_FOR_ML
    if len(df_historical) >= min_ml_periods:
        if train_ml:
            current_models = train_prediction_models(df_historical, ml_lags_list, weights_config)
            if current_models: pred_probs = predict_next_draw_probabilities(df_historical, current_models, ml_lags_list, weights_config)
        elif existing_models:
            current_models = existing_models
            if current_models: pred_probs = predict_next_draw_probabilities(df_historical, current_models, ml_lags_list, weights_config)

    scores_res = calculate_scores(freq_om_data, patt_an_data, pred_probs, weights_config)
    _, win_seg_pcts = analyze_winning_red_ball_score_segments(
        df_historical, scores_res.get('red_scores',{}), SCORE_SEGMENT_BOUNDARIES, SCORE_SEGMENT_LABELS
    )
    recs_data, recs_strs = generate_combinations(
        scores_res, patt_an_data, association_rules_df_main, win_seg_pcts, weights_config
    )
    return recs_data, recs_strs, analysis_res, current_models, scores_res, win_seg_pcts

def get_prize_level(red_hits: int, blue_hit: bool) -> Optional[str]:
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
             association_rules_full_history: pd.DataFrame,
             backtest_periods_to_eval: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    max_hist_lag = max(ml_lags_list) if ml_lags_list else 0
    min_initial_train_periods = max_hist_lag + 1 + MIN_POSITIVE_SAMPLES_FOR_ML
    if len(df) < min_initial_train_periods + 1:
        logger.warning(f"Backtest: Data不足({len(df)})，需要至少 {min_initial_train_periods + 1} 期。")
        return pd.DataFrame(), {}

    start_prediction_loop_idx = min_initial_train_periods
    end_prediction_loop_idx = len(df) - 1
    if start_prediction_loop_idx > end_prediction_loop_idx:
        logger.warning("Backtest: 无足够后续数据进行预测。")
        return pd.DataFrame(), {}

    available_periods_for_eval = end_prediction_loop_idx - start_prediction_loop_idx + 1
    actual_periods_to_evaluate_in_loop = min(backtest_periods_to_eval, available_periods_for_eval)

    evaluation_loop_start_df_idx = max(start_prediction_loop_idx, end_prediction_loop_idx - actual_periods_to_evaluate_in_loop + 1)

    results_list = []
    red_cols_list = [f'red{i+1}' for i in range(6)]
    is_opt_run_flag = backtest_periods_to_eval == OPTIMIZATION_BACKTEST_PERIODS
    
    prize_counts = Counter()
    best_hit_per_period = []
    periods_with_any_blue_hit = set()
    num_combinations_generated_per_run = weights_config.get('NUM_COMBINATIONS_TO_GENERATE', 10)

    for df_idx_for_prediction in range(evaluation_loop_start_df_idx, end_prediction_loop_idx + 1):
        if not is_opt_run_flag and (df_idx_for_prediction - evaluation_loop_start_df_idx + 1) % 10 == 0 :
            # Use specific logger for console to ensure simple format for progress
            current_console_level = global_console_handler.level
            current_console_formatter = global_console_handler.formatter
            set_console_verbosity(logging.INFO, use_simple_formatter=True)
            logger.info(f"  回测进度: {df_idx_for_prediction - evaluation_loop_start_df_idx + 1} / {actual_periods_to_evaluate_in_loop}")
            global_console_handler.setLevel(current_console_level) # Restore
            global_console_handler.setFormatter(current_console_formatter)


        current_train_data = df.iloc[:df_idx_for_prediction].copy()
        if len(current_train_data) < min_initial_train_periods:
            continue

        actual_outcome_row = df.loc[df_idx_for_prediction]
        current_period_actual = actual_outcome_row['期号']
        try:
            actual_red_set = set(actual_outcome_row[red_cols_list].astype(int).tolist())
            actual_blue_val = int(actual_outcome_row['blue'])
            if not (all(1<=r_val<=33 for r_val in actual_red_set) and 1<=actual_blue_val<=16):
                raise ValueError("实际球号超出范围")
        except Exception as e_actual:
            logger.debug(f"Backtest: 获取期号 {current_period_actual} 实际结果失败: {e_actual}")
            continue

        original_logger_level = logger.level # For file logger
        original_console_level = global_console_handler.level
        
        if is_opt_run_flag: 
            logger.setLevel(logging.CRITICAL) # Suppress file log heavily for optuna
            set_console_verbosity(logging.CRITICAL) # Suppress console heavily

        current_arm_rules = analyze_associations(current_train_data, weights_config)

        if is_opt_run_flag:
            with SuppressOutput(suppress_stdout=True, capture_stderr=True): # Also suppress direct stdout/stderr
                 predicted_combos_list, _, _, _, _, _ = analyze_and_recommend(
                     current_train_data, ml_lags_list, weights_config, current_arm_rules, train_ml=True)
        else:
            predicted_combos_list, _, _, _, _, _ = analyze_and_recommend(
                current_train_data, ml_lags_list, weights_config, current_arm_rules, train_ml=True)

        if is_opt_run_flag: 
            logger.setLevel(original_logger_level)
            set_console_verbosity(original_console_level)


        period_max_red_hits = 0
        period_blue_hit_achieved = False

        if predicted_combos_list:
            for combo_dict_info in predicted_combos_list:
                pred_r_set = set(combo_dict_info['combination']['red'])
                pred_b_val = combo_dict_info['combination']['blue']
                red_h = len(pred_r_set.intersection(actual_red_set))
                blue_h = (pred_b_val == actual_blue_val)

                results_list.append({
                    'period': current_period_actual,
                    'predicted_red': sorted(list(pred_r_set)), 'predicted_blue': pred_b_val,
                    'actual_red': sorted(list(actual_red_set)), 'actual_blue': actual_blue_val,
                    'red_hits': red_h,
                    'blue_hit': blue_h,
                    'combination_score': combo_dict_info['score']
                })
                
                prize = get_prize_level(red_h, blue_h)
                if prize:
                    prize_counts[prize] += 1
                
                if blue_h:
                    periods_with_any_blue_hit.add(current_period_actual)
                    period_blue_hit_achieved = True # if any combo hits blue for this period
                if red_h > period_max_red_hits:
                    period_max_red_hits = red_h
            
            best_hit_per_period.append({
                'period': current_period_actual,
                'max_red_hits': period_max_red_hits,
                'blue_hit_in_period': period_blue_hit_achieved 
            })

    if not results_list: return pd.DataFrame(), {}
    
    results_df_final = pd.DataFrame(results_list)
    extended_stats = {
        'prize_counts': dict(prize_counts),
        'best_hit_per_period_df': pd.DataFrame(best_hit_per_period) if best_hit_per_period else pd.DataFrame(),
        'total_combinations_evaluated': len(results_df_final),
        'num_combinations_per_draw_tested': num_combinations_generated_per_run,
        'periods_with_any_blue_hit_count': len(periods_with_any_blue_hit)
    }
    
    if '期号' in df.columns and evaluation_loop_start_df_idx < len(df) and end_prediction_loop_idx < len(df):
        try:
            results_df_final.attrs['start_period'] = df.loc[evaluation_loop_start_df_idx, '期号']
            results_df_final.attrs['end_period'] = df.loc[end_prediction_loop_idx, '期号']
        except KeyError:
             logger.warning("Backtest: Could not set start/end period attributes due to index issues.")
    return results_df_final, extended_stats

def objective(trial: optuna.trial.Trial, df_for_optimization: pd.DataFrame, fixed_ml_lags: List[int],
              arm_rules_for_opt: pd.DataFrame) -> float:
    weights_to_eval_base = {
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
        'ARM_MIN_SUPPORT': trial.suggest_float('ARM_MIN_SUPPORT', 0.005, 0.02),
        'ARM_MIN_CONFIDENCE': trial.suggest_float('ARM_MIN_CONFIDENCE', 0.25, 0.5),
        'ARM_MIN_LIFT': trial.suggest_float('ARM_MIN_LIFT', 1.0, 1.5),
        'ARM_COMBINATION_BONUS_WEIGHT': trial.suggest_float('ARM_COMBINATION_BONUS_WEIGHT', 0, 20),
        'ARM_BONUS_LIFT_FACTOR': trial.suggest_float('ARM_BONUS_LIFT_FACTOR', 0.05, 0.5),
        'ARM_BONUS_CONF_FACTOR': trial.suggest_float('ARM_BONUS_CONF_FACTOR', 0.05, 0.5),
        'CANDIDATE_POOL_MIN_PER_SEGMENT': trial.suggest_int('CANDIDATE_POOL_MIN_PER_SEGMENT', 1, 4),
        'CANDIDATE_POOL_PROPORTIONS_HIGH': trial.suggest_float('CANDIDATE_POOL_PROPORTIONS_HIGH', 0.2, 0.7),
        'CANDIDATE_POOL_PROPORTIONS_MEDIUM': trial.suggest_float('CANDIDATE_POOL_PROPORTIONS_MEDIUM', 0.1, 0.5),
        'DIVERSITY_MIN_DIFFERENT_REDS': trial.suggest_int('DIVERSITY_MIN_DIFFERENT_REDS', 2, 4),
        'DIVERSITY_SELECTION_MAX_ATTEMPTS': trial.suggest_int('DIVERSITY_SELECTION_MAX_ATTEMPTS', 10, 50),
    }
    
    if weights_to_eval_base['CANDIDATE_POOL_PROPORTIONS_HIGH'] + weights_to_eval_base['CANDIDATE_POOL_PROPORTIONS_MEDIUM'] > 1.0:
        weights_to_eval_base['CANDIDATE_POOL_PROPORTIONS_MEDIUM'] = 1.0 - weights_to_eval_base['CANDIDATE_POOL_PROPORTIONS_HIGH']
        if weights_to_eval_base['CANDIDATE_POOL_PROPORTIONS_MEDIUM'] < 0 : 
             weights_to_eval_base['CANDIDATE_POOL_PROPORTIONS_MEDIUM'] = 0 

    weights_to_eval = weights_to_eval_base

    backtest_results_df, _ = backtest(df_for_optimization, fixed_ml_lags, weights_to_eval,
                                   arm_rules_for_opt,
                                   OPTIMIZATION_BACKTEST_PERIODS)

    if backtest_results_df.empty or len(backtest_results_df) == 0:
        return float('inf')

    avg_weighted_red_hits = (backtest_results_df['red_hits'] ** 1.5).mean()

    unique_periods_tested = backtest_results_df['period'].nunique()
    if unique_periods_tested == 0: # Should not happen if backtest_results_df is not empty
        blue_hit_rate_per_period = 0.0
    else:
        # Count periods where at least one combination hit the blue ball
        blue_hit_periods_count = backtest_results_df.groupby('period')['blue_hit'].any().sum()
        blue_hit_rate_per_period = blue_hit_periods_count / unique_periods_tested
    
    RED_PERF_WEIGHT = 1.0
    BLUE_PERF_WEIGHT = 10.0 # Blue ball hits are rarer and more impactful for prize

    performance_score = (RED_PERF_WEIGHT * avg_weighted_red_hits) + \
                        (BLUE_PERF_WEIGHT * blue_hit_rate_per_period)

    if blue_hit_rate_per_period == 0 and avg_weighted_red_hits < 0.5 :
        performance_score -= 1.0 # Penalty for very poor performance

    return -performance_score


if __name__ == "__main__":
    log_filename = os.path.join(SCRIPT_DIR, f"ssq_analysis_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    set_console_verbosity(logging.INFO, use_simple_formatter=True)

    logger.info(f"--- 双色球分析报告 ---")
    logger.info(f"运行日期: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"日志文件: {log_filename}")
    logger.info(f"红球候选池分数阈值: High > {CANDIDATE_POOL_SCORE_THRESHOLDS['High']}, Medium > {CANDIDATE_POOL_SCORE_THRESHOLDS['Medium']}")
    logger.info("-" * 30)

    CURRENT_WEIGHTS = load_weights_from_file(WEIGHTS_CONFIG_FILE, DEFAULT_WEIGHTS)
    weights_loaded_from_file = not (all(CURRENT_WEIGHTS.get(k) == v for k,v in DEFAULT_WEIGHTS.items()) and \
                                    all(k in CURRENT_WEIGHTS for k in DEFAULT_WEIGHTS) and \
                                    not os.path.exists(WEIGHTS_CONFIG_FILE))
    
    original_console_level = global_console_handler.level # Save current console level
    set_console_verbosity(logging.INFO, use_simple_formatter=False) # Detailed for data loading initially
    main_df = None
    if os.path.exists(PROCESSED_CSV_PATH):
        df_proc = load_data(PROCESSED_CSV_PATH)
        required_cols = [f'red{i+1}' for i in range(6)] + ['blue', '期号', 'red_sum']
        if df_proc is not None and not df_proc.empty and all(c in df_proc.columns for c in required_cols):
            main_df = df_proc
            logger.info(f"成功加载已处理数据: {PROCESSED_CSV_PATH}")
    
    if main_df is None:
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
    
    for r_col_m in [f'red{i+1}' for i in range(6)]:
        if r_col_m in main_df.columns: main_df[r_col_m] = pd.to_numeric(main_df[r_col_m], errors='coerce')
    if 'blue' in main_df.columns: main_df['blue'] = pd.to_numeric(main_df['blue'], errors='coerce')
    main_df.dropna(subset=([f'red{i+1}' for i in range(6)] + ['blue']), inplace=True)

    set_console_verbosity(original_console_level, use_simple_formatter=True) # Restore console level (likely simple INFO)

    full_history_arm_rules = analyze_associations(main_df, CURRENT_WEIGHTS)

    if not weights_loaded_from_file:
        logger.info("\n>>> 开始权重优化过程...")
        min_data_for_opt = (max(ML_LAG_FEATURES) if ML_LAG_FEATURES else 0) + 1 + MIN_POSITIVE_SAMPLES_FOR_ML + OPTIMIZATION_BACKTEST_PERIODS
        if len(main_df) >= min_data_for_opt:
            df_for_opt_objective = main_df.copy()
            optuna.logging.set_verbosity(optuna.logging.WARNING) # Suppress Optuna's own INFO logs
            
            optuna_study = optuna.create_study(direction='minimize')
            logger.info(f"Optuna优化试验次数: {OPTIMIZATION_TRIALS}, 超时: {7200}s")
            
            optuna_study.optimize(lambda trial: objective(trial, df_for_opt_objective, ML_LAG_FEATURES, full_history_arm_rules),
                                  n_trials=OPTIMIZATION_TRIALS, timeout=7200)

            best_params_from_optuna = optuna_study.best_params
            updated_weights = DEFAULT_WEIGHTS.copy()

            for key, value in best_params_from_optuna.items():
                 if key in updated_weights:
                     if isinstance(updated_weights[key], int):
                         updated_weights[key] = int(round(value))
                     elif isinstance(updated_weights[key], float):
                         updated_weights[key] = float(value)
            
            prop_h_opt = updated_weights.get('CANDIDATE_POOL_PROPORTIONS_HIGH', DEFAULT_WEIGHTS['CANDIDATE_POOL_PROPORTIONS_HIGH'])
            prop_m_opt = updated_weights.get('CANDIDATE_POOL_PROPORTIONS_MEDIUM', DEFAULT_WEIGHTS['CANDIDATE_POOL_PROPORTIONS_MEDIUM'])
            if prop_h_opt + prop_m_opt > 1.0: prop_m_opt = 1.0 - prop_h_opt
            if prop_m_opt < 0: prop_m_opt = 0
            updated_weights['CANDIDATE_POOL_PROPORTIONS_HIGH'] = prop_h_opt
            updated_weights['CANDIDATE_POOL_PROPORTIONS_MEDIUM'] = prop_m_opt

            logger.info(f"\n>>> Optuna优化完成。")
            logger.info(f"  最佳目标函数值: {optuna_study.best_value:.4f}")
            CURRENT_WEIGHTS = updated_weights
            save_weights_to_file(WEIGHTS_CONFIG_FILE, CURRENT_WEIGHTS)
            full_history_arm_rules = analyze_associations(main_df, CURRENT_WEIGHTS) 
        else:
            logger.warning(f"数据不足 ({len(main_df)}期) 进行权重优化 (需要至少 {min_data_for_opt}期)。将使用默认/已加载权重。")
    else:
        logger.info("\n>>> 已加载现有权重，跳过优化。")

    logger.info(f"\n>>> 当前使用权重 (部分展示):")
    logger.info(f"  NUM_COMBINATIONS_TO_GENERATE: {CURRENT_WEIGHTS['NUM_COMBINATIONS_TO_GENERATE']}")
    logger.info(f"  DIVERSITY_MIN_DIFFERENT_REDS: {CURRENT_WEIGHTS['DIVERSITY_MIN_DIFFERENT_REDS']}")
    logger.info(f"  ML_PROB_SCORE_WEIGHT_RED: {CURRENT_WEIGHTS.get('ML_PROB_SCORE_WEIGHT_RED', 0.0):.2f}")
    logger.info(f"  ARM_COMBINATION_BONUS_WEIGHT: {CURRENT_WEIGHTS.get('ARM_COMBINATION_BONUS_WEIGHT',0.0):.2f}")

    min_p_val, max_p_val, total_p_val = main_df['期号'].min(), main_df['期号'].max(), len(main_df)
    last_draw_dt = main_df['日期'].iloc[-1] if '日期' in main_df.columns and not main_df.empty else "未知"
    last_draw_period = main_df['期号'].iloc[-1] if not main_df.empty else "未知"
    
    set_console_verbosity(logging.INFO, use_simple_formatter=False) 
    logger.info(f"\n{'='*15} 数据概况 {'='*15}")
    logger.info(f"  数据范围: {min_p_val} - {max_p_val} (共 {total_p_val} 期)")
    logger.info(f"  最后开奖: {last_draw_dt} (期号: {last_draw_period})")

    min_periods_for_full_run = (max(ML_LAG_FEATURES) if ML_LAG_FEATURES else 0) + 1 + MIN_POSITIVE_SAMPLES_FOR_ML + BACKTEST_PERIODS_COUNT
    if total_p_val < min_periods_for_full_run:
        logger.error(f"数据不足 ({total_p_val}期) 进行完整分析和回测报告 (需 {min_periods_for_full_run}期)。")
    else:
        logger.info(f"\n{'='*10} 完整历史统计分析 {'='*10}")
        original_console_level_stats = global_console_handler.level # Store before changing
        set_console_verbosity(logging.INFO, use_simple_formatter=False) # Ensure detailed for this section
        full_freq_d = analyze_frequency_omission(main_df, CURRENT_WEIGHTS)
        full_patt_d = analyze_patterns(main_df, CURRENT_WEIGHTS)
        
        logger.info(f"  热门红球 (Top 5): {[int(x) for x in full_freq_d.get('hot_reds', [])[:5]]}")
        logger.info(f"  冷门红球 (Bottom 5): {[int(x) for x in full_freq_d.get('cold_reds', [])[-5:]]}")
        logger.info(f"  最近 {RECENT_FREQ_WINDOW} 期热门红球: " + str(sorted([(int(k),v) for k,v in full_freq_d.get('recent_N_freq_red', {}).items() if v > 0], key=lambda x: x[1], reverse=True)[:5]))
        logger.info(f"  最常见红球奇偶比: {full_patt_d.get('most_common_odd_even_count')}")
        if not full_history_arm_rules.empty: logger.info(f"  发现 {len(full_history_arm_rules)} 条关联规则 (Top 3 LIFT): \n{full_history_arm_rules.head(3).to_string(index=False)}")
        else: logger.info("  未找到显著关联规则.")
        global_console_handler.setLevel(original_console_level_stats) # Restore console level

        logger.info(f"\n{'='*15} 回测摘要 {'='*15}")
        set_console_verbosity(logging.INFO, use_simple_formatter=True) 
        backtest_res_df, extended_bt_stats = backtest(main_df, ML_LAG_FEATURES, CURRENT_WEIGHTS, full_history_arm_rules, BACKTEST_PERIODS_COUNT)
        set_console_verbosity(logging.INFO, use_simple_formatter=False)

        if not backtest_res_df.empty:
            s_p_f = backtest_res_df.attrs.get('start_period', 'N/A'); e_p_f = backtest_res_df.attrs.get('end_period', 'N/A')
            num_tested_periods = backtest_res_df['period'].nunique()
            logger.info(f"  回测期范围: {s_p_f} 至 {e_p_f} (共测试 {num_tested_periods} 期)")
            logger.info(f"  每期生成组合数: {extended_bt_stats.get('num_combinations_per_draw_tested', 'N/A')}")
            logger.info(f"  总评估组合数: {extended_bt_stats.get('total_combinations_evaluated', 'N/A')}")
            
            logger.info(f"  --- 整体命中表现 ---")
            logger.info(f"    每个组合平均红球命中: {backtest_res_df['red_hits'].mean():.3f}")
            logger.info(f"    每个组合加权(x^1.5)平均红球命中: {(backtest_res_df['red_hits']**1.5).mean():.3f}")
            
            blue_hit_overall_rate = backtest_res_df['blue_hit'].mean() * 100 
            logger.info(f"    蓝球命中率 (每个组合): {blue_hit_overall_rate:.2f}%")
            
            periods_any_blue_hit = extended_bt_stats.get('periods_with_any_blue_hit_count', 0)
            if num_tested_periods > 0:
                logger.info(f"    至少一个组合命中蓝球的期数占比: {periods_any_blue_hit / num_tested_periods:.2%}")

            logger.info(f"  --- 红球命中数分布 (按组合) ---")
            hit_counts_dist = backtest_res_df['red_hits'].value_counts(normalize=True).sort_index() * 100
            for hit_num, pct in hit_counts_dist.items():
                logger.info(f"    命中 {hit_num} 红球: {pct:.2f}%")

            logger.info(f"  --- 中奖等级统计 (按组合) ---")
            prize_dist = extended_bt_stats.get('prize_counts', {})
            if prize_dist:
                prize_order = {"一等奖": 1, "二等奖": 2, "三等奖": 3, "四等奖": 4, "五等奖": 5, "六等奖": 6}
                # Use OrderedDict to preserve custom sort order if needed, or just sort by value from prize_order
                sorted_prize_dist = sorted(prize_dist.items(), key=lambda item: prize_order.get(item[0], 99))
                for prize_level, count in sorted_prize_dist:
                    logger.info(f"    {prize_level}: {count} 次")
            else:
                logger.info("    未命中任何奖级。")

            best_hits_df = extended_bt_stats.get('best_hit_per_period_df')
            if best_hits_df is not None and not best_hits_df.empty:
                logger.info(f"  --- 每期最佳红球命中数分布 ---") # Best hit among N combinations for that period
                best_red_dist = best_hits_df['max_red_hits'].value_counts(normalize=True).sort_index() * 100
                for hit_num, pct in best_red_dist.items():
                    logger.info(f"    最佳命中 {hit_num} 红球的期数占比: {pct:.2f}%")
                
                # Periods where blue was hit by any of the N combinations
                if 'blue_hit_in_period' in best_hits_df.columns:
                     periods_with_best_blue_hit = best_hits_df['blue_hit_in_period'].sum()
                     if num_tested_periods > 0:
                         logger.info(f"    至少一个组合命中蓝球的期数占比 (来自best_hit_per_period): {periods_with_best_blue_hit / num_tested_periods:.2%}")


        else: logger.info("  最终回测未产生结果。")

        logger.info(f"\n{'='*12} 最终推荐号码 {'='*12}")
        set_console_verbosity(logging.INFO, use_simple_formatter=True)
        final_recs_list, final_rec_strs_list, _, _, final_scores_dict, _ = analyze_and_recommend(
            main_df, ML_LAG_FEATURES, CURRENT_WEIGHTS, full_history_arm_rules, train_ml=True
        )
        for line_str in final_rec_strs_list: logger.info(line_str)

        set_console_verbosity(logging.INFO, use_simple_formatter=False)
        logger.info(f"\n{'='*8} 中奖红球分数段历史分析 {'='*8}")
        if final_scores_dict.get('red_scores'):
            disp_cts, disp_pcts_vals = analyze_winning_red_ball_score_segments(main_df, final_scores_dict['red_scores'], SCORE_SEGMENT_BOUNDARIES, SCORE_SEGMENT_LABELS)
            tot_win_reds_d = sum(disp_cts.values())
            logger.info(f"  历史中奖红球分数段分布 (总计 {tot_win_reds_d} 个):")
            for seg_name in sorted(disp_cts.keys()):
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

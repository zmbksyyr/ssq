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
from sklearn.ensemble import RandomForestClassifier # Only Classifier is used
from sklearn.metrics import accuracy_score, mean_squared_error # mean_squared_error is not used
# import joblib # For saving/loading models (optional for backtest, good practice) - Currently not used
from typing import Union, Optional, List, Dict, Tuple, Any 

import sys
import datetime
import os
import requests
from bs4 import BeautifulSoup
import io
import logging
from contextlib import redirect_stdout, redirect_stderr # Keep redirect_stderr for custom SuppressOutput

# --- Configuration ---
# Get the directory of the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the CSV file
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'shuangseqiu.csv') # Assumes shuangseqiu.csv is in the same directory as the script

RED_BALL_RANGE = range(1, 34)
BLUE_BALL_RANGE = range(1, 17)
RED_ZONES = {
    'Zone1': (1, 11),
    'Zone2': (12, 22),
    'Zone3': (23, 33)
}
NUM_COMBINATIONS_TO_GENERATE = 5 # 最终推荐的号码组合数量 (单式或小复式)
TOP_N_RED_FOR_CANDIDATE = 25 # 用于生成组合的红球候选池大小（按分数从高到低选择）
TOP_N_BLUE_FOR_CANDIDATE = 10 # 用于生成组合的蓝球候选池大小（按分数从高到低选择）
ML_LAG_FEATURES = [1, 3, 5] # ML模型使用的滞后特征期数，例如 [1, 3, 5] 表示使用前1期、前3期、前5期的数据作为特征
BACKTEST_PERIODS_COUNT = 200 # 回测使用的最近历史期数 (This will be maximum periods if data is sufficient)
SHOW_PLOTS = False # 是否显示分析过程中生成的图表 (True 显示, False 屏蔽)

# Association Rule Mining Config (Adjust if needed)
ARM_MIN_SUPPORT = 0.005
ARM_MIN_CONFIDENCE = 0.3
ARM_MIN_LIFT = 1.0

# Scoring Weights (Heuristic - Can be tuned)
FREQ_SCORE_WEIGHT = 30
OMISSION_SCORE_WEIGHT = 20
ODD_EVEN_TENDENCY_BONUS = 10 # For red balls matching predicted odd/even tendency
ZONE_TENDENCY_BONUS_MULTIPLIER = 2 # Multiplier for zone count in most common pattern
BLUE_FREQ_SCORE_WEIGHT = 40
BLUE_OMISSION_SCORE_WEIGHT = 30
BLUE_ODD_TENDENCY_BONUS = 20 # For blue ball matching predicted odd/even tendency
BLUE_SIZE_TENDENCY_BONUS = 10 # For blue ball matching most common size tendency
COMBINATION_ODD_COUNT_MATCH_BONUS = 20 # Bonus for combination matching predicted odd count
COMBINATION_BLUE_ODD_MATCH_BONUS = 15 # Bonus for combination matching predicted blue odd
COMBINATION_ZONE_MATCH_BONUS = 15 # Bonus for combination matching most common zone pattern
COMBINATION_BLUE_SIZE_MATCH_BONUS = 10 # Bonus for combination matching most common blue size

# ML Model Parameters (RandomForest)
RF_ESTIMATORS = 50
RF_MAX_DEPTH = 10


# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Output to console stdout
    ]
)
logger = logging.getLogger('ssq_analyzer')

# 添加一个进度条显示函数
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
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', file=sys.stdout, flush=True) # Ensure it prints to stdout and flushes
    if current >= total:
        print(file=sys.stdout, flush=True)


# 创建一个上下文管理器来暂时重定向输出，并捕获 stderr
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
            # Redirect stdout to /dev/null or equivalent
            sys.stdout = open(os.devnull, 'w')

        if self.capture_stderr:
            self.old_stderr = sys.stderr
            self.stderr_io = io.StringIO()
            sys.stderr = self.stderr_io

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore stderr first
        if self.capture_stderr and self.old_stderr:
            sys.stderr = self.old_stderr
            captured_stderr_content = self.stderr_io.getvalue()
            if captured_stderr_content.strip(): # Log captured stderr if not empty
                 logger.warning(f"Captured stderr:\n{captured_stderr_content.strip()}")

        # Restore stdout
        if self.suppress_stdout and self.old_stdout:
            if sys.stdout and not sys.stdout.closed: # Check if redirect object is valid before closing
                 sys.stdout.close()
            sys.stdout = self.old_stdout

        # Don't suppress exceptions
        return False


# --- 新增：从网站获取最新数据 ---

def fetch_latest_data(url: str = "https://www.17500.cn/chart/ssq-tjb.html") -> List[Dict]:
    """从指定网站获取最新双色球数据"""
    logger.info("正在从网站获取最新双色球数据...")
    data = []
    try:
        # 发送请求获取页面内容，禁用代理
        session = requests.Session()
        session.trust_env = False  # 禁用环境变量中的代理设置

        response = session.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }, timeout=30)  # 添加超时设置

        response.raise_for_status()  # 检查请求是否成功

        # 解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # 找到表格
        table = soup.find('table')
        if not table:
            logger.warning("无法在网页中找到表格数据")
            return []

        # 解析表格数据
        rows = table.find_all('tr')

        # 跳过表头，从第一行数据开始
        for row in rows[1:]:  # 假设第一行是表头
            cells = row.find_all('td')
            # Ensure enough cells exist and are not empty
            if len(cells) >= 3 and all(cell and cell.text.strip() for cell in cells[:3]):
                try:
                    # Extract period number
                    period_cell = cells[0]
                    if '<a' in str(period_cell):
                        period = period_cell.find('a').text.strip()
                    else:
                        period = period_cell.text.strip()

                    period = period.replace("期", "").strip()

                    # Check if period is a valid integer
                    if not period.isdigit():
                        # logger.debug(f"Skipping row with invalid period format: {cells[0].text.strip()}") # Too verbose
                        continue

                    red_balls_str = cells[1].text.strip().replace(" ", ",")
                    blue_ball_str = cells[2].text.strip()

                    # Basic validation for red/blue balls
                    if not red_balls_str or not blue_ball_str:
                         # logger.debug(f"Skipping row with empty red or blue ball data: {period}") # Too verbose
                         continue

                    # Optional: More rigorous format check for red balls
                    try:
                        red_numbers = [int(x) for x in red_balls_str.split(',')]
                        if len(red_numbers) != 6:
                            # logger.debug(f"Skipping row {period} due to incorrect number of red balls: {red_numbers}") # Too verbose
                            continue
                    except ValueError:
                         # logger.debug(f"Skipping row {period} due to invalid red ball format: {red_balls_str}") # Too verbose
                         continue

                    try:
                        blue_number = int(blue_ball_str)
                        if not (1 <= blue_number <= 16):
                            # logger.debug(f"Skipping row {period} due to invalid blue ball value: {blue_ball_str}") # Too verbose
                            continue
                    except ValueError:
                         # logger.debug(f"Skipping row {period} due to invalid blue ball format: {blue_ball_str}") # Too verbose
                         continue


                    data.append({
                        '期号': period,
                        '红球': red_balls_str,
                        '蓝球': blue_ball_str
                    })
                except Exception as e:
                    logger.warning(f"Error processing row: {cells}. Error: {e}")
                    continue # Skip problematic row

        logger.info(f"从网站成功获取了 {len(data)} 期双色球数据")
        return data

    except requests.exceptions.ProxyError:
        logger.warning("代理连接错误，尝试直接连接...")
        try:
            # 尝试不使用代理直接连接
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }, proxies={"http": None, "https": None}, timeout=30)

            response.raise_for_status()

            # Parse HTML - same logic as above
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table')
            if not table:
                logger.warning("直接连接：无法在网页中找到表格数据")
                return []

            data = []
            rows = table.find_all('tr')
            for row in rows[1:]:
                cells = row.find_all('td')
                if len(cells) >= 3 and all(cell and cell.text.strip() for cell in cells[:3]):
                    try:
                         period_cell = cells[0]
                         if '<a' in str(period_cell):
                              period = period_cell.find('a').text.strip()
                         else:
                              period = period_cell.text.strip()

                         period = period.replace("期", "").strip()
                         if not period.isdigit():
                              continue

                         red_balls_str = cells[1].text.strip().replace(" ", ",")
                         blue_ball_str = cells[2].text.strip()

                         if not red_balls_str or not blue_ball_str:
                              continue

                         try:
                              red_numbers = [int(x) for x in red_balls_str.split(',')]
                              if len(red_numbers) != 6:
                                   continue
                         except ValueError:
                              continue

                         try:
                              blue_number = int(blue_ball_str)
                              if not (1 <= blue_number <= 16):
                                   continue
                         except ValueError:
                              continue

                         data.append({
                             '期号': period,
                             '红球': red_balls_str,
                             '蓝球': blue_ball_str
                         })
                    except Exception as e:
                        logger.warning(f"Direct connect error processing row: {cells}. Error: {e}")
                        continue

            logger.info(f"直接连接成功获取了 {len(data)} 期双色球数据")
            return data

        except Exception as e:
            logger.error(f"直接连接也失败: {e}")
            return []

    except Exception as e:
        logger.error(f"获取网站数据时出错: {e}")
        return []


def update_csv_with_latest_data(csv_file_path: str):
    """获取最新数据并更新CSV文件"""
    logger.info("正在检查并更新最新双色球数据...")
    latest_data = fetch_latest_data()
    if not latest_data:
        logger.info("没有获取到新数据，CSV文件保持不变")
        return False

    try:
        # Read existing CSV file
        existing_df = pd.DataFrame(columns=['期号', '红球', '蓝球']) # Start with empty in case file doesn't exist or is empty
        if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
            try:
                existing_df = pd.read_csv(csv_file_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    existing_df = pd.read_csv(csv_file_path, encoding='gbk')
                except UnicodeDecodeError:
                    try:
                        existing_df = pd.read_csv(csv_file_path, encoding='latin-1')
                    except Exception as e:
                        logger.error(f"Failed to read CSV with multiple encodings: {e}")
                        existing_df = pd.DataFrame(columns=['期号', '红球', '蓝球']) # Fallback to empty DF on read error
            except pd.errors.EmptyDataError:
                 logger.warning("Existing CSV file is empty.")
                 existing_df = pd.DataFrame(columns=['期号', '红球', '蓝球']) # Handle empty file

        # Ensure '期号' exists and can be converted to int
        if '期号' not in existing_df.columns:
             logger.warning("Existing CSV does not have a '期号' column. Starting fresh with latest data.")
             existing_df = pd.DataFrame(columns=['期号', '红球', '蓝球'])
        else:
            try:
                existing_df['期号'] = pd.to_numeric(existing_df['期号'], errors='coerce').astype('Int64') # Use nullable integer type
                existing_df.dropna(subset=['期号'], inplace=True) # Drop rows where 期号 could not be converted
                existing_df['期号'] = existing_df['期号'].astype(int) # Convert to standard int after dropping NaNs
            except Exception as e:
                logger.error(f"Failed to convert '期号' in existing CSV to integer: {e}. Starting fresh.")
                existing_df = pd.DataFrame(columns=['期号', '红球', '蓝球'])


        # Create new data DataFrame
        new_df = pd.DataFrame(latest_data)
        if '期号' not in new_df.columns:
             logger.error("Fetched data does not have a '期号' column.")
             return False

        # Convert new data '期号' to int, handling potential errors
        try:
            new_df['期号'] = pd.to_numeric(new_df['期号'], errors='coerce').astype('Int64') # Use nullable integer type
            new_df.dropna(subset=['期号'], inplace=True)
            new_df['期号'] = new_df['期号'].astype(int)
        except Exception as e:
             logger.error(f"Failed to convert '期号' in fetched data to integer: {e}")
             return False

        # Find new entries (in new_df but not in existing_df based on '期号')
        existing_periods = set(existing_df['期号'])
        new_entries_df = new_df[~new_df['期号'].isin(existing_periods)].copy()

        # If there is new data, append and save
        if not new_entries_df.empty:
            # Ensure columns match before concatenating
            # If existing_df was empty or reset, use new_entries_df columns
            if existing_df.empty:
                combined_df = new_entries_df
            else:
                # Ensure both dataframes have the same columns in the same order
                common_cols = list(existing_df.columns.intersection(new_entries_df.columns))
                combined_df = pd.concat([existing_df[common_cols], new_entries_df[common_cols]], ignore_index=True)


            combined_df = combined_df.sort_values(by='期号', ascending=True).drop_duplicates(subset=['期号'], keep='last') # Sort and remove duplicates just in case
            combined_df.reset_index(drop=True, inplace=True)

            # Save updated data - always use utf-8 for consistency
            combined_df.to_csv(csv_file_path, index=False, encoding='utf-8')
            logger.info(f"成功添加了 {len(new_entries_df)} 期新数据到CSV文件")
            return True
        else:
            logger.info("没有找到新的期号数据，CSV文件保持不变")
            return False

    except Exception as e:
        logger.error(f"更新CSV文件时出错: {e}")
        return False


# --- Phase 1: Data Preparation & Basic Processing ---

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """Loads data from CSV."""
    try:
        # Assuming CSV has columns like '期号', '开奖日期', '红球', '蓝球'
        # Try different encodings if default utf-8 fails
        try:
             df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
             try:
                 df = pd.read_csv(file_path, encoding='gbk')
             except UnicodeDecodeError:
                 df = pd.read_csv(file_path, encoding='latin-1') # Fallback to latin-1 or other
        logger.info("Data loaded successfully.")
        logger.info(f"Total periods read: {len(df)}")
        return df
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"Error: File is empty at {file_path}")
        return None
    except Exception as e:
        logger.error(f"An error occurred during data loading from {file_path}: {e}")
        return None


def clean_and_structure(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Cleans data and structures red/blue balls."""
    if df is None or df.empty:
        logger.warning("No data to clean and structure.")
        return None

    initial_rows = len(df)
    # Drop rows with missing '期号', '红球', or '蓝球'
    df.dropna(subset=['期号', '红球', '蓝球'], inplace=True)
    if len(df) < initial_rows:
        logger.warning(f"Dropped {initial_rows - len(df)} rows with missing essential values.")
        initial_rows = len(df) # Update for subsequent checks

    if df.empty:
        logger.warning("No data remaining after dropping rows with missing essential values.")
        return None

    # Ensure '期号' is integer and sort by it
    try:
        df['期号'] = pd.to_numeric(df['期号'], errors='coerce').astype('Int64') # Use nullable integer type
        df.dropna(subset=['期号'], inplace=True) # Drop rows where conversion failed
        df['期号'] = df['期号'].astype(int) # Convert to standard int
        df.sort_values(by='期号', ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True) # Reset index after sorting
    except Exception as e:
        logger.error(f"Error: '期号' column could not be cleaned or converted to integer. {e}")
        return None

    if df.empty:
        logger.warning("No data remaining after cleaning '期号'.")
        return None

    red_balls_list = []
    blue_balls_list = []
    # Also add columns for red balls in sorted position
    red_pos_cols_list = [[] for _ in range(6)]

    rows_skipped_parsing = 0

    for index, row in df.iterrows():
        try:
            red_str = row['红球']
            blue_val = row['蓝球']

            if not isinstance(red_str, str):
                 rows_skipped_parsing += 1
                 continue # Skip if red ball data is not a string

            # Handle potential non-integer blue ball data early
            try:
                 blue_num = int(blue_val)
                 if not (1 <= blue_num <= 16):
                     rows_skipped_parsing += 1
                     # logger.debug(f"Skipping row {row['期号']} due to invalid blue ball value: {blue_val}") # Too verbose
                     continue
                 blue_balls_list.append(blue_num)
            except ValueError:
                 rows_skipped_parsing += 1
                 # logger.debug(f"Skipping row {row['期号']} due to invalid blue ball format: {blue_val}") # Too verbose
                 continue


            # Process red balls after blue ball is validated
            try:
                reds = sorted([int(x) for x in red_str.split(',')]) # Sorted red balls
                if len(reds) != 6 or not all(1 <= r <= 33 for r in reds):
                    rows_skipped_parsing += 1
                    # logger.debug(f"Skipping row {row['期号']} due to incorrect red ball count or values: {reds}") # Too verbose
                    continue
                red_balls_list.append(reds)

                # Populate positional red ball lists
                for i in range(6):
                    red_pos_cols_list[i].append(reds[i])

            except ValueError:
                rows_skipped_parsing += 1
                # logger.debug(f"Skipping row {row['期号']} due to invalid red ball number format: {red_str}") # Too verbose
                continue
            except Exception as e:
                rows_skipped_parsing += 1
                logger.warning(f"An unexpected error occurred parsing red balls at row {row['期号']}: {e}. Skipping.")
                continue

        except Exception as e:
            rows_skipped_parsing += 1
            logger.warning(f"An unexpected error occurred processing row {row['期号']}: {e}. Skipping.")
            continue


    if rows_skipped_parsing > 0:
         logger.warning(f"Skipped {rows_skipped_parsing} rows due to parsing errors in red or blue ball data.")


    # Create a new DataFrame with only successfully parsed rows
    if not red_balls_list or not blue_balls_list or len(red_balls_list) != len(blue_balls_list):
         logger.error("Data parsing resulted in inconsistent red and blue ball lists.")
         return None

    # Filter original df to only include valid rows based on the length of parsed lists
    # This assumes successful parsing happens in order of original df rows after sorting/dropping NaNs
    # A safer approach might be to build a new list of dicts for valid rows and create a new DF
    # Let's rebuild the DataFrame from scratch with parsed data for robustness

    parsed_data = []
    original_columns_to_keep = [col for col in df.columns if col not in ['红球', '蓝球']] # Keep original columns except red/blue strings

    for i in range(len(red_balls_list)):
         row_data = {}
         # Add original columns from the i-th successfully parsed row in the original (now filtered) df
         # This relies on the assumption that red_balls_list and blue_balls_list are built
         # in the same order as the rows in the 'df' DataFrame at this point.
         # A more explicit mapping or rebuilding is safer. Let's rebuild explicitly.
         pass # Will rebuild below


    # Rebuild DataFrame from successfully parsed lists
    if not red_balls_list:
         logger.warning("No valid data rows remain after parsing.")
         return None

    # Create a DataFrame from the parsed ball lists
    processed_df = pd.DataFrame(red_balls_list, columns=[f'red{i+1}' for i in range(6)])
    processed_df['blue'] = blue_balls_list

    # Add back other potentially useful original columns (like '期号', '开奖日期')
    # Assuming '期号' was successfully cleaned and aligned
    # Need to ensure the indices align. Since we iterated and skipped, a simple join might not work.
    # The safest way is to store the original index or key ('期号') with the parsed data.
    # Let's modify the parsing loop to store '期号' as well.

    # --- Modified Parsing Loop (incorporating rebuilding logic) ---
    parsed_rows_data = []
    rows_skipped_parsing = 0 # Reset counter for clarity

    for index, row in df.iterrows():
         try:
             red_str = row.get('红球') # Use .get for safety
             blue_val = row.get('蓝球')
             period_val = row.get('期号')

             if not isinstance(red_str, str) or not blue_val or period_val is None:
                  rows_skipped_parsing += 1
                  # logger.debug(f"Skipping row (period check failed) at index {index}") # Too verbose
                  continue # Skip if essential data is missing or wrong type

             # Handle potential non-integer blue ball data early
             try:
                  blue_num = int(blue_val)
                  if not (1 <= blue_num <= 16):
                      rows_skipped_parsing += 1
                      # logger.debug(f"Skipping row {period_val} due to invalid blue ball value: {blue_val}") # Too verbose
                      continue
             except ValueError:
                  rows_skipped_parsing += 1
                  # logger.debug(f"Skipping row {period_val} due to invalid blue ball format: {blue_val}") # Too verbose
                  continue


             # Process red balls after blue ball is validated
             try:
                 reds = sorted([int(x) for x in red_str.split(',')]) # Sorted red balls
                 if len(reds) != 6 or not all(1 <= r <= 33 for r in reds):
                     rows_skipped_parsing += 1
                     # logger.debug(f"Skipping row {period_val} due to incorrect red ball count or values: {reds}") # Too verbose
                     continue

             except ValueError:
                 rows_skipped_parsing += 1
                 # logger.debug(f"Skipping row {period_val} due to invalid red ball number format: {red_str}") # Too verbose
                 continue
             except Exception as e:
                 rows_skipped_parsing += 1
                 logger.warning(f"An unexpected error occurred parsing red balls at period {period_val}: {e}. Skipping.")
                 continue

             # If we reached here, the row is valid. Add to parsed_rows_data.
             row_data = {'期号': int(period_val)} # Ensure期号 is int
             if '开奖日期' in row: # Include '开奖日期' if it exists
                 row_data['开奖日期'] = row['开奖日期']
             # Add sorted red balls and blue ball
             for i in range(6):
                 row_data[f'red{i+1}'] = reds[i]
                 row_data[f'red_pos{i+1}'] = reds[i] # Add positional as well (which is the same after sorting)
             row_data['blue'] = blue_num

             parsed_rows_data.append(row_data)

         except Exception as e:
             rows_skipped_parsing += 1
             period_val = row.get('期号', 'N/A')
             logger.warning(f"An unexpected error occurred processing row (general error) for period {period_val}: {e}. Skipping.")
             continue

    if rows_skipped_parsing > 0:
         logger.warning(f"Skipped {rows_skipped_parsing} rows due to parsing errors.")

    if not parsed_rows_data:
         logger.error("No valid data rows remain after comprehensive parsing.")
         return None

    processed_df = pd.DataFrame(parsed_rows_data)

    # Ensure sorted by 期号 and reset index
    processed_df.sort_values(by='期号', ascending=True, inplace=True)
    processed_df.reset_index(drop=True, inplace=True)


    logger.info(f"Data cleaned and structured. Remaining valid periods: {len(processed_df)}")
    return processed_df


def feature_engineer(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Extracts features from structured data."""
    if df is None or df.empty:
        logger.warning("No data to engineer features.")
        return None

    # Ensure essential columns exist after cleaning/structuring
    red_cols = [f'red{i+1}' for i in range(6)]
    red_pos_cols = [f'red_pos{i+1}' for i in range(6)] # These should be the same as red_cols after sorting
    essential_cols = red_cols + red_pos_cols + ['blue']
    if not all(col in df.columns for col in essential_cols):
        logger.error("Essential columns for feature engineering are missing after cleaning.")
        return None

    df_fe = df.copy() # Work on a copy

    # Red Ball Sum
    df_fe['red_sum'] = df_fe[red_cols].sum(axis=1)

    # Red Ball Span
    df_fe['red_span'] = df_fe[red_cols].max(axis=1) - df_fe[red_cols].min(axis=1)

    # Red Ball Odd/Even Count
    df_fe['red_odd_count'] = df_fe[red_cols].apply(lambda row: sum(x % 2 != 0 for x in row), axis=1)
    df_fe['red_even_count'] = 6 - df_fe['red_odd_count']

    # Red Ball Zone Count
    for zone, (start, end) in RED_ZONES.items():
        df_fe[f'red_{zone}_count'] = df_fe[red_cols].apply(lambda row: sum(start <= x <= end for x in row), axis=1)

    # Consecutive Numbers Count (simple pair count using sorted balls)
    def count_consecutive_pairs(row):
        count = 0
        # Use red_pos_cols which represent sorted positions
        if pd.isna(row[red_pos_cols]).any(): return 0 # Handle potential missing positional data (shouldn't happen after robust clean)
        for i in range(5):
            if row[f'red_pos{i+1}'] + 1 == row[f'red_pos{i+2}']:
                count += 1
        return count

    # Apply only if df_fe is not empty
    if not df_fe.empty:
        df_fe['red_consecutive_pairs'] = df_fe.apply(count_consecutive_pairs, axis=1)
    else:
         df_fe['red_consecutive_pairs'] = pd.Series(dtype=int)


    # Repeat from Previous Period (Red Balls)
    # Need to handle the first row. Add a shift first, then calculate.
    df_fe['prev_reds_str'] = df_fe['red1'].astype(str) + ',' + df_fe['red2'].astype(str) + ',' + df_fe['red3'].astype(str) + ',' + \
                             df_fe['red4'].astype(str) + ',' + df_fe['red5'].astype(str) + ',' + df_fe['red6'].astype(str)
    df_fe['prev_reds_shifted'] = df_fe['prev_reds_str'].shift(1)

    df_fe['red_repeat_count'] = 0 # Initialize
    if len(df_fe) > 1:
        for i in range(1, len(df_fe)):
            prev_reds_str = df_fe.loc[i, 'prev_reds_shifted']
            if pd.notna(prev_reds_str):
                 try:
                     prev_reds = set(int(x) for x in prev_reds_str.split(','))
                     current_reds = set(df_fe.loc[i, red_cols].tolist())
                     df_fe.loc[i, 'red_repeat_count'] = len(prev_reds.intersection(current_reds))
                 except ValueError:
                     # Should not happen with robust cleaning, but as a safeguard
                     logger.warning(f"Error parsing previous red balls string at index {i}.")
                     df_fe.loc[i, 'red_repeat_count'] = 0 # Default to 0 repeats on parse error


    df_fe.drop(columns=['prev_reds_str', 'prev_reds_shifted'], errors='ignore', inplace=True)


    # Blue Ball Features
    df_fe['blue_is_odd'] = df_fe['blue'] % 2 != 0
    df_fe['blue_is_large'] = df_fe['blue'] > 8 # 1-8 Small, 9-16 Large
    # Add blue ball prime status (simplified)
    primes = {2, 3, 5, 7, 11, 13}
    df_fe['blue_is_prime'] = df_fe['blue'].apply(lambda x: x in primes)


    logger.info("Features engineered.")
    return df_fe

# --- Phase 2: Historical Statistics & Pattern Analysis ---

def analyze_frequency_omission(df: pd.DataFrame) -> dict:
    """Analyzes frequency and current omission for each number and position."""
    if df is None or df.empty:
        logger.warning("No data to analyze frequency and omission.")
        return {} # Return empty dict

    red_cols = [f'red{i+1}' for i in range(6)]
    red_pos_cols = [f'red_pos{i+1}' for i in range(6)]
    # Use the *current* index for omission calculation relative to the latest period in this df slice
    most_recent_period_index = len(df) - 1

    if most_recent_period_index < 0: # Should not happen with empty check, but as safeguard
        logger.warning("DataFrame is empty after checks in analyze_frequency_omission.")
        return {}


    all_reds = df[red_cols].values.flatten()
    all_blues = df['blue'].values

    # Frequency (Overall)
    red_freq = Counter(all_reds)
    blue_freq = Counter(all_blues)

    # Positional Red Ball Frequency
    red_pos_freq = {}
    for col in red_pos_cols:
        red_pos_freq[col] = Counter(df[col])


    # Omission (Current) - Number of periods since last seen
    current_omission = {}
    # Omission is calculated relative to the *end* of the provided df.
    # Index `most_recent_period_index` is the latest period.
    # An omission of 0 means it appeared in the latest period (index most_recent_period_index).
    # An omission of 1 means it last appeared at index most_recent_period_index - 1.
    # An omission of k means it last appeared at index most_recent_period_index - k.
    # If never seen, omission is most_recent_period_index + 1 (or length of df)

    # Red Balls (Any position)
    for number in RED_BALL_RANGE:
        # Find the index of the latest appearance *within the provided df*
        latest_appearance_index = df.index[
            (df[red_cols] == number).any(axis=1)
        ].max() # This gives the index *within the current df*

        if pd.isna(latest_appearance_index):
             current_omission[number] = len(df) # Never seen in this data
        else:
             current_omission[number] = most_recent_period_index - latest_appearance_index


    # Red Balls (By Position)
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


    # Blue Balls
    for number in BLUE_BALL_RANGE:
         latest_appearance_index = df.index[
             (df['blue'] == number)
         ].max()
         if pd.isna(latest_appearance_index):
             current_omission[number] = len(df)
         else:
            current_omission[number] = most_recent_period_index - latest_appearance_index


    # Average Interval (Proxy for Average Omission) - calculated over the data provided
    average_interval = {}
    total_periods = len(df)
    for number in RED_BALL_RANGE:
        # Add 1 to frequency to avoid division by zero if number never seen
        average_interval[number] = total_periods / (red_freq.get(number, 0) + 1)
    for number in BLUE_BALL_RANGE:
        average_interval[number] = total_periods / (blue_freq.get(number, 0) + 1)

    # Positional Average Interval
    red_pos_average_interval = {}
    for col in red_pos_cols:
        red_pos_average_interval[col] = {}
        col_freq = red_pos_freq.get(col, {})
        for number in RED_BALL_RANGE:
             red_pos_average_interval[col][number] = total_periods / (col_freq.get(number, 0) + 1)


    # Identify Hot/Cold (based on overall frequency)
    # Handle empty frequency data gracefully
    red_freq_items = sorted(red_freq.items(), key=lambda item: item[1], reverse=True) if red_freq else []
    blue_freq_items = sorted(blue_freq.items(), key=lambda item: item[1], reverse=True) if blue_freq else []


    # Define hot/cold based on top/bottom percentage (ensure thresholds are valid indices)
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

    # Only print this summary during full analysis, not during backtest steps
    # This function is called within the SuppressOutput context during backtest, so print won't show.
    # If calling outside, this would print.

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

    # Optional: Plot Frequencies if SHOW_PLOTS is True and outside suppressed context (or handle plots separately)
    # Plotting logic is moved outside this core analysis function to be controlled by the main script flow and SuppressOutput.

    return analysis_results


def analyze_patterns(df: pd.DataFrame) -> dict:
    """Analyzes distributions of calculated features."""
    if df is None or df.empty:
        logger.warning("No data to analyze patterns.")
        return {} # Return empty dict

    # Pattern analysis results
    pattern_results = {}

    # Red Ball Sum Distribution
    if 'red_sum' in df.columns and not df['red_sum'].empty:
        pattern_results['sum_stats'] = df['red_sum'].describe().to_dict()
        pattern_results['most_common_sum'] = df['red_sum'].mode()[0] if not df['red_sum'].mode().empty else None
    else:
         pattern_results['sum_stats'] = {}
         pattern_results['most_common_sum'] = None

    # Red Ball Span Distribution
    if 'red_span' in df.columns and not df['red_span'].empty:
        pattern_results['span_stats'] = df['red_span'].describe().to_dict()
        pattern_results['most_common_span'] = df['red_span'].mode()[0] if not df['red_span'].mode().empty else None
    else:
        pattern_results['span_stats'] = {}
        pattern_results['most_common_span'] = None


    # Odd/Even Ratio Distribution
    if 'red_odd_count' in df.columns and not df['red_odd_count'].empty:
        odd_even_counts = df['red_odd_count'].value_counts().sort_index()
        pattern_results['odd_even_ratios'] = {f'{odd}:{6-odd}': int(count) for odd, count in odd_even_counts.items()} # Convert numpy int to python int
        pattern_results['most_common_odd_even_count'] = odd_even_counts.idxmax() if not odd_even_counts.empty else None
    else:
        pattern_results['odd_even_ratios'] = {}
        pattern_results['most_common_odd_even_count'] = None

    # Zone Distribution
    zone_cols = [f'red_{zone}_count' for zone in RED_ZONES.keys()]
    if all(col in df.columns for col in zone_cols) and not df.empty:
        zone_counts_df = df[zone_cols]
        if not zone_counts_df.empty:
            # Ensure counts are integers before forming tuples
            zone_counts_df = zone_counts_df.astype(int)
            zone_distribution_counts = zone_counts_df.apply(lambda row: tuple(row), axis=1).value_counts()
            pattern_results['zone_distribution_counts'] = {tuple(int(c) for c in dist): int(count) for dist, count in zone_distribution_counts.items()} # Convert keys/values to python types
            pattern_results['most_common_zone_distribution'] = zone_distribution_counts.index[0] if not zone_distribution_counts.empty else (0, 0, 0)
        else:
            pattern_results['zone_distribution_counts'] = {}
            pattern_results['most_common_zone_distribution'] = (0, 0, 0) # Default
    else:
         pattern_results['zone_distribution_counts'] = {}
         pattern_results['most_common_zone_distribution'] = (0, 0, 0) # Default


    # Consecutive Pairs Distribution
    if 'red_consecutive_pairs' in df.columns and not df['red_consecutive_pairs'].empty:
        consecutive_counts = df['red_consecutive_pairs'].value_counts().sort_index()
        pattern_results['consecutive_counts'] = {int(count): int(freq) for count, freq in consecutive_counts.items()}
    else:
        pattern_results['consecutive_counts'] = {}

    # Repeat from Previous Period Frequency
    if 'red_repeat_count' in df.columns and not df['red_repeat_count'].empty:
        repeat_counts = df['red_repeat_count'].value_counts().sort_index()
        pattern_results['repeat_counts'] = {int(count): int(freq) for count, freq in repeat_counts.items()}
    else:
        pattern_results['repeat_counts'] = {}


    # Blue Ball Pattern Analysis
    if 'blue_is_odd' in df.columns and not df['blue_is_odd'].empty:
        blue_odd_counts = df['blue_is_odd'].value_counts()
        pattern_results['blue_odd_counts'] = {bool(is_odd): int(count) for is_odd, count in blue_odd_counts.items()} # Convert bool key, int value
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

    # Optional: Plot Patterns if SHOW_PLOTS is True
    # Plotting logic is moved outside this core analysis function.

    return pattern_results


def analyze_associations(df: pd.DataFrame, min_support: float = ARM_MIN_SUPPORT, min_confidence: float = ARM_MIN_CONFIDENCE, min_lift: float = ARM_MIN_LIFT) -> pd.DataFrame:
    """Finds frequent itemsets and association rules for red balls."""
    # Need at least 2 periods to find associations
    if df is None or df.empty or len(df) < 2:
        # logger.debug("Not enough data to analyze associations.") # Too verbose during backtest
        return pd.DataFrame() # Return empty DataFrame

    red_cols = [f'red{i+1}' for i in range(6)]
    # Check if red ball columns exist and are not all NaN/empty in the slice
    if not all(col in df.columns for col in red_cols) or df[red_cols].isnull().all().all():
         # logger.debug("Red ball columns missing or empty in data slice for association analysis.") # Too verbose
         return pd.DataFrame()

    # Convert ball numbers to strings for TransactionEncoder (often safer)
    transactions = df[red_cols].astype(str).values.tolist()

    # Filter out empty transactions if any (shouldn't happen after cleaning, but as safeguard)
    transactions = [t for t in transactions if all(item and item != 'nan' for item in t)]
    if not transactions:
         # logger.debug("No valid transactions after filtering for association analysis.") # Too verbose
         return pd.DataFrame()


    te = TransactionEncoder()
    try:
        te_ary = te.fit(transactions).transform(transactions)
        df_onehot = pd.DataFrame(te_ary, columns=te.columns_)
    except Exception as e:
        logger.warning(f"Error during TransactionEncoder transformation for association rules: {e}")
        return pd.DataFrame()

    if df_onehot.empty:
        # logger.debug("One-hot encoded DataFrame is empty for association analysis.") # Too verbose
        return pd.DataFrame()

    try:
        # Adjust min_support based on data size to require a minimum absolute frequency
        # min_support_abs = max(int(min_support * len(df_onehot)), 2) # Require at least 2 occurrences
        # Using relative support as per function signature, but be mindful of data size
        frequent_itemsets = apriori(df_onehot, min_support=min_support, use_colnames=True)

        if frequent_itemsets.empty:
             # logger.debug("No frequent itemsets found with min_support={min_support:.4f}.") # Too verbose
             return pd.DataFrame()

        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    except Exception as e:
        logger.warning(f"Error during Apriori algorithm: {e}")
        return pd.DataFrame()


    try:
        # Generate rules with minimum confidence and lift
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
        if min_confidence is not None: # Apply confidence threshold if specified
             rules = rules[rules['confidence'] >= min_confidence]

        rules.sort_values(by='lift', ascending=False, inplace=True)
    except Exception as e:
        logger.warning(f"Error during association rule generation: {e}")
        return pd.DataFrame()


    # logger.debug(f"Found {len(rules)} association rules (min_support={min_support:.4f}, min_confidence={min_confidence:.2f}, min_lift={min_lift:.2f}).") # Too verbose
    # print("\nTop 10 Rules by Lift:") # Too verbose
    # print(rules.head(10)) # Too verbose

    return rules

# --- ML for Feature Prediction ---

def create_lagged_features(df: pd.DataFrame, lags: List[int]) -> Optional[pd.DataFrame]:
    """Creates lagged features for ML models."""
    if df is None or df.empty or not lags:
         # logger.debug("No data or lags provided to create lagged features.") # Too verbose
         return None

    # Select features to lag - ensure these columns exist
    lag_base_cols = ['red_sum', 'red_span', 'red_odd_count', 'blue_is_odd', 'red_consecutive_pairs', 'red_repeat_count']
    # Filter out columns that might not exist in a small df slice
    existing_lag_cols = [col for col in lag_base_cols if col in df.columns]

    if not existing_lag_cols:
         # logger.debug("Base columns for lagging not found in DataFrame.") # Too verbose
         return None

    df_lagged = df[existing_lag_cols].copy()

    for lag in lags:
        if lag > 0:
            for col in existing_lag_cols:
                 # Use .name to get the original column name
                 df_lagged[f'{col}_lag{lag}'] = df_lagged[col].shift(lag)
        # else: # lag <= 0 is invalid for shift
             # logger.debug(f"Warning: Lag value {lag} is not positive. Skipping.") # Too verbose


    # Drop rows with NaNs introduced by lagging
    # Check if there are any rows left after dropping NaNs
    initial_rows = len(df_lagged)
    df_lagged.dropna(inplace=True)
    if len(df_lagged) < initial_rows:
         # logger.debug(f"Dropped {initial_rows - len(df_lagged)} rows due to NaN values from lagging.") # Too verbose
         pass # No need to log this often during backtest

    if df_lagged.empty:
        # logger.debug("No data left after dropping rows with NaNs from lagging.") # Too verbose
        return None

    # Features include lagged values
    feature_cols = [col for col in df_lagged.columns if any(f'_lag{lag}' in col for lag in lags)]

    # Ensure feature columns actually exist
    feature_cols = [col for col in feature_cols if col in df_lagged.columns]

    if not feature_cols:
         logger.warning("No feature columns created after lagging and dropping NaNs.")
         return None

    # Return DataFrame containing only the created features and the original (non-lagged) base columns
    # The non-lagged base columns in the resulting df_lagged are the targets corresponding to the lagged features.
    # So the returned DF contains X (lagged features) and y (current period's values for those features).
    return df_lagged[feature_cols + existing_lag_cols]


def train_feature_prediction_models(df_train_raw: pd.DataFrame, lags: List[int]) -> Optional[dict]:
    """Trains ML models to predict next period's features."""
    # logger.debug("Training Feature Prediction Models...") # Too verbose in loop

    # Create lagged features from the training data
    df_lagged_prepared = create_lagged_features(df_train_raw.copy(), lags)

    if df_lagged_prepared is None or df_lagged_prepared.empty:
        # logger.debug("Not enough data to create lagged features and train models.") # Too verbose
        return None # Return None if training is not possible


    # Define features (X) and targets (y) from the prepared data
    # Features are the lagged columns
    feature_cols = [col for col in df_lagged_prepared.columns if any(f'_lag{lag}' in col for lag in lags)]
    # Targets are the non-lagged base columns
    target_odd_col = 'red_odd_count'
    target_blue_odd_col = 'blue_is_odd'

    # Ensure target columns exist in the prepared data
    if target_odd_col not in df_lagged_prepared.columns or target_blue_odd_col not in df_lagged_prepared.columns:
         logger.warning("Target columns ('red_odd_count' or 'blue_is_odd') not found after lagging.")
         return None

    X = df_lagged_prepared[feature_cols]
    y_odd = df_lagged_prepared[target_odd_col]
    # Convert boolean target to integer for RandomForestClassifier
    y_blue_odd = df_lagged_prepared[target_blue_odd_col].astype(int)

    if X.empty or y_odd.empty or y_blue_odd.empty:
         # logger.debug("Features or targets are empty after preparation for training.") # Too verbose
         return None


    # --- Train Models ---
    # Use RandomForestClassifiers for classification tasks (odd/even count, blue odd)
    trained_models = {} # Dictionary to store models and feature columns

    try:
        # Train Red Odd Count model
        model_odd_count = RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=42, max_depth=RF_MAX_DEPTH)
        model_odd_count.fit(X, y_odd)
        trained_models['odd_count_model'] = model_odd_count

        # Train Blue Odd model if there are at least two classes in the target
        if len(y_blue_odd.unique()) > 1:
             model_blue_odd = RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=42, max_depth=RF_MAX_DEPTH)
             model_blue_odd.fit(X, y_blue_odd)
             trained_models['blue_odd_model'] = model_blue_odd
        else:
             # logger.debug("Blue Odd target has only one class. Cannot train classifier.") # Too verbose
             pass # blue_odd_model will not be added to trained_models

    except Exception as e:
        logger.warning(f"Error during ML model training: {e}")
        return None # Return None if training fails


    trained_models['feature_cols'] = feature_cols # Store feature columns used for training


    return trained_models


def predict_feature_tendency_ml(df_historical: pd.DataFrame, trained_models: Optional[dict], lags: List[int]) -> Dict:
    """Predicts next period's feature tendencies using trained ML models and latest data."""
    # logger.debug("Predicting Feature Tendency (ML-based)...") # Too verbose in loop

    predicted_tendency = {} # Default to empty dict

    if trained_models is None or df_historical is None or df_historical.empty:
        # logger.debug("ML models not trained or no historical data available for prediction.") # Too verbose
        return predicted_tendency # Return empty dict

    model_odd_count = trained_models.get('odd_count_model')
    model_blue_odd = trained_models.get('blue_odd_model') # This might be None if not trained
    feature_cols = trained_models.get('feature_cols')

    if model_odd_count is None or feature_cols is None or not feature_cols:
         # logger.debug("ML models or feature columns are not available for prediction.") # Too verbose
         return predicted_tendency # Return empty dict

    # Need at least max(lags) + 1 rows to create features for the very last period
    max_lag = max(lags) if lags else 0
    if len(df_historical) < max_lag + 1:
         # logger.debug(f"Not enough historical data ({len(df_historical)} rows) to create lagged features for prediction (need at least {max_lag + 1} rows).") # Too verbose
         return predicted_tendency


    # Create prediction features from the latest relevant history
    # Need the last max(lags) + 1 rows to calculate all lags for the *latest* period (which becomes the single prediction row).
    df_latest_history_for_lagging = df_historical.tail(max_lag + 1).copy()

    # Create lagged features for this small df
    df_predict_prep = create_lagged_features(df_latest_history_for_lagging, lags)

    if df_predict_prep is None or df_predict_prep.empty:
        # logger.debug("Failed to create lagged features for prediction.") # Too verbose
        return predicted_tendency


    # The features for prediction are the *last* row of the prepared lagged features
    # Ensure we select only the feature columns used during training
    predict_df = df_predict_prep[feature_cols].tail(1).copy()


    if predict_df.empty:
        logger.warning("Prediction features DataFrame is empty after preparation.")
        return predicted_tendency

    # Ensure prediction features have the same columns as training features, fill missing with 0
    try:
        predict_df = predict_df.reindex(columns=feature_cols, fill_value=0)
        # Check for NaNs introduced by reindexing or other issues and fill with 0 as a fallback
        if predict_df.isnull().values.any():
             logger.warning("NaN values found in prediction features after reindexing. Filling with 0.")
             predict_df.fillna(0, inplace=True) # Fallback fillna

    except Exception as e:
        logger.warning(f"Error during prediction feature preparation (reindex/fillna): {e}")
        return predicted_tendency


    # Make predictions
    try:
        # Use .iloc[0] to get the prediction for the single row
        predicted_tendency['predicted_odd_count'] = model_odd_count.predict(predict_df).tolist()[0]
    except Exception as e:
        logger.warning(f"Warning: Error predicting Red Odd Count: {e}")


    try:
        # Only predict if blue odd model was successfully trained
        if model_blue_odd is not None:
             # Use .iloc[0] and convert integer prediction back to boolean
             predicted_blue_is_odd_int = model_blue_odd.predict(predict_df).tolist()[0]
             predicted_tendency['predicted_blue_is_odd'] = bool(predicted_blue_is_odd_int)
    except Exception as e:
        logger.warning(f"Warning: Error predicting Blue Odd: {e}")


    # logger.debug(f"ML Predicted Tendencies: {predicted_tendency}") # Too verbose

    return predicted_tendency


# --- Phase 3: Prediction Tendency & Number Scoring ---

def calculate_scores(freq_omission_data: dict, pattern_analysis_data: dict, predicted_tendency: dict) -> dict:
    """Calculates a composite score for each number."""
    # logger.debug("Calculating Number Scores (Advanced)...") # Too verbose in loop

    red_scores = {}
    blue_scores = {}

    # --- Scoring Factors ---
    red_freq = freq_omission_data.get('red_freq', {})
    blue_freq = freq_omission_data.get('blue_freq', {})
    current_omission = freq_omission_data.get('current_omission', {}) # Any position
    average_interval = freq_omission_data.get('average_interval', {}) # Any position

    # Convert frequencies to ranks (handle cases where freq data is empty)
    # Use all possible numbers in the range to create the series for consistent ranking
    red_freq_series = pd.Series(red_freq).reindex(RED_BALL_RANGE, fill_value=0) # Ensure all numbers are included
    red_freq_rank = red_freq_series.rank(method='min', ascending=False) # Rank 1 is highest freq

    blue_freq_series = pd.Series(blue_freq).reindex(BLUE_BALL_RANGE, fill_value=0) # Ensure all numbers are included
    blue_freq_rank = blue_freq_series.rank(method='min', ascending=False)


    # --- Scoring Formula (More factors) ---
    max_red_rank = len(RED_BALL_RANGE)
    max_blue_rank = len(BLUE_BALL_RANGE)

    # Tendency prediction results
    predicted_odd_count = predicted_tendency.get('predicted_odd_count', None)
    predicted_blue_is_odd = predicted_tendency.get('predicted_blue_is_odd', None)

    # Get historical mode tendencies for scoring if ML didn't predict them
    hist_most_common_odd_count = pattern_analysis_data.get('most_common_odd_even_count')
    hist_most_common_zone_dist = pattern_analysis_data.get('most_common_zone_distribution')
    hist_most_common_blue_large = pattern_analysis_data.get('blue_large_counts', {}).get(True, None) is not None and (pattern_analysis_data['blue_large_counts'].get(True, 0) > pattern_analysis_data['blue_large_counts'].get(False, 0)) # Check if True is more frequent
    # Note: hist_most_common_blue_large is now a boolean indicating if large is more common

    # Use ML prediction if available, otherwise fallback to historical mode
    actual_odd_count_tendency = predicted_odd_count if predicted_odd_count is not None else hist_most_common_odd_count
    actual_blue_odd_tendency = predicted_blue_is_odd if predicted_blue_is_odd is not None else (hist_most_common_blue_large is not None and hist_most_common_blue_large) # Use blue_large for blue odd fallback? This seems wrong. Should use blue_odd_counts.
    hist_most_common_blue_odd_val = pattern_analysis_data.get('blue_odd_counts', {}).get(True, None) is not None and (pattern_analysis_data['blue_odd_counts'].get(True, 0) > pattern_analysis_data['blue_odd_counts'].get(False, 0))
    actual_blue_odd_tendency = predicted_blue_is_odd if predicted_blue_is_odd is not None else hist_most_common_blue_odd_val # Corrected blue odd tendency fallback

    actual_zone_dist_tendency = hist_most_common_zone_dist # Assuming ML doesn't predict zones

    # Ensure tendency values are not None before using them in scoring
    use_odd_count_tendency = actual_odd_count_tendency is not None
    use_blue_odd_tendency = actual_blue_odd_tendency is not None
    use_zone_dist_tendency = actual_zone_dist_tendency is not None
    use_blue_size_tendency = hist_most_common_blue_large is not None


    for num in RED_BALL_RANGE:
        # Factor 1: Frequency Rank (Inverse) - Higher frequency gets higher base score
        # Ensure num is in the index of red_freq_rank before calling .get
        freq_score = (max_red_rank - red_freq_rank.get(num, max_red_rank)) / max_red_rank * FREQ_SCORE_WEIGHT

        # Factor 2: Omission Deviation (Reward close to average)
        # Use .get with a default value (e.g., 0) if number not in current_omission or average_interval
        dev = current_omission.get(num, len(RED_BALL_RANGE) * 2) - average_interval.get(num, len(RED_BALL_RANGE) * 2) # Use a large default deviation if not seen
        omission_score = OMISSION_SCORE_WEIGHT * np.exp(-0.005 * dev**2) # Adjust decay rate if needed

        # Factor 3: Tendency Fitting (Odd/Even)
        tendency_score = 0
        if use_odd_count_tendency:
             # Simple bonus: if tendency suggests more odd, odd numbers get bonus; if less odd, even numbers get bonus.
             # Let's use a threshold for odd count (e.g., >= 3 is more odd, < 3 is less odd)
             is_odd_num = num % 2 != 0
             if (is_odd_num and actual_odd_count_tendency >= 3) or (not is_odd_num and actual_odd_count_tendency < 3):
                 tendency_score += ODD_EVEN_TENDENCY_BONUS

        # Factor 4: Tendency Fitting (Zone) - Use historical most common zone
        if use_zone_dist_tendency:
             num_zone_idx = None
             if RED_ZONES['Zone1'][0] <= num <= RED_ZONES['Zone1'][1]: num_zone_idx = 0
             elif RED_ZONES['Zone2'][0] <= num <= RED_ZONES['Zone2'][1]: num_zone_idx = 1
             elif RED_ZONES['Zone3'][0] <= num <= RED_ZONES['Zone3'][1]: num_zone_idx = 2

             # Check if the number's zone count is positive in the most common historical pattern
             if num_zone_idx is not None and num_zone_idx < len(actual_zone_dist_tendency) and actual_zone_dist_tendency[num_zone_idx] > 0:
                  tendency_score += actual_zone_dist_tendency[num_zone_idx] * ZONE_TENDENCY_BONUS_MULTIPLIER


        # Combine factors (weights can be tuned)
        red_scores[num] = freq_score + omission_score + tendency_score


    for num in BLUE_BALL_RANGE:
        # Factor 1: Frequency Rank (Inverse)
        # Ensure num is in the index of blue_freq_rank before calling .get
        freq_score = (max_blue_rank - blue_freq_rank.get(num, max_blue_rank)) / max_blue_rank * BLUE_FREQ_SCORE_WEIGHT

        # Factor 2: Omission Deviation
        # Use .get with a default value (e.g., 0) if number not in current_omission or average_interval
        dev = current_omission.get(num, len(BLUE_BALL_RANGE) * 2) - average_interval.get(num, len(BLUE_BALL_RANGE) * 2) # Use a large default deviation if not seen
        omission_score = BLUE_OMISSION_SCORE_WEIGHT * np.exp(-0.01 * dev**2) # Adjust decay rate if needed

        # Factor 3: Tendency Fitting (Odd/Even)
        tendency_score = 0
        if use_blue_odd_tendency:
            actual_blue_is_odd = num % 2 != 0
            if actual_blue_is_odd == actual_blue_odd_tendency:
                tendency_score += BLUE_ODD_TENDENCY_BONUS

        # Factor 4: Tendency Fitting (Size) - Use historical mode
        if use_blue_size_tendency:
             is_large = num > 8
             # hist_most_common_blue_large is True if large is more common, False otherwise
             if is_large == hist_most_common_blue_large:
                  tendency_score += BLUE_SIZE_TENDENCY_BONUS


        # Combine factors
        blue_scores[num] = freq_score + omission_score + tendency_score


    # Normalize scores to a fixed range (e.g., 0-100)
    all_scores = list(red_scores.values()) + list(blue_scores.values())
    if all_scores: # Avoid min/max on empty list
        min_score, max_score = min(all_scores), max(all_scores)
        # Add tolerance for floating point comparison to handle cases where all scores are identical
        if (max_score - min_score) > 1e-9:
            red_scores = {num: (score - min_score) / (max_score - min_score) * 100 for num, score in red_scores.items()}
            blue_scores = {num: (score - min_score) / (max_score - min_score) * 100 for num, score in blue_scores.items()}
        else: # Handle case where all scores are very close or the same
             red_scores = {num: 50.0 for num in RED_BALL_RANGE}
             blue_scores = {num: 50.0 for num in BLUE_BALL_RANGE}
    else: # If all_scores is empty, return default scores
        red_scores = {num: 50.0 for num in RED_BALL_RANGE}
        blue_scores = {num: 50.0 for num in BLUE_BALL_RANGE}


    # logger.debug("Advanced Scores calculated.") # Too verbose
    # logger.debug(f"Top 15 Red Balls by Score: {sorted(red_scores.items(), key=lambda item: item[1], reverse=True)[:15]}") # Too verbose
    # logger.debug(f"Top 8 Blue Balls by Score: {sorted(blue_scores.items(), key=lambda item: item[1], reverse=True)[:8]}") # Too verbose


    return {'red_scores': red_scores, 'blue_scores': blue_scores}


# --- Phase 4: Number Combination Generation & Filtering ---

def generate_combinations(scores_data: dict, pattern_analysis_data: dict, predicted_tendency: dict, num_combinations: int = NUM_COMBINATIONS_TO_GENERATE) -> tuple[List[Dict], list[str]]:
    """Generates potential combinations based on scores and tendencies.
       Returns a list of combination dictionaries and a list of formatted strings for output.
    """
    # logger.debug(f"Generating {num_combinations} Combinations...") # Too verbose in loop

    red_scores = scores_data.get('red_scores', {})
    blue_scores = scores_data.get('blue_scores', {})
    tendency = predicted_tendency # ML predicted tendencies

    # Select candidate pools based on scores
    # Handle cases where scores are empty or have insufficient numbers
    sorted_red_scores = sorted(red_scores.items(), key=lambda item: item[1], reverse=True) if red_scores else []
    red_candidate_pool = [num for num, score in sorted_red_scores[:TOP_N_RED_FOR_CANDIDATE]]

    sorted_blue_scores = sorted(blue_scores.items(), key=lambda item: item[1], reverse=True) if blue_scores else []
    blue_candidate_pool = [num for num, score in sorted_blue_scores[:TOP_N_BLUE_FOR_CANDIDATE]]

    # Ensure minimum pool sizes for sampling
    if len(red_candidate_pool) < 6:
         logger.warning(f"Red candidate pool size ({len(red_candidate_pool)}) is less than 6. Using full range for red balls.")
         red_candidate_pool = list(RED_BALL_RANGE)
    if len(blue_candidate_pool) < 1:
         logger.warning(f"Blue candidate pool size ({len(blue_candidate_pool)}) is less than 1. Using full range for blue balls.")
         blue_candidate_pool = list(BLUE_BALL_RANGE)


    # Revised Generation Strategy: Generate Many from pool, Score/Rank, Select Top N
    # Generate a pool significantly larger than the final desired number of combinations
    large_pool_size = num_combinations * 500 # Reduced pool size multiplier for potentially faster generation
    if large_pool_size < 100: large_pool_size = 100 # Ensure a minimum pool size

    generated_pool = []
    attempts = 0
    max_attempts_pool = large_pool_size * 10 # Safety limit

    # Calculate probabilities based on scores, handling potential division by zero
    red_weights = np.array([red_scores.get(num, 0) for num in red_candidate_pool])
    red_weights[red_weights < 0] = 0 # Ensure weights are non-negative
    total_red_weight = np.sum(red_weights)
    red_probabilities = red_weights / total_red_weight if total_red_weight > 1e-9 else np.ones(len(red_candidate_pool)) / len(red_candidate_pool) # Uniform probability if total weight is near zero

    blue_weights = np.array([blue_scores.get(num, 0) for num in blue_candidate_pool])
    blue_weights[blue_weights < 0] = 0
    total_blue_weight = np.sum(blue_weights)
    blue_probabilities = blue_weights / total_blue_weight if total_blue_weight > 1e-9 else np.ones(len(blue_candidate_pool)) / len(blue_candidate_pool)


    while len(generated_pool) < large_pool_size and attempts < max_attempts_pool:
         attempts += 1
         try:
              # Sample 6 unique red balls from the candidate pool with calculated probabilities
              sampled_red_balls = sorted(np.random.choice(
                  red_candidate_pool, size=6, replace=False, p=red_probabilities
              ).tolist())

              # Sample 1 blue ball
              sampled_blue_ball = np.random.choice(
                   blue_candidate_pool, size=1, replace=False, p=blue_probabilities
              ).tolist()[0]

              generated_pool.append({'red': sampled_red_balls, 'blue': sampled_blue_ball})

         except ValueError as e:
              # This can happen if probabilities are not properly normalized or sum to near zero
             # logger.debug(f"Warning during weighted sampling ({e}). Falling back to simple random sampling from candidate pool.") # Too verbose
             try:
                  # Fallback to simple random sampling from the candidate pool
                  if len(red_candidate_pool) >= 6:
                     sampled_red_balls = sorted(random.sample(red_candidate_pool, 6))
                  else:
                      # This case should ideally not be reached if initial pool check works, but as a safeguard
                      sampled_red_balls = sorted(random.sample(list(RED_BALL_RANGE), 6)) # Fallback to full range

                  if blue_candidate_pool:
                       sampled_blue_ball = random.choice(blue_candidate_pool)
                  else:
                       sampled_blue_ball = random.choice(list(BLUE_BALL_RANGE)) # Fallback to full range

                  generated_pool.append({'red': sampled_red_balls, 'blue': sampled_blue_ball})
             except ValueError as e_fallback:
                 logger.error(f"Fallback sampling failed: {e_fallback}. Stopping combination generation attempts.")
                 break # Give up if sampling fails even with fallback
         except Exception as e:
             logger.warning(f"An unexpected error occurred during combination sampling attempt {attempts}: {e}. Skipping this attempt.")
             continue # Continue with next attempt


    # logger.debug(f"Generated {len(generated_pool)} candidate combinations.") # Too verbose

    if not generated_pool:
         logger.warning("No combinations were generated.")
         return [], [] # Return empty lists

    # Now, score/rank the generated combinations based on how well they fit number scores AND tendencies
    scored_combinations = []

    # Get historical mode tendencies for combination scoring if ML didn't predict them
    hist_most_common_odd_count = pattern_analysis_data.get('most_common_odd_even_count')
    hist_most_common_zone_dist = pattern_analysis_data.get('most_common_zone_distribution')
    hist_most_common_blue_large = pattern_analysis_data.get('blue_large_counts', {}).get(True, None) is not None and (pattern_analysis_data['blue_large_counts'].get(True, 0) > pattern_analysis_data['blue_large_counts'].get(False, 0))
    hist_most_common_blue_odd_val = pattern_analysis_data.get('blue_odd_counts', {}).get(True, None) is not None and (pattern_analysis_data['blue_odd_counts'].get(True, 0) > pattern_analysis_data['blue_odd_counts'].get(False, 0))

    # Use ML prediction if available, otherwise fallback to historical mode
    actual_odd_count_tendency = tendency.get('predicted_odd_count', hist_most_common_odd_count)
    actual_blue_odd_tendency = tendency.get('predicted_blue_is_odd', hist_most_common_blue_odd_val)
    actual_zone_dist_tendency = hist_most_common_zone_dist # Assuming ML doesn't predict zones
    actual_blue_size_tendency = hist_most_common_blue_large # Using historical mode

    # Ensure tendency values are not None before using them in scoring
    use_odd_count_tendency = actual_odd_count_tendency is not None
    use_blue_odd_tendency = actual_blue_odd_tendency is not None
    use_zone_dist_tendency = actual_zone_dist_tendency is not None
    use_blue_size_tendency = actual_blue_size_tendency is not None


    for combo in generated_pool:
        red_balls = combo['red']
        blue_ball = combo['blue']

        # Calculate a score for the combination
        # Sum of individual ball scores (use .get with default 0 if number not in scores)
        combo_score = sum(scores_data.get('red_scores', {}).get(r, 0) for r in red_balls) + scores_data.get('blue_scores', {}).get(blue_ball, 0)

        # Add bonus based on fitting predicted or historical features
        feature_match_score = 0

        # Red Odd Count Match
        if use_odd_count_tendency:
             actual_odd_count = sum(x % 2 != 0 for x in red_balls)
             if actual_odd_count == actual_odd_count_tendency:
                  feature_match_score += COMBINATION_ODD_COUNT_MATCH_BONUS

        # Blue Odd Match
        if use_blue_odd_tendency:
            actual_blue_is_odd = blue_ball % 2 != 0
            if actual_blue_is_odd == actual_blue_odd_tendency:
                feature_match_score += COMBINATION_BLUE_ODD_MATCH_BONUS

        # Zone Distribution Match (Using historical mode as prediction)
        if use_zone_dist_tendency:
             actual_zone_counts = [0, 0, 0]
             for ball in red_balls:
                 if RED_ZONES['Zone1'][0] <= ball <= RED_ZONES['Zone1'][1]: actual_zone_counts[0] += 1
                 elif RED_ZONES['Zone2'][0] <= ball <= RED_ZONES['Zone2'][1]: actual_zone_counts[1] += 1
                 elif RED_ZONES['Zone3'][0] <= ball <= RED_ZONES['Zone3'][1]: actual_zone_counts[2] += 1
             if tuple(actual_zone_counts) == actual_zone_dist_tendency:
                 feature_match_score += COMBINATION_ZONE_MATCH_BONUS


        # Blue Size Match (Using historical mode as prediction)
        if use_blue_size_tendency:
             is_large = blue_ball > 8
             # actual_blue_size_tendency is True if large is more common, False otherwise
             if is_large == actual_blue_size_tendency:
                  feature_match_score += COMBINATION_BLUE_SIZE_MATCH_BONUS


        # Combine individual score and feature match score
        total_combo_score = combo_score + feature_match_score

        scored_combinations.append({'combination': combo, 'score': total_combo_score})

    # Sort combinations by score and select the top N
    scored_combinations.sort(key=lambda x: x['score'], reverse=True)
    final_recommendations_data = scored_combinations[:num_combinations]

    # --- Format output strings ---
    output_strings = []
    output_strings.append("Recommended Combinations:")
    if final_recommendations_data:
         for i, rec in enumerate(final_recommendations_data):
             output_strings.append(f"Combination {i+1}: 红球 {sorted(rec['combination']['red'])} 蓝球 {rec['combination']['blue']} (Score: {rec['score']:.2f})")
    else:
         output_strings.append("Could not generate recommended combinations.")


    # Return the list of combination dictionaries AND the formatted output strings
    return final_recommendations_data, output_strings

# --- Core Analysis and Recommendation Function (New) ---
# This function encapsulates the main logic flow for a given dataset slice
def analyze_and_recommend(
    df_historical: pd.DataFrame,
    lags: List[int],
    num_combinations: int,
    train_ml: bool = True, # Whether to train ML models in this run
    existing_models: Optional[Dict] = None # Pass existing models if train_ml is False
) -> tuple[List[Dict], list[str], dict, Optional[Dict]]:
    """
    Performs analysis, predicts tendencies, calculates scores, and generates combinations
    for the next period based on the provided historical data.
    Optionally trains ML models or uses existing ones.

    Returns: (recommendations_data, recommendations_strings, analysis_data, trained_models)
    """
    if df_historical is None or df_historical.empty:
        logger.error("No historical data provided for analysis and recommendation.")
        return [], [], {}, None

    # 1. Perform Historical Analysis (Frequency, Omission, Patterns)
    # This analysis is based on the provided df_historical slice
    freq_omission_data = analyze_frequency_omission(df_historical)
    pattern_analysis_data = analyze_patterns(df_historical)
    # association_rules_data = analyze_associations(df_historical, ARM_MIN_SUPPORT, ARM_MIN_CONFIDENCE, ARM_MIN_LIFT) # Not used in scoring currently

    analysis_data = {
        'freq_omission': freq_omission_data,
        'patterns': pattern_analysis_data,
        # 'association_rules': association_rules_data # Include if needed later
    }

    # 2. Predict Feature Tendency (using ML if trained/provided, otherwise rely on historical modes)
    current_trained_models = None
    predicted_tendency = {}

    if train_ml:
        # Train ML models using the provided historical data
        current_trained_models = train_feature_prediction_models(df_historical, lags)
        if current_trained_models:
            # Predict tendencies using the newly trained models and the latest history
            predicted_tendency = predict_feature_tendency_ml(df_historical, current_trained_models, lags)
        else:
             logger.warning("ML model training failed. Relying solely on historical patterns for scoring tendencies.")
             # predicted_tendency remains {}

    elif existing_models:
        # Use existing trained models to predict tendencies on the latest data
        current_trained_models = existing_models # Use the provided models
        predicted_tendency = predict_feature_tendency_ml(df_historical, current_trained_models, lags)
        if not predicted_tendency:
             logger.warning("ML prediction failed with existing models. Relying solely on historical patterns for scoring tendencies.")
             # predicted_tendency remains {}

    else:
         logger.info("ML training skipped and no existing models provided. Relying solely on historical patterns for scoring tendencies.")
         # predicted_tendency remains {}


    # 3. Calculate Number Scores (incorporating historical analysis and predicted tendency)
    scores_data = calculate_scores(
        freq_omission_data,
        pattern_analysis_data,
        predicted_tendency # This dict might be empty if ML failed
    )

    # 4. Generate Combinations based on scores and tendencies
    recommendations_data, recommendations_strings = generate_combinations(
        scores_data,
        pattern_analysis_data, # Pass pattern data for combination scoring fallback
        predicted_tendency, # Pass predicted tendency for combination scoring
        num_combinations=num_combinations
    )

    return recommendations_data, recommendations_strings, analysis_data, current_trained_models


# --- Phase 5: Validation, Backtesting & Continuous Optimization ---

def backtest(df: pd.DataFrame, lags: List[int], num_combinations_per_period: int, backtest_periods_count: int) -> pd.DataFrame:
    """
    Performs backtesting on historical data, including retraining ML models.
    """
    logger.info("\n" + "="*50)
    logger.info(" STARTING BACKTESTING ")
    logger.info("="*50)

    # Need enough history for initial lags and analysis before the first backtest period
    # We need max_lag periods for the features, plus 1 period for the target, plus 1 period
    # as the target of the first prediction.
    # So, to predict period index K, we need data up to index K-1.
    # To calculate lags for index K-1, we need data up to index K-1 - max_lag.
    # Thus, the minimum data needed to make the *first* prediction (for period index start_index)
    # is data up to index start_index - 1, which requires data from index start_index - 1 - max_lag.
    # Total periods needed before the first prediction: (start_index - 1) - (start_index - 1 - max_lag) + 1 = max_lag + 1.
    # So, the first period we can *predict* is at index `max_lag`.
    # We need data up to index `max_lag - 1` to make this prediction.
    # Total data points needed up to the first prediction's analysis period (index max_lag - 1) is max_lag.
    # But we need enough data for analysis (frequency, omission, etc.) which might require more than just max_lag periods.
    # Let's ensure we have at least enough periods for the max lag + a buffer for initial analysis stability.
    # A simple approach: ensure enough data exists for the *first* training/analysis set.
    # The first period we predict is at index `start_index`. We train/analyze on data up to `start_index - 1`.
    # To train ML for `start_index - 1`, we need data up to `start_index - 1`, with lags going back to `start_index - 1 - max_lag`.
    # So, the earliest data index needed is `start_index - 1 - max_lag`. This index must be >= 0.
    # `start_index - 1 - max_lag >= 0` => `start_index >= max_lag + 1`.
    # The minimum `start_index` (the index of the first period *to predict*) is `max_lag + 1`.
    # However, for stable initial analysis (frequency etc.), more history is better.
    # Let's use `max(max_lag + 1, some_minimum_analysis_periods)` as the first prediction period index.
    # For simplicity and assuming typical dataset sizes, let's start prediction after enough periods for lags + a small buffer.
    # Let's say we need at least `max_lag + 10` periods for initial analysis before making the first prediction.
    min_periods_for_initial_analysis = max(max(lags) if lags else 0, 1) + 10 # Need at least lags + a buffer
    if len(df) < min_periods_for_initial_analysis + 1: # Need data up to the period before the first prediction
         logger.warning(f"Not enough data ({len(df)}) for backtesting with initial analysis buffer (need at least {min_periods_for_initial_analysis + 1} periods). Skipping backtest.")
         logger.info("="*50)
         logger.info(" BACKTESTING SKIPPED ")
         logger.info("="*50)
         return pd.DataFrame() # Return empty DataFrame

    # Determine the range of periods to backtest over
    # Start predicting from the period *after* the initial analysis data ends.
    # The index of the first period to predict is `start_prediction_index`.
    # We need data up to `start_prediction_index - 1` for analysis and training.
    # The amount of history used for analysis/training before the first prediction is `min_periods_for_initial_analysis`.
    start_prediction_index = min_periods_for_initial_analysis
    end_prediction_index = len(df) - 1 # Predict up to the second to last period (last period is for actual validation)

    if start_prediction_index >= end_prediction_index + 1:
         logger.warning(f"Not enough data remaining after initial analysis history ({min_periods_for_initial_analysis} periods) for backtesting. Start index: {start_prediction_index}, end index: {end_prediction_index}. Skipping backtest.")
         logger.info("="*50)
         logger.info(" BACKTESTING SKIPPED ")
         logger.info("="*50)
         return pd.DataFrame()

    # Adjust the actual number of periods to backtest based on available data and requested count
    # We can backtest predictions for periods from `start_prediction_index` up to `end_prediction_index`.
    available_backtest_periods = end_prediction_index - start_prediction_index + 1
    actual_backtest_periods_count = min(backtest_periods_count, available_backtest_periods)

    # Calculate the index of the *first* period whose result we evaluate.
    # This period's index is `start_prediction_index`. We use data up to `start_prediction_index - 1` to predict it.
    backtest_start_period_index = end_prediction_index - actual_backtest_periods_count + 1
    if backtest_start_period_index < start_prediction_index:
        backtest_start_period_index = start_prediction_index # Ensure we don't go before the minimum required history allows


    logger.info(f"Backtesting predictions for {actual_backtest_periods_count} periods.")
    logger.info(f"Predicting periods from index {backtest_start_period_index} to {end_prediction_index}.")
    logger.info(f"Using data up to index {backtest_start_period_index - 1} for the first prediction's analysis/training.")


    results = []
    red_cols = [f'red{i+1}' for i in range(6)]

    # Display initial progress bar
    total_steps = end_prediction_index - backtest_start_period_index + 1
    show_progress(0, total_steps, prefix='Backtesting Progress:', suffix='Complete', length=50)

    # Iterate through periods starting from the calculated index (this is the index of the period being PREDICTED)
    for i in range(backtest_start_period_index, end_prediction_index + 1):
        # Update progress bar
        current_progress = i - backtest_start_period_index + 1
        show_progress(current_progress, total_steps, prefix='Backtesting Progress:', suffix='Complete', length=50)

        # Data available *up to* the period *before* the one being predicted (index i-1)
        # We need data up to index i-1 for analysis and training for period i
        train_data = df.iloc[:i].copy()

        if train_data.empty or len(train_data) < (max(lags) if lags else 0) + 1:
             # This should ideally not happen if backtest_start_period_index is calculated correctly,
             # but as a safeguard, skip if training data is insufficient for lagging.
             logger.warning(f"Insufficient training data ({len(train_data)} rows) for prediction of period index {i}. Skipping.")
             continue


        # Actual winning numbers for the *current* period being predicted (index i)
        actual_row_index = i
        actual_period = df.loc[actual_row_index, '期号']

        # Ensure actual results are available
        if actual_row_index not in df.index:
             logger.error(f"Actual results for period index {actual_row_index} (期号: {actual_period}) not found in DataFrame. Skipping prediction for this period.")
             continue # Skip this period if actual results are missing

        try:
            actual_red = set(df.loc[actual_row_index, red_cols].tolist())
            actual_blue = df.loc[actual_row_index, 'blue']
        except KeyError as e:
             logger.error(f"Missing red or blue ball data for actual results at period index {actual_row_index} (期号: {actual_period}): {e}. Skipping prediction.")
             continue # Skip if red/blue actuals are missing


        # --- Perform analysis, scoring, and prediction based on data up to period i-1 ---
        # Use SuppressOutput to hide the internal print/logger.info calls during backtest
        with SuppressOutput(suppress_stdout=True, capture_stderr=True):
             predicted_combinations_data, predicted_combinations_strings, analysis_data, trained_models = analyze_and_recommend(
                 train_data, # Use data up to the period before
                 lags,
                 num_combinations=num_combinations_per_period,
                 train_ml=True # Retrain ML models for each period in backtest
             )

        # --- Evaluate Predictions against Actual Results (period i) ---
        if predicted_combinations_data:
            for combo_info in predicted_combinations_data:
                predicted_red = set(combo_info['combination']['red'])
                predicted_blue = combo_info['combination']['blue']

                # Red ball hits
                red_hits = len(predicted_red.intersection(actual_red))

                # Blue ball hit
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
            # If no combinations were generated for this period
            logger.warning(f"No combinations generated for period {actual_period}.")

    # Ensure the final progress bar is displayed at 100%
    show_progress(total_steps, total_steps, prefix='Backtesting Progress:', suffix='Complete', length=50)

    logger.info("="*50)
    logger.info(" BACKTESTING COMPLETE ")
    logger.info("="*50)

    if not results:
        logger.warning("No backtest results recorded.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # --- Backtest Summary ---
    logger.info("\n--- Backtest Summary ---")
    periods_with_results = results_df['period'].nunique()
    logger.info(f"Total periods tested (with combinations generated): {periods_with_results}")
    logger.info(f"Total combinations generated: {len(results_df)}")
    if periods_with_results > 0:
        logger.info(f"Combinations generated per period (average): {len(results_df) / periods_with_results:.2f}")
    else:
        logger.warning("No periods had combinations generated to summarize.")
        return results_df # Return empty or partial results_df if no combos were generated

    # Average red hits per combination
    avg_red_hits = results_df['red_hits'].mean()
    logger.info(f"Average red ball hits per combination: {avg_red_hits:.2f}")

    # Percentage of periods where blue ball was hit by at least one combination
    blue_hit_by_period = results_df.groupby('period')['blue_hit'].any()
    blue_hit_rate_per_period = blue_hit_by_period.mean() if not blue_hit_by_period.empty else 0.0
    logger.info(f"Percentage of tested periods where blue ball was hit by at least one combination: {blue_hit_rate_per_period:.2%}")

    # Frequency of Red Ball Hits per Combination
    red_hit_counts = results_df['red_hits'].value_counts().sort_index()
    logger.info("Frequency of Red Ball Hits per Combination:")
    for hits, count in red_hit_counts.items():
         logger.info(f"  Exactly {hits} Red Hits: {count}")


    # Check hit rates for typical winning tiers (per combination)
    logger.info("\nWinning Tier Hits (per combination):")
    logger.info(f"  6 Red + Blue: {len(results_df[(results_df['red_hits'] == 6) & (results_df['blue_hit'] == True)])}")
    logger.info(f"  6 Red (no Blue): {len(results_df[(results_df['red_hits'] == 6) & (results_df['blue_hit'] == False)])}")
    logger.info(f"  5 Red + Blue: {len(results_df[(results_df['red_hits'] == 5) & (results_df['blue_hit'] == True)])}")
    logger.info(f"  5 Red (no Blue): {len(results_df[(results_df['red_hits'] == 5) & (results_df['blue_hit'] == False)])}")
    logger.info(f"  4 Red + Blue: {len(results_df[(results_df['red_hits'] == 4) & (results_df['blue_hit'] == True)])}")
    logger.info(f"  4 Red (no Blue): {len(results_df[(results_df['red_hits'] == 4) & (results_df['blue_hit'] == False)])}")
    logger.info(f"  3 Red + Blue: {len(results_df[(results_df['red_hits'] == 3) & (results_df['blue_hit'] == True)])}")
    logger.info(f"  Exact Blue Hit (any Red count): {(results_df['blue_hit'] == True).sum()}")


    # Compare to random chance (approximate)
    logger.info("\nComparison to Random Chance (approximate):")
    expected_avg_red_hits_random = 6 * (6/33.0)
    expected_blue_hits_random = 1/16.0
    logger.info(f"  Expected average red ball hits per combination by pure chance: ~{expected_avg_red_hits_random:.2f}")
    logger.info(f"  Expected blue ball hits per combination by pure chance: ~{expected_blue_hits_random:.4f}")

    # Optional: Analyze hits by combination score ranges? (More complex)


    return results_df


# --- Plotting Function (Moved outside analysis functions) ---
def plot_analysis_results(freq_omission_data: dict, pattern_analysis_data: dict):
     """Generates plots from analysis results."""
     if not SHOW_PLOTS:
          plt.close('all') # Close any lingering figures
          return

     logger.info("Generating plots...")

     # Check if data is available before plotting
     if not freq_omission_data or not pattern_analysis_data:
          logger.warning("Analysis data not available for plotting.")
          return

     # Frequency Plots
     red_freq = freq_omission_data.get('red_freq', {})
     blue_freq = freq_omission_data.get('blue_freq', {})
     red_pos_freq = freq_omission_data.get('red_pos_freq', {})
     red_pos_cols = [f'red_pos{i+1}' for i in range(6)] # Assuming these keys exist in red_pos_freq structure

     if red_freq or blue_freq:
          plt.figure(figsize=(14, 6))
          if red_freq:
              plt.subplot(1, 2, 1)
              sns.barplot(x=list(red_freq.keys()), y=list(red_freq.values()))
              plt.title('Red Ball Overall Frequency')
              plt.xlabel('Number'); plt.ylabel('Frequency')
          if blue_freq:
              plt.subplot(1, 2, 2)
              sns.barplot(x=list(blue_freq.keys()), y=list(blue_freq.values()))
              plt.title('Blue Ball Frequency')
              plt.xlabel('Number'); plt.ylabel('Frequency')
          plt.tight_layout()
          plt.show()

     # Positional Red Ball Frequency Plots
     if red_pos_freq and any(red_pos_freq.values()): # Check if red_pos_freq is not empty and has data
          fig, axes = plt.subplots(2, 3, figsize=(15, 10))
          axes = axes.flatten()
          for i, col in enumerate(red_pos_cols):
               if col in red_pos_freq and red_pos_freq[col]:
                  # Sort keys for consistent plotting order
                  sorted_freq_items = sorted(red_pos_freq[col].items())
                  sns.barplot(x=[item[0] for item in sorted_freq_items], y=[item[1] for item in sorted_freq_items], ax=axes[i])
                  axes[i].set_title(f'Red Ball Position {i+1} Frequency')
                  axes[i].set_xlabel('Number')
                  axes[i].set_ylabel('Frequency')
               else:
                  axes[i].set_title(f'Red Ball Position {i+1} Frequency (No Data)')
                  axes[i].set_xlabel('Number')
                  axes[i].set_ylabel('Frequency')
          plt.tight_layout()
          plt.show()

     # Pattern Distribution Plots (Sum, Span, Odd/Even, Consecutive, Repeat)
     # Check for key existence and non-empty data before plotting
     if 'sum_stats' in pattern_analysis_data and pattern_analysis_data['sum_stats']:
          # Assuming 'red_sum' column was present in the original DataFrame passed to analyze_patterns
          # Need access to the original DataFrame or pass the data explicitly for plotting distributions
          # Re-plotting distributions from stats is not meaningful. Need the actual data column.
          # Let's skip plotting distributions here for simplicity or require the original df be passed.
          # Or, pass the relevant series from the original df to this function.
          pass # Skipping distribution plots for now, or assuming the original df is available if needed

     # Example: Plotting Odd/Even Ratio distribution if data available
     odd_even_ratios = pattern_analysis_data.get('odd_even_ratios', {})
     if odd_even_ratios:
          plt.figure(figsize=(6, 4))
          sns.barplot(x=list(odd_even_ratios.keys()), y=list(odd_even_ratios.values()))
          plt.title('Red Ball Odd:Even Ratio Distribution')
          plt.xlabel('Odd:Even Ratio'); plt.ylabel('Frequency'); plt.show()

     # Example: Plotting Consecutive Pairs distribution if data available
     consecutive_counts = pattern_analysis_data.get('consecutive_counts', {})
     if consecutive_counts:
          plt.figure(figsize=(6, 4))
          sns.barplot(x=list(consecutive_counts.keys()), y=list(consecutive_counts.values()))
          plt.title('Red Ball Consecutive Pairs Distribution')
          plt.xlabel('Number of Consecutive Pairs'); plt.ylabel('Frequency'); plt.show()

     # Example: Plotting Repeat Count distribution if data available
     repeat_counts = pattern_analysis_data.get('repeat_counts', {})
     if repeat_counts:
          plt.figure(figsize=(6, 4))
          sns.barplot(x=list(repeat_counts.keys()), y=list(repeat_counts.values()))
          plt.title('Red Ball Repeat from Previous Period Frequency')
          plt.xlabel('Number of Repeated Balls'); plt.ylabel('Frequency'); plt.show()


     logger.info("Plot generation complete.")


# --- Main Execution Flow ---

if __name__ == "__main__":
    # --- Configure Output File ---
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(SCRIPT_DIR, f"ssq_analysis_output_{timestamp}.txt")

    output_file = None # Initialize file handle

    try:
        # Open the output file
        output_file = open(output_filename, 'w', encoding='utf-8')
        # Redirect stdout to the output file for the main report content
        sys.stdout = output_file

        print(f"--- Shuangseqiu Analysis Report ---", file=sys.stdout)
        print(f"Run Date: {now.strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stdout)
        print(f"Output File: {output_filename}", file=sys.stdout)
        print("-" * 30, file=sys.stdout)
        print("\n", file=sys.stdout) # Add some spacing

        # --- Start Analysis and Prediction ---

        # 1. Update Data
        # Use logger for console output during data update
        # No need to suppress this part
        update_csv_with_latest_data(CSV_FILE_PATH)


        # 2. Load and Prepare Data
        # Use SuppressOutput for potentially noisy data loading/cleaning functions
        with SuppressOutput(suppress_stdout=False, capture_stderr=True): # Allow stdout to go to file, capture stderr
            df = load_data(CSV_FILE_PATH)
            if df is not None and not df.empty:
                df = clean_and_structure(df)
                if df is not None and not df.empty:
                     df = feature_engineer(df)
                     if df is None or df.empty:
                          logger.error("Feature engineering failed or resulted in empty data.") # Log to console/default stderr
                          print("\nError: Feature engineering failed. Cannot proceed with analysis.", file=sys.stdout) # Print to file
                else:
                    logger.error("Data cleaning and structuring failed or resulted in empty data.") # Log to console/default stderr
                    print("\nError: Data cleaning and structuring failed. Cannot proceed with analysis.", file=sys.stdout) # Print to file
            else:
                logger.error("Data loading failed or resulted in empty data.") # Log to console/default stderr
                print("\nError: Data loading failed. Cannot proceed with analysis.", file=sys.stdout) # Print to file


        if df is not None and not df.empty:
            # Check if there's enough data after feature engineering for the minimum analysis buffer and lags
            max_lag = max(ML_LAG_FEATURES) if ML_LAG_FEATURES else 0
            min_periods_needed = max(max_lag + 1, 10) # Need at least lags + 1 for prediction base, plus a buffer for analysis
            if len(df) < min_periods_needed:
                 logger.error(f"Not enough valid periods ({len(df)}) after cleaning/feature engineering for analysis with current lag settings and analysis buffer (need at least {min_periods_needed}).")
                 print(f"\nError: Not enough valid periods ({len(df)}) after cleaning/feature engineering for analysis (need at least {min_periods_needed}). Cannot proceed with analysis.", file=sys.stdout)
            else:
                # 3. Perform Full Historical Analysis (on the full dataset for general insights)
                print("\n" + "="*50, file=sys.stdout)
                print(" FULL HISTORICAL ANALYSIS ", file=sys.stdout)
                print("="*50, file=sys.stdout)
                # Use SuppressOutput to hide the internal analysis function prints from the file, but still log stderr
                with SuppressOutput(suppress_stdout=True, capture_stderr=True):
                    full_freq_omission_data = analyze_frequency_omission(df)
                    full_pattern_analysis_data = analyze_patterns(df)
                    full_association_rules = analyze_associations(df, ARM_MIN_SUPPORT, ARM_MIN_CONFIDENCE, ARM_MIN_LIFT) # Analyze associations on full data

                print("\nHistorical Analysis Summary (based on full data):", file=sys.stdout)
                print("\nFrequency and Omission Highlights:", file=sys.stdout)
                # Print selected frequency/omission data to file
                print(f"  Hot Red Balls: {full_freq_omission_data.get('hot_reds', [])}", file=sys.stdout)
                print(f"  Cold Red Balls: {full_freq_omission_data.get('cold_reds', [])}", file=sys.stdout)
                print(f"  Hot Blue Balls: {full_freq_omission_data.get('hot_blues', [])}", file=sys.stdout)
                print(f"  Cold Blue Balls: {full_freq_omission_data.get('cold_blues', [])}", file=sys.stdout)
                print("\nPattern Analysis Highlights:", file=sys.stdout)
                print(f"  Most Common Red Odd:Even Count: {full_pattern_analysis_data.get('most_common_odd_even_count')}", file=sys.stdout)
                print(f"  Most Common Zone Distribution (Zone1:Zone2:Zone3): {full_pattern_analysis_data.get('most_common_zone_distribution')}", file=sys.stdout)
                print(f"  Most Common Red Sum: {full_pattern_analysis_data.get('most_common_sum')}", file=sys.stdout)
                print(f"  Most Common Red Span: {full_pattern_analysis_data.get('most_common_span')}", file=sys.stdout)

                if not full_association_rules.empty:
                    print("\nTop 10 Association Rules (by Lift):", file=sys.stdout)
                    # Format rules for file output
                    for _, rule in full_association_rules.head(10).iterrows():
                        print(f"  {set(rule['antecedents'])} -> {set(rule['consequents'])} (Support: {rule['support']:.4f}, Confidence: {rule['confidence']:.2f}, Lift: {rule['lift']:.2f})", file=sys.stdout)
                else:
                    print("\nNo significant association rules found with current thresholds.", file=sys.stdout)


                print("="*50, file=sys.stdout)
                print(" HISTORICAL ANALYSIS COMPLETE ", file=sys.stdout)
                print("="*50, file=sys.stdout)

                # 4. Perform Backtesting
                # Backtest output is logged to console and summarized in the file later
                # The backtest function itself handles its console output (progress bar) and internal logging
                # No need to redirect sys.stdout here as backtest logs directly or uses show_progress
                print("\n", file=sys.stdout) # Add spacing before backtest summary in file
                # backtest function prints its own start/end headers and progress
                backtest_results = backtest(df, ML_LAG_FEATURES, NUM_COMBINATIONS_TO_GENERATE, BACKTEST_PERIODS_COUNT)

                print("\n" + "="*50, file=sys.stdout)
                print(" BACKTESTING SUMMARY ", file=sys.stdout)
                print("="*50, file=sys.stdout)
                if not backtest_results.empty:
                     # Print backtest summary to file
                     periods_with_results = backtest_results['period'].nunique()
                     print(f"Total periods tested (with combinations generated): {periods_with_results}", file=sys.stdout)
                     print(f"Total combinations generated: {len(backtest_results)}", file=sys.stdout)
                     if periods_with_results > 0:
                          print(f"Combinations generated per period (average): {len(backtest_results) / periods_with_results:.2f}", file=sys.stdout)

                     avg_red_hits = backtest_results['red_hits'].mean()
                     print(f"Average red ball hits per combination: {avg_red_hits:.2f}", file=sys.stdout)

                     blue_hit_by_period = backtest_results.groupby('period')['blue_hit'].any()
                     blue_hit_rate_per_period = blue_hit_by_period.mean() if not blue_hit_by_period.empty else 0.0
                     print(f"Percentage of tested periods where blue ball was hit by at least one combination: {blue_hit_rate_per_period:.2%}", file=sys.stdout)

                     print("\nWinning Tier Hits (per combination):", file=sys.stdout)
                     print(f"  6 Red + Blue: {len(backtest_results[(backtest_results['red_hits'] == 6) & (backtest_results['blue_hit'] == True)])}", file=sys.stdout)
                     print(f"  6 Red (no Blue): {len(backtest_results[(backtest_results['red_hits'] == 6) & (backtest_results['blue_hit'] == False)])}", file=sys.stdout)
                     print(f"  5 Red + Blue: {len(backtest_results[(backtest_results['red_hits'] == 5) & (backtest_results['blue_hit'] == True)])}", file=sys.stdout)
                     print(f"  5 Red (no Blue): {len(backtest_results[(backtest_results['red_hits'] == 5) & (backtest_results['blue_hit'] == False)])}", file=sys.stdout)
                     print(f"  4 Red + Blue: {len(backtest_results[(backtest_results['red_hits'] == 4) & (backtest_results['blue_hit'] == True)])}", file=sys.stdout)
                     print(f"  4 Red (no Blue): {len(backtest_results[(backtest_results['red_hits'] == 4) & (backtest_results['blue_hit'] == False)])}", file=sys.stdout)
                     print(f"  3 Red + Blue: {len(backtest_results[(backtest_results['red_hits'] == 3) & (backtest_results['blue_hit'] == True)])}", file=sys.stdout)
                     print(f"  Exact Blue Hit (any Red count): {(backtest_results['blue_hit'] == True).sum()}", file=sys.stdout)

                     print("\nComparison to Random Chance (approximate):", file=sys.stdout)
                     expected_avg_red_hits_random = 6 * (6/33.0)
                     expected_blue_hits_random = 1/16.0
                     print(f"  Expected average red ball hits per combination by pure chance: ~{expected_avg_red_hits_random:.2f}", file=sys.stdout)
                     print(f"  Expected blue ball hits per combination by pure chance: ~{expected_blue_hits_random:.4f}", file=sys.stdout)

                else:
                     print("No backtest results available to summarize.", file=sys.stdout)

                print("="*50, file=sys.stdout)
                print(" BACKTESTING SUMMARY COMPLETE ", file=sys.stdout)
                print("="*50, file=sys.stdout)

                # 5. Generate Final Recommended Combinations for the Next Period
                print("\n" + "="*50, file=sys.stdout)
                print(" GENERATING FINAL RECOMMENDATIONS ", file=sys.stdout)
                print("="*50, file=sys.stdout)

                # Use SuppressOutput to hide internal prints/logs of analyze_and_recommend from the file
                # The final recommendation strings will be explicitly printed below
                with SuppressOutput(suppress_stdout=True, capture_stderr=True):
                     final_recommendations_data, final_recommendations_strings, final_analysis_data, final_trained_models = analyze_and_recommend(
                         df, # Use the full available data for the final prediction
                         ML_LAG_FEATURES,
                         NUM_COMBINATIONS_TO_GENERATE,
                         train_ml=True # Train models one last time on full data for final prediction
                     )

                # Print final recommended combinations to the output file
                if final_recommendations_strings:
                    for line in final_recommendations_strings:
                        print(line, file=sys.stdout)
                else:
                    print("Could not generate final recommended combinations.", file=sys.stdout)


                print("="*50, file=sys.stdout)
                print(" FINAL RECOMMENDATIONS COMPLETE ", file=sys.stdout)
                print("="*50, file=sys.stdout)


                # 6. Generate 7+7 Multi-bet Selection
                print("\n" + "="*50, file=sys.stdout)
                print(" 7+7 MULTI-BET SELECTION ", file=sys.stdout)
                print("="*50, file=sys.stdout)

                final_scores_data = calculate_scores(
                     final_analysis_data.get('freq_omission', {}),
                     final_analysis_data.get('patterns', {}),
                     {} # 使用空字典代替未定义的变量
                )


                red_scores_for_7_7 = final_scores_data.get('red_scores', {})
                blue_scores_for_7_7 = final_scores_data.get('blue_scores', {})

                if not red_scores_for_7_7 or len(red_scores_for_7_7) < 7 or not blue_scores_for_7_7 or len(blue_scores_for_7_7) < 7:
                     logger.error("Not enough scored numbers to select 7 red and 7 blue for 7+7 multi-bet.") # Log to console/default stderr
                     print("Cannot generate 7+7 multi-bet selection.", file=sys.stdout)
                else:
                     # Sort scores and select top 7 red and top 7 blue
                     sorted_red_scores = sorted(red_scores_for_7_7.items(), key=lambda item: item[1], reverse=True)
                     top_7_red_balls = [num for num, score in sorted_red_scores[:7]]

                     sorted_blue_scores = sorted(blue_scores_for_7_7.items(), key=lambda item: item[1], reverse=True)
                     top_7_blue_balls = [num for num, score in sorted_blue_scores[:7]]

                     # Print to file
                     print("Based on overall scores, select the following for a 7+7 multi-bet:", file=sys.stdout)
                     print(f"Selected 7 Red Balls: {sorted(top_7_red_balls)}", file=sys.stdout)
                     print(f"Selected 7 Blue Balls: {sorted(top_7_blue_balls)}", file=sys.stdout)
                     print("\nThis 7+7 selection covers C(7,6) * C(7,1) = 49 combinations.", file=sys.stdout)
                     print("Consider how these numbers fit historical patterns and your risk tolerance.", file=sys.stdout)

                     # Also print selected 7+7 to console for immediate feedback (use logger)
                     logger.info("\n--- 7+7 Multi-bet Selection ---")
                     logger.info("Based on overall scores, selected the following for a 7+7 multi-bet:")
                     logger.info(f"Selected 7 Red Balls: {sorted(top_7_red_balls)}")
                     logger.info(f"Selected 7 Blue Balls: {sorted(top_7_blue_balls)}")
                     logger.info("This 7+7 selection covers 49 combinations.")

                print("="*50, file=sys.stdout)
                print(" 7+7 SELECTION COMPLETE ", file=sys.stdout)
                print("="*50, file=sys.stdout)

                # 7. Plot Results (if enabled) - requires matplotlib to be run in an interactive environment or saving plots
                # The plot generation is called here, but plt.show() will block or plots will not appear unless the environment supports it.
                # Consider saving plots to file instead if running non-interactively.
                # For now, the function is called if SHOW_PLOTS is True.
                # plot_analysis_results(full_freq_omission_data, full_pattern_analysis_data)


        else:
            # Errors during data loading or cleaning/engineering already logged and printed to file.
            pass # Analysis skipped due to data issues.

    except Exception as e:
        # Catch any unexpected errors in the main try block
        logger.error(f"An unexpected error occurred during execution: {e}", exc_info=True) # Log to console with traceback
        # Print error to file if file is open (sys.stdout is redirected)
        print(f"\nAn unexpected error occurred during execution: {e}", file=sys.stdout)
        # Print traceback to file as well
        import traceback
        print("\n--- Traceback ---", file=sys.stdout)
        traceback.print_exc(file=sys.stdout)
        print("--- End Traceback ---", file=sys.stdout)


    finally:
        # --- Close File and Restore stdout ---
        if sys.stdout is not None and sys.stdout != sys.__stdout__:
             sys.stdout.close()
             sys.stdout = sys.__stdout__ # Restore original stdout

        # Final message to console
        logger.info(f"\nAnalysis complete. Full report saved to {output_filename}")
# -*- coding: utf-8 -*-
"""
双色球数据处理器
================

本脚本负责从网络上获取双色球的历史开奖数据，并将其与本地 CSV 文件合并，
最终生成一个全面、更新的数据文件。

主要功能:
1. 从文本文件获取包含开奖日期的权威历史数据。
2. 从 HTML 网页抓取号码，与权威源的重叠期交叉核验。
3. 校验、去重并原子更新主 CSV 文件。
"""

import pandas as pd
import sys
import os
import requests
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup
import logging
import csv
import tempfile
from datetime import date, datetime
from urllib3.util.retry import Retry
from ssq_core import parse_blue_ball, parse_issue, parse_red_balls

# ==============================================================================
# --- 配置区 ---
# ==============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
CSV_FILE_PATH = os.path.join(root_dir, 'shuangseqiu.csv')

# 网络数据源URL
# TXT源：提供包括日期在内的全量历史数据
TXT_DATA_URL = 'https://data.17500.cn/ssq_asc.txt'
# HTML源：提供最新的开奖数据，通常用于快速更新（但不含日期）
HTML_DATA_URL = "https://www.17500.cn/chart/ssq-tjb.html"
MIN_FULL_SNAPSHOT_RECORDS = 100

# 配置日志系统，用于跟踪脚本运行状态和错误信息
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # 将日志输出到控制台
    ]
)
logger = logging.getLogger('ssq_data_processor')


def create_http_session():
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=('GET',),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


# ==============================================================================
# --- 数据获取模块 ---
# ==============================================================================

def fetch_latest_data_from_html(url: str = HTML_DATA_URL) -> list:
    """
    从指定的HTML网页抓取最新的双色球数据。
    注意：此数据源通常不包含开奖日期。

    Args:
        url (str): 目标网页的URL。

    Returns:
        list: 一个包含字典的列表，每个字典代表一期数据，格式为
              {'期号': '...', '红球': '...', '蓝球': '...'}。
              如果失败则返回空列表。
    """
    logger.info("正在从HTML网页抓取最新双色球数据...")
    data = []
    try:
        session = create_http_session()

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()  # 如果请求失败 (如 404, 500)，则抛出异常

        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        if not table:
            logger.warning("在网页中未能找到数据表格。")
            return []

        # 遍历表格的每一行
        for row in table.find_all('tr')[1:]:  # 跳过表头
            cells = row.find_all('td')
            # 数据行有效性校验
            if len(cells) < 3:
                continue

            try:
                # 提取期号
                period_text = cells[0].text.strip().replace("期", "")
                if not period_text.isdigit():
                    continue

                # 提取红球和蓝球
                red_balls_str = cells[1].text.strip().replace(" ", ",")
                blue_ball_str = cells[2].text.strip()

                # 验证号码格式
                red_numbers = parse_red_balls(red_balls_str)
                blue_number = parse_blue_ball(blue_ball_str)

                data.append({
                    '期号': period_text,
                    '红球': ",".join(f"{number:02d}" for number in red_numbers),
                    '蓝球': f"{blue_number:02d}"
                })
            except (ValueError, IndexError) as e:
                logger.warning(f"解析表格行时出错: {row.text.strip()}. 错误: {e}. 跳过此行。")
                continue

        logger.info(f"从HTML网页成功获取 {len(data)} 期数据。")
        return data

    except requests.exceptions.RequestException as e:
        logger.error(f"从HTML网页获取数据失败: {e}")
        return []


def fetch_full_data_from_txt(url: str = TXT_DATA_URL) -> list:
    """
    从指定的文本文件URL下载完整的历史数据。
    此数据源包含开奖日期，是数据的主要来源。

    Args:
        url (str): 目标 .txt 文件的URL。

    Returns:
        list: 包含文件中每一行字符串的列表。如果失败则返回空列表。
    """
    logger.info(f"正在从TXT文件源 ({url}) 下载全量数据...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = create_http_session().get(url, headers=headers, timeout=30)
        response.raise_for_status()
        response.encoding = 'utf-8'  # 显式设置编码
        data_lines = response.text.strip().split('\n')
        logger.info(f"成功下载 {len(data_lines)} 行数据。")
        return data_lines
    except requests.exceptions.RequestException as e:
        logger.error(f"从TXT文件源下载数据失败: {e}")
        return []


def parse_txt_data(data_lines: list) -> list:
    """
    解析从TXT文件获取的行数据，提取期号、日期、红球和蓝球。

    Args:
        data_lines (list): 包含原始行字符串的列表。

    Returns:
        list: 一个包含列表的列表，每个子列表代表一期格式化后的数据
              [期号, 日期, 红球, 蓝球]。
    """
    if not data_lines:
        return []
    logger.info("正在解析TXT数据...")
    parsed_data = []
    for line in data_lines:
        fields = line.strip().split()
        if len(fields) < 9:
            continue  # 忽略格式不正确的行
        try:
            # 数据格式: [期号, 日期, 红1, 红2, 红3, 红4, 红5, 红6, 蓝]
            qihao = parse_issue(fields[0])
            date = fields[1]
            red_balls = ",".join(fields[2:8])
            blue_ball = fields[8]
            red_numbers = parse_red_balls(red_balls)
            blue_number = parse_blue_ball(blue_ball)
            datetime.strptime(date, "%Y-%m-%d")
            parsed_data.append([
                str(qihao), date, ",".join(f"{number:02d}" for number in red_numbers),
                f"{blue_number:02d}"
            ])
        except (IndexError, ValueError) as exc:
            logger.warning(f"解析TXT行失败: {line}. 错误: {exc}")
            continue
    logger.info(f"从TXT数据中成功解析出 {len(parsed_data)} 条有效记录。")
    return parsed_data


# ==============================================================================
# --- 数据合并与存储模块 ---
# ==============================================================================


def normalize_lottery_frame(frame):
    """Validate and normalize a draw DataFrame before it reaches the CSV."""
    required_columns = ['期号', '日期', '红球', '蓝球']
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"缺少字段: {', '.join(missing)}")

    normalized = frame[required_columns].copy()
    normalized['期号'] = normalized['期号'].apply(parse_issue)
    if normalized['期号'].duplicated().any():
        duplicates = normalized.loc[normalized['期号'].duplicated(), '期号'].tolist()
        raise ValueError(f"存在重复期号: {duplicates}")
    parsed_dates = pd.to_datetime(
        normalized['日期'], format='%Y-%m-%d', errors='raise'
    )
    issue_years = normalized['期号'] // 1000
    if (issue_years != parsed_dates.dt.year).any():
        raise ValueError("期号年份与开奖日期不一致")
    issue_order = normalized['期号'].sort_values().index
    if not parsed_dates.loc[issue_order].is_monotonic_increasing:
        raise ValueError("期号与开奖日期顺序不一致")
    normalized['日期'] = parsed_dates.dt.strftime('%Y-%m-%d')
    normalized['红球'] = normalized['红球'].apply(
        lambda value: ','.join(f'{number:02d}' for number in parse_red_balls(value))
    )
    normalized['蓝球'] = normalized['蓝球'].apply(
        lambda value: f'{parse_blue_ball(value):02d}'
    )
    return normalized


def validate_authoritative_snapshot(new_data, existing_data, today=None):
    """Ensure the advertised full snapshot cannot truncate local history."""
    if len(new_data) < MIN_FULL_SNAPSHOT_RECORDS:
        raise ValueError(
            f"权威全量快照仅有 {len(new_data)} 条，少于最低要求 "
            f"{MIN_FULL_SNAPSHOT_RECORDS} 条"
        )
    today = today or date.today()
    latest_date = pd.to_datetime(new_data['日期'], format='%Y-%m-%d').max().date()
    if latest_date > today:
        raise ValueError(f"权威全量快照包含未来开奖日期: {latest_date}")
    if existing_data.empty:
        return
    missing_issues = sorted(set(existing_data['期号']) - set(new_data['期号']))
    if missing_issues:
        preview = ', '.join(str(issue) for issue in missing_issues[:5])
        suffix = ' ...' if len(missing_issues) > 5 else ''
        raise ValueError(
            f"权威全量快照缺少本地已有期号: {preview}{suffix}"
        )


def cross_check_sources(primary_records, secondary_records):
    """Report number disagreements for issues present in both data sources."""
    secondary_by_issue = {str(item['期号']): item for item in secondary_records}
    mismatches = []
    checked = 0
    for item in primary_records:
        other = secondary_by_issue.get(str(item['期号']))
        if other is None:
            continue
        checked += 1
        if item['红球'] != other['红球'] or item['蓝球'] != other['蓝球']:
            mismatches.append(str(item['期号']))
    if mismatches:
        logger.warning(f"数据源号码不一致，期号: {', '.join(mismatches)}；保留 TXT 权威数据。")
    elif checked:
        logger.info(f"两个数据源交叉核验通过，共 {checked} 期重叠记录。")
    else:
        logger.warning("两个数据源没有可交叉核验的重叠期号。")
    return mismatches


def read_existing_csv(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        logger.info("CSV文件不存在或为空，将创建新文件。")
        return pd.DataFrame()
    logger.info(f"正在读取现有CSV文件: {csv_path}")
    try:
        return pd.read_csv(
            csv_path, dtype={'期号': str}, encoding='utf-8'
        )
    except (UnicodeDecodeError, pd.errors.ParserError):
        logger.warning("UTF-8编码读取失败，尝试GBK编码...")
        return pd.read_csv(
            csv_path, dtype={'期号': str}, encoding='gbk'
        )
    except pd.errors.EmptyDataError:
        logger.warning("现有CSV文件为空。")
        return pd.DataFrame()


def atomic_write_csv(frame, csv_path):
    target_directory = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(target_directory, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', newline='', dir=target_directory,
            prefix='.ssq-', suffix='.tmp', delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            frame.to_csv(
                temporary_file, index=False, quoting=csv.QUOTE_MINIMAL
            )
        os.replace(temporary_path, csv_path)
        temporary_path = None
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def update_csv_file(csv_path: str, all_new_data: list, require_full_snapshot=False):
    """
    使用新获取的数据更新或创建CSV文件。

    此函数实现了智能合并逻辑：
    - 读取现有的CSV文件（如果存在）。
    - 将新数据与旧数据合并，并基于“期号”去重。
    - 对于重复的期号，优先保留新数据中的记录。
    - 最后，按期号升序排序并写回CSV文件。

    Args:
        csv_path (str): 目标CSV文件的路径。
        all_new_data (list): 包含所有新获取数据的DataFrame。
    """
    if not all_new_data:
        logger.info("没有新的数据可供更新，CSV文件保持不变。")
        return False

    try:
        new_data_df = normalize_lottery_frame(pd.DataFrame(all_new_data))
        existing_df = read_existing_csv(csv_path)
        if not existing_df.empty:
            existing_df = normalize_lottery_frame(existing_df)

        if require_full_snapshot:
            validate_authoritative_snapshot(new_data_df, existing_df)

        # 合并新旧数据
        # 使用 concat 和 drop_duplicates 来实现“保留后者（新数据）”的更新策略
        if not existing_df.empty:
            combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
        else:
            combined_df = new_data_df

        # 按'期号'去重，并保留最后出现的记录（即新数据）
        final_df = normalize_lottery_frame(
            combined_df.drop_duplicates(subset=['期号'], keep='last')
        ).sort_values(by='期号', ascending=True).reset_index(drop=True)

        atomic_write_csv(final_df, csv_path)
        logger.info(f"CSV文件已成功更新并保存至: {csv_path}。总计 {len(final_df)} 条记录。")
        return True

    except Exception as e:
        logger.error(f"更新CSV文件时发生严重错误: {e}")
        return False


# ==============================================================================
# --- 主执行逻辑 ---
# ==============================================================================

if __name__ == "__main__":
    logger.info("--- 开始执行双色球数据处理任务 ---")

    # 步骤 1: 从TXT文件获取全量数据（包含日期）
    txt_data_lines = fetch_full_data_from_txt()
    txt_parsed_list = parse_txt_data(txt_data_lines)
    # 将解析后的列表转换为更易于处理的字典列表
    txt_data_dicts = [{'期号': r[0], '日期': r[1], '红球': r[2], '蓝球': r[3]} for r in txt_parsed_list]
    if not txt_data_dicts:
        logger.error("TXT 权威数据源未返回有效记录，拒绝使用无日期的 HTML 数据更新 CSV。")
        raise SystemExit(1)

    # 步骤 2: 从 HTML 网页获取号码，对 TXT 中的重叠期做交叉核验。
    html_data_dicts = fetch_latest_data_from_html()
    if html_data_dicts:
        cross_check_sources(txt_data_dicts, html_data_dicts)
    
    # 步骤 3: 仅使用含完整日期的 TXT 权威数据更新主 CSV。
    if not update_csv_file(
        CSV_FILE_PATH, txt_data_dicts, require_full_snapshot=True
    ):
        raise SystemExit(1)

    logger.info("--- 双色球数据处理任务完成 ---")

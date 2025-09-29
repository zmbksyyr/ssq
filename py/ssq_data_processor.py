# -*- coding: utf-8 -*-
"""
双色球数据处理器
================

本脚本负责从网络上获取双色球的历史开奖数据，并将其与本地的CSV文件合并，
最终生成一个全面、更新的数据文件。

主要功能:
1.  从文本文件 (ssq_asc.txt) 获取包含开奖日期的完整历史数据。
2.  从HTML网页抓取最新的开奖数据（可能不含日期），作为补充。
3.  将两种来源的数据智能合并到主CSV文件 ('shuangseqiu.csv') 中。
    - 优先使用文本文件中的数据（尤其是日期）。
    - 能够处理新旧数据，自动去重和更新。
4.  具备良好的错误处理和日志记录能力，能应对网络波动和数据格式问题。
"""

import pandas as pd
import sys
import os
import requests
from bs4 import BeautifulSoup
import io
import logging
from contextlib import redirect_stdout, redirect_stderr
import csv

# ==============================================================================
# --- 配置区 ---
# ==============================================================================

# 获取当前脚本所在的目录 (e.g., /path/to/your_project/py)
script_dir = os.path.dirname(os.path.abspath(__file__))

# <--- MODIFIED: 关键修改 ---
# 获取项目的根目录，即 py/ 文件夹的上一级目录
root_dir = os.path.dirname(script_dir)

# [核心] 目标CSV文件的完整路径。现在它指向项目的根目录。
# 您可以根据需要修改文件名，例如改为 'ssq_results.csv'。
CSV_FILE_PATH = os.path.join(root_dir, 'shuangseqiu.csv')
# <--- END OF MODIFICATION ---

# 网络数据源URL
# TXT源：提供包括日期在内的全量历史数据
TXT_DATA_URL = 'http://data.17500.cn/ssq_asc.txt'
# HTML源：提供最新的开奖数据，通常用于快速更新（但不含日期）
HTML_DATA_URL = "https://www.17500.cn/chart/ssq-tjb.html"

# 配置日志系统，用于跟踪脚本运行状态和错误信息
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # 将日志输出到控制台
    ]
)
logger = logging.getLogger('ssq_data_processor')


# ==============================================================================
# --- 工具函数 ---
# ==============================================================================

class SuppressOutput:
    """
    一个上下文管理器，用于临时抑制标准输出和/或捕获标准错误。
    这在调用会产生大量无关输出的库函数时非常有用。
    捕获的错误信息会通过日志系统记录下来，避免信息丢失。
    """
    def __init__(self, suppress_stdout: bool = True, capture_stderr: bool = True):
        self.suppress_stdout = suppress_stdout
        self.capture_stderr = capture_stderr
        self.old_stdout = None
        self.old_stderr = None
        self.stderr_io = io.StringIO()

    def __enter__(self):
        if self.suppress_stdout:
            self.old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w', encoding='utf-8')

        if self.capture_stderr:
            self.old_stderr = sys.stderr
            sys.stderr = self.stderr_io
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复标准错误
        if self.capture_stderr and self.old_stderr:
            sys.stderr = self.old_stderr
            captured_stderr = self.stderr_io.getvalue()
            if captured_stderr.strip():
                logger.warning(f"在一个被抑制的输出块中捕获到标准错误:\n{captured_stderr.strip()}")
            self.stderr_io.close()

        # 恢复标准输出
        if self.suppress_stdout and self.old_stdout:
            if sys.stdout and not sys.stdout.closed:
                sys.stdout.close()
            sys.stdout = self.old_stdout

        return False  # 不抑制任何发生的异常


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
        session = requests.Session()
        session.trust_env = False  # 禁用环境变量中的代理，提高连接成功率

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
                red_numbers = [int(x) for x in red_balls_str.split(',')]
                blue_number = int(blue_ball_str)
                if len(red_numbers) != 6 or not (1 <= blue_number <= 16):
                    continue

                data.append({
                    '期号': period_text,
                    '红球': red_balls_str,
                    '蓝球': blue_ball_str
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
        response = requests.get(url, headers=headers, timeout=30)
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
            qihao = fields[0]
            date = fields[1]
            red_balls = ",".join(fields[2:8])
            blue_ball = fields[8]
            # 基础验证
            if qihao.isdigit() and len(date.split('-')) == 3:
                parsed_data.append([qihao, date, red_balls, blue_ball])
        except IndexError:
            logger.warning(f"解析TXT行时索引错误: {line}")
            continue
    logger.info(f"从TXT数据中成功解析出 {len(parsed_data)} 条有效记录。")
    return parsed_data


# ==============================================================================
# --- 数据合并与存储模块 ---
# ==============================================================================

def update_csv_file(csv_path: str, all_new_data: list):
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
        return

    try:
        # 将新数据列表转换为DataFrame
        new_data_df = pd.DataFrame(all_new_data)
        new_data_df['期号'] = new_data_df['期号'].astype(str)

        # 读取现有CSV文件
        existing_df = pd.DataFrame()
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            logger.info(f"正在读取现有CSV文件: {csv_path}")
            try:
                # 尝试用多种编码读取，增加兼容性
                existing_df = pd.read_csv(csv_path, dtype={'期号': str}, encoding='utf-8')
            except (UnicodeDecodeError, pd.errors.ParserError):
                try:
                    logger.warning("UTF-8编码读取失败，尝试GBK编码...")
                    existing_df = pd.read_csv(csv_path, dtype={'期号': str}, encoding='gbk')
                except Exception as e:
                    logger.error(f"使用多种编码读取CSV文件均失败: {e}")
                    existing_df = pd.DataFrame() # 创建空DataFrame以继续
            except pd.errors.EmptyDataError:
                logger.warning("现有CSV文件为空。")
        else:
            logger.info("CSV文件不存在或为空，将创建新文件。")

        # 合并新旧数据
        # 使用 concat 和 drop_duplicates 来实现“保留后者（新数据）”的更新策略
        if not existing_df.empty:
            combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
        else:
            combined_df = new_data_df

        # 按'期号'去重，并保留最后出现的记录（即新数据）
        final_df = combined_df.drop_duplicates(subset=['期号'], keep='last')

        # 确保列的顺序正确，并按期号排序
        final_columns = ['期号', '日期', '红球', '蓝球']
        # 补全可能缺失的列
        for col in final_columns:
            if col not in final_df.columns:
                final_df[col] = None
        
        final_df = final_df[final_columns].sort_values(by='期号', ascending=True).reset_index(drop=True)

        # 保存到CSV
        final_df.to_csv(csv_path, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
        logger.info(f"CSV文件已成功更新并保存至: {csv_path}。总计 {len(final_df)} 条记录。")

    except Exception as e:
        logger.error(f"更新CSV文件时发生严重错误: {e}")


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

    # 步骤 2: 从HTML网页获取最新数据（不含日期），作为补充
    html_data_dicts = fetch_latest_data_from_html()
    
    # 步骤 3: 准备合并数据
    # 创建一个以期号为键的字典，用于智能合并
    # 优先级：TXT数据 > HTML数据，因为TXT数据包含更关键的日期信息
    merged_data_dict = {}

    # 首先添加HTML数据
    for item in html_data_dicts:
        merged_data_dict[item['期号']] = item
    
    # 然后用TXT数据覆盖或添加，TXT数据有更高优先级
    for item in txt_data_dicts:
        merged_data_dict[item['期号']] = item

    # 将合并后的字典转换回列表
    final_new_data = list(merged_data_dict.values())
    
    # 步骤 4: 更新主CSV文件
    update_csv_file(CSV_FILE_PATH, final_new_data)

    logger.info("--- 双色球数据处理任务完成 ---")

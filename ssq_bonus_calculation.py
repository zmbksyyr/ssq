# -*- coding: utf-8 -*-
"""
双色球推荐结果验证与奖金计算器
=================================

本脚本旨在自动评估 `ssq_analyzer.py` 生成的推荐号码的实际表现。

工作流程:
1.  读取 `shuangseqiu.csv` 文件，获取所有历史开奖数据。
2.  确定最新的一期为“评估期”，倒数第二期为“报告数据截止期”。
3.  根据“报告数据截止期”，在当前目录下查找对应的分析报告文件
    (ssq_analysis_output_*.txt)。
4.  从找到的报告中解析出“单式推荐”和“复式参考”的号码。
5.  将复式参考号码展开为所有可能的单式投注。
6.  使用“评估期”的实际开奖号码，核对所有推荐投注的中奖情况。
7.  计算总奖金，并将详细的中奖结果（包括中奖号码、奖级、金额）
    追加记录到主报告文件 `latest_ssq_calculation.txt` 中。
8.  主报告文件会自动管理记录数量，只保留最新的N条评估记录和错误日志。
"""

import os
import re
import glob
import csv
from itertools import combinations
from datetime import datetime
import traceback
from typing import Optional, Tuple, List, Dict # <--- [FIX] 导入兼容的类型提示

# ==============================================================================
# --- 配置区 ---
# ==============================================================================

# 脚本需要查找的分析报告文件名的模式
REPORT_PATTERN = "ssq_analysis_output_*.txt"
# 开奖数据源CSV文件
CSV_FILE = "shuangseqiu.csv"
# 最终生成的主评估报告文件名
MAIN_REPORT_FILE = "latest_ssq_calculation.txt"

# 主报告文件中保留的最大记录数
MAX_NORMAL_RECORDS = 10  # 保留最近10次评估
MAX_ERROR_LOGS = 20      # 保留最近20条错误日志

# 奖金对照表 (元)
PRIZE_TABLE = {
    1: 5_000_000,  # 一等奖 (浮动，此处为估算)
    2: 150_000,    # 二等奖 (浮动，此处为估算)
    3: 3_000,      # 三等奖
    4: 200,        # 四等奖
    5: 10,         # 五等奖
    6: 5,          # 六等奖
}

# ==============================================================================
# --- 工具函数 ---
# ==============================================================================

def log_message(message: str, level: str = "INFO"):
    """一个简单的日志打印函数，用于在控制台显示脚本执行状态。"""
    print(f"[{level}] {datetime.now().strftime('%H:%M:%S')} - {message}")

def robust_file_read(file_path: str) -> Optional[str]: # <--- [FIX] 使用 Optional[str] 替代 str | None
    """
    一个健壮的文件读取函数，能自动尝试多种编码格式。

    Args:
        file_path (str): 待读取文件的路径。

    Returns:
        Optional[str]: 文件内容字符串，如果失败则返回 None。
    """
    if not os.path.exists(file_path):
        log_message(f"文件未找到: {file_path}", "ERROR")
        return None
    encodings = ['utf-8', 'gbk', 'latin-1']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, IOError):
            continue
    log_message(f"无法使用任何支持的编码打开文件: {file_path}", "ERROR")
    return None

# ==============================================================================
# --- 数据解析与查找模块 ---
# ==============================================================================

def get_period_data_from_csv(csv_content: str) -> Tuple[Optional[Dict], Optional[List]]: # <--- [FIX] 使用 Tuple 和 Optional 替代
    """
    从CSV文件内容中解析出所有期号的开奖数据。

    Args:
        csv_content (str): 从CSV文件读取的字符串内容。

    Returns:
        Tuple[Optional[Dict], Optional[List]]:
            - 一个以期号为键，开奖数据为值的字典。
            - 一个按升序排序的期号列表。
            如果解析失败则返回 (None, None)。
    """
    if not csv_content:
        log_message("输入的CSV内容为空。", "WARNING")
        return None, None
    period_map, periods_list = {}, []
    try:
        reader = csv.reader(csv_content.splitlines())
        next(reader)  # 跳过表头
        for i, row in enumerate(reader):
            if len(row) >= 4 and re.match(r'^\d{4,7}$', row[0]):
                try:
                    period, date, red_str, blue_str = row[0], row[1], row[2], row[3]
                    red_balls = sorted(map(int, red_str.split(',')))
                    blue_ball = int(blue_str)
                    if len(red_balls) != 6 or not all(1 <= r <= 33 for r in red_balls) or not (1 <= blue_ball <= 16):
                        continue
                    period_map[period] = {'date': date, 'red': red_balls, 'blue': blue_ball}
                    periods_list.append(period)
                except (ValueError, IndexError):
                    log_message(f"CSV文件第 {i+2} 行数据格式无效，已跳过: {row}", "WARNING")
    except Exception as e:
        log_message(f"解析CSV数据时发生严重错误: {e}", "ERROR")
        return None, None
    
    if not period_map:
        log_message("未能从CSV中解析到任何有效的开奖数据。", "WARNING")
        return None, None
        
    return period_map, sorted(periods_list, key=int)

def find_matching_report(target_period: str) -> Optional[str]: # <--- [FIX] 使用 Optional[str] 替代 str | None
    """
    在当前目录查找其数据截止期与 `target_period` 匹配的最新分析报告。

    Args:
        target_period (str): 目标报告的数据截止期号。

    Returns:
        Optional[str]: 找到的报告文件的路径，如果未找到则返回 None。
    """
    log_message(f"正在查找数据截止期为 {target_period} 的分析报告...")
    candidates = []
    # 使用 SCRIPT_DIR 确保在任何工作目录下都能找到与脚本同级的报告文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for file_path in glob.glob(os.path.join(script_dir, REPORT_PATTERN)):
        content = robust_file_read(file_path)
        if not content: continue
        
        match = re.search(r'分析基于数据:\s*截至\s*(\d+)\s*期', content)
        if match and match.group(1) == target_period:
            try:
                # 从文件名中提取时间戳以确定最新报告
                timestamp_str_match = re.search(r'_(\d{8}_\d{6})\.txt$', file_path)
                if timestamp_str_match:
                    timestamp_str = timestamp_str_match.group(1)
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    candidates.append((timestamp, file_path))
            except (AttributeError, ValueError):
                continue
    
    if not candidates:
        log_message(f"未找到数据截止期为 {target_period} 的分析报告。", "WARNING")
        return None
        
    candidates.sort(reverse=True)
    latest_report = candidates[0][1]
    log_message(f"找到匹配的最新报告: {os.path.basename(latest_report)}", "INFO")
    return latest_report

def parse_recommendations_from_report(content: str) -> Tuple[List, List, List]: # <--- [FIX] 使用 Tuple 和 List
    """
    从分析报告内容中解析出单式和复式推荐号码。

    Args:
        content (str): 分析报告的文本内容。

    Returns:
        Tuple[List, List, List]:
            - 单式推荐列表, e.g., [([1,2,3,4,5,6], 7), ...]
            - 复式红球列表, e.g., [1,2,3,4,5,6,7]
            - 复式蓝球列表, e.g., [8,9,10]
    """
    # 解析单式推荐
    rec_pattern = re.compile(r'注\s*\d+:\s*红球\s*\[([\d\s]+)\]\s*蓝球\s*\[(\d+)\]')
    rec_tickets = []
    for match in rec_pattern.finditer(content):
        try:
            reds = sorted(map(int, match.group(1).split()))
            blue = int(match.group(2))
            if len(reds) == 6: rec_tickets.append((reds, blue))
        except ValueError: continue
    
    # 解析复式推荐
    complex_reds, complex_blues = [], []
    red_match = re.search(r'红球\s*\(Top\s*\d+\):\s*([\d\s]+)', content)
    if red_match:
        try: complex_reds = sorted(map(int, red_match.group(1).split()))
        except ValueError: pass
        
    blue_match = re.search(r'蓝球\s*\(Top\s*\d+\):\s*([\d\s]+)', content)
    if blue_match:
        try: complex_blues = sorted(map(int, blue_match.group(1).split()))
        except ValueError: pass
        
    log_message(f"解析到 {len(rec_tickets)} 注单式推荐, 复式红球 {len(complex_reds)} 个, 蓝球 {len(complex_blues)} 个。")
    return rec_tickets, complex_reds, complex_blues

# ==============================================================================
# --- 奖金计算与报告生成模块 ---
# ==============================================================================

def generate_complex_tickets(reds: List, blues: List) -> List: # <--- [FIX] 使用 List
    """从复式号码中生成所有可能的单式投注组合。"""
    if len(reds) < 6 or not blues: return []
    max_tickets_limit = 20000  # 防止组合爆炸
    
    try:
        # 使用数学公式预先计算组合数，避免生成超大列表
        from math import comb
        num_combs = comb(len(reds), 6) * len(blues)

        if num_combs > max_tickets_limit:
            log_message(f"复式号码将生成 {num_combs:,} 注，超过 {max_tickets_limit:,} 的限制，已跳过。", "WARNING")
            return []
            
        tickets = [(sorted(list(r_combo)), b) for r_combo in combinations(reds, 6) for b in blues]
        log_message(f"从复式号码中成功生成 {len(tickets):,} 注投注。")
        return tickets
    except ImportError: # Fallback for Python < 3.8
        log_message("math.comb 不可用，正在使用自定义函数计算组合数。您的 Python 版本可能较低。", "WARNING")
        def combinations_count(n, k):
            if k < 0 or k > n: return 0
            if k == 0 or k == n: return 1
            if k > n // 2: k = n - k
            res = 1
            for i in range(k): res = res * (n - i) // (i + 1)
            return res
        num_combs = combinations_count(len(reds), 6) * len(blues)
        if num_combs > max_tickets_limit:
            log_message(f"复式号码将生成 {num_combs:,} 注，超过 {max_tickets_limit:,} 的限制，已跳过。", "WARNING")
            return []
        tickets = [(sorted(list(r_combo)), b) for r_combo in combinations(reds, 6) for b in blues]
        log_message(f"从复式号码中成功生成 {len(tickets):,} 注投注。")
        return tickets
    except Exception as e:
        log_message(f"生成复式投注时出错: {e}", "ERROR")
        return []

def calculate_prize(tickets: List, prize_red: List, prize_blue: int) -> Tuple[int, Dict, List]: # <--- [FIX] 使用 Tuple, Dict, List
    """
    计算给定投注列表的总奖金、奖级分布和中奖详情。

    Args:
        tickets (List): 投注列表, 格式为 [([r1..r6], b), ...]。
        prize_red (List): 中奖红球列表。
        prize_blue (int): 中奖蓝球。

    Returns:
        Tuple[int, Dict, List]: 总奖金, 奖级分布字典, 中奖号码详情列表。
    """
    prize_red_set = set(prize_red)
    breakdown = {level: 0 for level in PRIZE_TABLE}
    total_prize = 0
    winning_tickets_details = []

    for red, blue in tickets:
        red_hits = len(set(red) & prize_red_set)
        blue_hit = blue == prize_blue
        
        level = None
        if blue_hit:
            if red_hits == 6: level = 1
            elif red_hits == 5: level = 3
            elif red_hits == 4: level = 4
            elif red_hits == 3: level = 5
            elif red_hits <= 2: level = 6
        else:
            if red_hits == 6: level = 2
            elif red_hits == 5: level = 4
            elif red_hits == 4: level = 5

        if level and level in PRIZE_TABLE:
            prize_amount = PRIZE_TABLE[level]
            total_prize += prize_amount
            breakdown[level] += 1
            winning_tickets_details.append({'red': red, 'blue': blue, 'level': level})
            
    return total_prize, breakdown, winning_tickets_details

def format_winning_tickets_for_report(winning_list: List[Dict], prize_red: List, prize_blue: int) -> List[str]: # <--- [FIX]
    """格式化中奖号码，高亮命中的数字，用于报告输出。"""
    formatted_lines = []
    prize_red_set = set(prize_red)
    for ticket in winning_list:
        red, blue, level = ticket['red'], ticket['blue'], ticket['level']
        red_str = ' '.join(f"**{r:02d}**" if r in prize_red_set else f"{r:02d}" for r in red)
        blue_str = f"**{blue:02d}**" if blue == prize_blue else f"{blue:02d}"
        formatted_lines.append(f"  - 红球 [{red_str}] 蓝球 [{blue_str}]  -> {level}等奖")
    return formatted_lines

def manage_report(new_entry: Optional[Dict] = None, new_error: Optional[str] = None): # <--- [FIX]
    """
    维护主评估报告文件，自动追加新记录并清理旧记录。

    Args:
        new_entry (Optional[Dict]): 新的评估结果字典。
        new_error (Optional[str]): 新的错误日志字符串。
    """
    normal_marker, error_marker = "==== 评估记录 ====", "==== 错误日志 ===="
    content_str = robust_file_read(MAIN_REPORT_FILE) or ""
    
    # 分割文件内容为记录块和错误日志
    parts = content_str.split(error_marker)
    normal_part = parts[0]
    error_part = parts[1] if len(parts) > 1 else ""
    
    # 解析现有记录
    normal_entries = [entry.strip() for entry in normal_part.split('='*20) if entry.strip() and normal_marker not in entry]
    error_entries = [err.strip() for err in error_part.splitlines() if err.strip()]

    # 添加新记录
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if new_entry:
        entry_lines = [
            f"评估时间: {timestamp}",
            f"评估期号 (实际开奖): {new_entry['eval_period']}",
            f"分析报告数据截止期: {new_entry['report_cutoff_period']}",
            f"开奖号码: 红球 {new_entry['prize_red']} 蓝球 {new_entry['prize_blue']}",
            f"总奖金: {new_entry['total_prize']:,} 元",
            "", "--- 单式推荐详情 ---"
        ]
        rec_prize, rec_bd, rec_winners = new_entry['rec_prize'], new_entry['rec_breakdown'], new_entry['rec_winners']
        if rec_prize > 0:
            entry_lines.append(f"奖金: {rec_prize:,}元 | 明细: " + ", ".join(f"{k}等奖x{v}" for k,v in rec_bd.items() if v>0))
            entry_lines.extend(format_winning_tickets_for_report(rec_winners, new_entry['prize_red'], new_entry['prize_blue']))
        else: entry_lines.append("未中奖")
        
        entry_lines.extend(["", "--- 复式推荐详情 ---"])
        com_prize, com_bd, com_winners = new_entry['com_prize'], new_entry['com_breakdown'], new_entry['com_winners']
        if com_prize > 0:
            entry_lines.append(f"奖金: {com_prize:,}元 | 明细: " + ", ".join(f"{k}等奖x{v}" for k,v in com_bd.items() if v>0))
            entry_lines.extend(format_winning_tickets_for_report(com_winners, new_entry['prize_red'], new_entry['prize_blue']))
        else: entry_lines.append("未中奖或未生成投注")
        
        normal_entries.insert(0, "\n".join(entry_lines))

    if new_error:
        error_entries.insert(0, f"[{timestamp}] {new_error}")

    # 清理旧记录
    final_normal_entries = normal_entries[:MAX_NORMAL_RECORDS]
    final_error_entries = error_entries[:MAX_ERROR_LOGS]

    # 写回文件
    try:
        with open(MAIN_REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write(f"{normal_marker}\n")
            if final_normal_entries:
                f.write(("\n" + "="*20 + "\n").join(final_normal_entries))
            f.write(f"\n\n{error_marker}\n")
            if final_error_entries:
                f.write("\n".join(final_error_entries))
        log_message(f"主报告已更新: {MAIN_REPORT_FILE}", "INFO")
    except IOError as e:
        log_message(f"写入主报告文件失败: {e}", "ERROR")

# ==============================================================================
# --- 主流程 ---
# ==============================================================================

def main_process():
    """主处理流程，串联所有功能。"""
    log_message("====== 主流程启动 ======", "INFO")
    
    csv_content = robust_file_read(CSV_FILE)
    if not csv_content:
        manage_report(new_error=f"无法读取或未找到CSV数据文件: {CSV_FILE}")
        return

    period_map, sorted_periods = get_period_data_from_csv(csv_content)
    if not period_map or not sorted_periods or len(sorted_periods) < 2:
        manage_report(new_error="CSV数据不足两期或解析失败，无法进行评估。")
        return

    eval_period = sorted_periods[-1]
    report_cutoff_period = sorted_periods[-2]
    log_message(f"评估期号: {eval_period}, 报告数据截止期: {report_cutoff_period}", "INFO")

    report_path = find_matching_report(report_cutoff_period)
    if not report_path:
        manage_report(new_error=f"未找到数据截止期为 {report_cutoff_period} 的分析报告。")
        return

    report_content = robust_file_read(report_path)
    if not report_content:
        manage_report(new_error=f"无法读取分析报告文件: {report_path}")
        return

    rec_tickets, complex_reds, complex_blues = parse_recommendations_from_report(report_content)
    complex_tickets = generate_complex_tickets(complex_reds, complex_blues)
    
    if not rec_tickets and not complex_tickets:
         manage_report(new_error=f"未能从报告 {os.path.basename(report_path)} 中解析出任何有效投注。")
    
    prize_data = period_map[eval_period]
    prize_red, prize_blue = prize_data['red'], prize_data['blue']
    log_message(f"获取到期号 {eval_period} 的开奖数据: 红{prize_red} 蓝{prize_blue}", "INFO")
    
    rec_prize, rec_bd, rec_winners = calculate_prize(rec_tickets, prize_red, prize_blue)
    com_prize, com_bd, com_winners = calculate_prize(complex_tickets, prize_red, prize_blue)
    
    report_entry = {
        'eval_period': eval_period, 'report_cutoff_period': report_cutoff_period,
        'prize_red': prize_red, 'prize_blue': prize_blue,
        'total_prize': rec_prize + com_prize,
        'rec_prize': rec_prize, 'rec_breakdown': rec_bd, 'rec_winners': rec_winners,
        'com_prize': com_prize, 'com_breakdown': com_bd, 'com_winners': com_winners,
    }
    manage_report(new_entry=report_entry)
    
    log_message(f"处理完成！总计奖金: {report_entry['total_prize']:,}元。", "INFO")
    log_message("====== 主流程结束 ======", "INFO")

if __name__ == "__main__":
    try:
        main_process()
    except Exception as e:
        tb_str = traceback.format_exc()
        error_message = f"主流程发生未捕获的严重异常: {type(e).__name__} - {e}\n{tb_str}"
        log_message(error_message, "CRITICAL")
        try:
            manage_report(new_error=error_message)
        except Exception as report_e:
            log_message(f"在记录严重错误时再次发生错误: {report_e}", "CRITICAL")

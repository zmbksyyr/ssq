import os
import re
import glob
import csv
from itertools import combinations
from datetime import datetime

# 配置参数
REPORT_PATTERN = "ssq_analysis_output_*.txt"
CSV_FILE = "shuangseqiu.csv"
MAIN_REPORT_FILE = "ssq_master_report.txt"
MAX_NORMAL_RECORDS = 10
MAX_ERROR_LOGS = 20

# 奖金对照表
PRIZE_TABLE = {
    1: 5_000_000,
    2: 500_000,
    3: 3_000,
    4: 200,
    5: 10,
    6: 5
}

# 移除了 PeriodIntegrityError 异常，因为它不再需要

class PrizeDataNotFound(Exception):
    """开奖数据缺失异常"""
    def __init__(self, target_period):
        super().__init__(f"找不到期号 {target_period} 的开奖数据")
        self.target_period = target_period

def debug_log(message, level=1):
    """分级调试日志"""
    prefixes = {
        1: "[INFO]",
        2: "[WARNING]",
        3: "[ERROR]"
    }
    color_codes = {
        1: "\033[94m",   # 蓝色
        2: "\033[93m",   # 黄色
        3: "\033[91m"    # 红色
    }
    reset_code = "\033[0m"
    prefix = prefixes.get(level, "[DEBUG]")
    color = color_codes.get(level, "\033[90m")
    print(f"{color}{prefix} {datetime.now().strftime('%H:%M:%S')} {message}{reset_code}")

def robust_file_read(file_path):
    """带编码回退的文件读取"""
    encodings = ['utf-8', 'gbk', 'gb2312']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            debug_log(f"成功使用 {encoding} 编码读取文件: {file_path}")
            return content
        except UnicodeDecodeError:
            continue
        except Exception as e:
            debug_log(f"读取文件 {file_path} 异常：{str(e)}", 3)
            return None
    debug_log(f"无法使用所有尝试的编码读取文件: {file_path}", 3)
    return None

def get_period_data(csv_content):
    """获取CSV期号数据"""
    period_map = {}
    periods_list = [] # Keep track of periods in order
    try:
        reader = csv.reader(csv_content.splitlines())
        for row in reader:
            # Skip header or invalid rows
            if len(row) >= 4 and re.match(r'\d{7}', row[0]):
                period = row[0]
                period_map[period] = {
                    'date': row[1],
                    'red': sorted(list(map(int, row[2].split(',')))),
                    'blue': int(row[3])
                }
                periods_list.append(period) # Add to list in read order
        # Ensure list is sorted numerically just in case CSV isn't perfectly ordered
        sorted_periods = sorted(periods_list, key=int)
        return period_map, sorted_periods
    except Exception as e:
        debug_log(f"CSV数据解析失败: {str(e)}", 3)
        return None, None

# 移除了 build_period_chain 函数

def find_matching_report(target_period):
    """查找匹配指定期号的分析报告"""
    debug_log(f"开始查找处理期号 {target_period} 的分析报告...")
    candidates = []

    for file in glob.glob(REPORT_PATTERN):
        content = robust_file_read(file)
        if not content:
            continue

        # 修正：确保匹配期号时使用正确的变量
        match = re.search(r'数据期数范围:.*?第\s*(\d+)\s*期\s*至\s*第\s*(\d+)\s*期', content)
        # 检查报告的结束期号是否与 target_period 匹配
        if match and match.group(2) == target_period:
            time_match = re.search(r'_(\d{8}_\d{6})\.', file)
            if time_match:
                timestamp = datetime.strptime(time_match.group(1), "%Y%m%d_%H%M%S")
                candidates.append((timestamp, file))

    if not candidates:
        debug_log(f"未找到处理期号 {target_period} 的分析报告", 3)
        return None

    # 选择时间最新的报告
    candidates.sort(reverse=True)
    selected = candidates[0][1]
    debug_log(f"找到 {len(candidates)} 个匹配报告，选择最新: {selected}")
    return selected

def parse_recommendations(content):
    """解析推荐组合"""
    debug_log("解析推荐组合...")
    # 使用非贪婪匹配量词 *? 和合适的边界来更精确地匹配红球和蓝球
    pattern = re.compile(
        r'组合\s*\d+\s*:\s*红球\s*$$([\d\s,]+?)$$\s*蓝球\s*(\d+)', # 使用 +? 进行非贪婪匹配
        re.DOTALL
    )
    parsed_rec = []
    for red, blue in pattern.findall(content):
        # 移除红球字符串中的空格和可能的多余逗号
        red_str = red.replace(' ', '').strip(',')
        # 确保解析出的红球数量是6个
        if red_str and len(red_str.split(',')) == 6:
             try:
                parsed_rec.append((sorted(map(int, red_str.split(','))), int(blue)))
             except ValueError:
                 debug_log(f"解析推荐组合时发现无效数字: 红球 '{red_str}', 蓝球 '{blue}'", 2)
                 continue # Skip this combination if conversion fails

    # 只取前5个有效的推荐组合，如果少于5个则取全部
    return parsed_rec[:5]


def parse_complex(content):
    """解析复式组合"""
    debug_log("解析复式组合...")
    # 使用非贪婪匹配和更精确的红蓝球匹配
    match = re.search(
        r'7\+7复式选号.*?红球:\s*$$([\d\s,]+?)$$.*?蓝球:\s*$$([\d\s,]+?)$$', # 使用 +?
        content, re.DOTALL
    )
    if not match:
        debug_log("未找到 7+7 复式选号模式", 1)
        return [], []

    try:
        # 移除红球蓝球字符串中的空格和可能的多余逗号
        complex_red_str = match.group(1).replace(' ', '').strip(',')
        complex_blue_str = match.group(2).replace(' ', '').strip(',')

        complex_red = sorted(map(int, complex_red_str.split(','))) if complex_red_str else []
        complex_blue = sorted(map(int, complex_blue_str.split(','))) if complex_blue_str else []

        # 简单验证一下数量，虽然生成投注时还会再检查
        if len(complex_red) < 6 or len(complex_blue) < 1:
             debug_log(f"解析到不完整的复式号码: 红球 {complex_red}, 蓝球 {complex_blue}", 2)
             return [], []

        return complex_red, complex_blue
    except ValueError:
         debug_log(f"解析复式组合时发现无效数字: 红球 '{match.group(1)}', 蓝球 '{match.group(2)}'", 2)
         return [], []
    except Exception as e:
        debug_log(f"解析复式组合时发生未知错误: {str(e)}", 3)
        return [], []


def generate_complex_tickets(reds, blues):
    """生成复式投注"""
    if len(reds) < 6 or not blues:
        debug_log(f"复式号码不足，无法生成投注: 红球 {len(reds)}个, 蓝球 {len(blues)}个", 1)
        return []

    tickets = []
    # 限制生成的组合数量，避免过多
    max_combinations = 1000 # 例如，限制在1000注以内
    if len(reds) > 11 or len(blues) > 10: # 粗略估计超过一定范围可能组合过多 (C(12,6)*1=924, C(11,6)*10=4620)
         debug_log(f"复式号码数量过多，可能生成大量组合: 红球 {len(reds)}个, 蓝球 {len(blues)}个", 2)
         # 可以在这里添加更严格的限制或警告

    for combo in combinations(reds, 6):
        if len(tickets) >= max_combinations:
            debug_log(f"已生成 {max_combinations} 注复式投注，停止生成。", 2)
            break
        for blue in blues:
             if len(tickets) >= max_combinations:
                 break
             tickets.append((sorted(list(combo)), blue))

    debug_log(f"从复式号码中生成了 {len(tickets)} 注投注。", 1)
    return tickets


def calculate_prize(tickets, prize_red, prize_blue):
    """计算奖金"""
    prize_red_set = set(prize_red)
    breakdown = {k:0 for k in PRIZE_TABLE}

    for red, blue in tickets:
        matched_red = len(set(red) & prize_red_set)
        matched_blue = blue == prize_blue

        level = None # 初始化 level

        if matched_red == 6:
            level = 1 if matched_blue else 2 # 修复语法错误
        elif matched_red == 5:
            level = 3 if matched_blue else 4 # 修复语法错误
        elif matched_red == 4:
            level = 4 if matched_blue else 5 # 修复语法错误
        elif matched_red == 3 and matched_blue:
            level = 5
        elif matched_blue:
            level = 6

        if level is not None:
            breakdown[level] += 1

    total = sum(PRIZE_TABLE[k] * v for k, v in breakdown.items())
    debug_log(f"奖金计算明细: {breakdown}", 1)
    return total, breakdown

def manage_report(new_entry=None, new_error=None):
    """维护主报告文件"""
    normal_marker = "==== NORMAL RECORDS ===="
    error_marker = "==== ERROR LOGS ===="

    # 读取现有内容
    normal_entries = []
    error_logs = []
    if os.path.exists(MAIN_REPORT_FILE):
        try:
            with open(MAIN_REPORT_FILE, 'r', encoding='utf-8') as f:
                current_section = None
                # Read lines and remove potential BOM if present
                lines = [line.lstrip('\ufeff').strip() for line in f if line.strip()]

                for line in lines:
                    if line.startswith(normal_marker):
                        current_section = 'normal'
                        continue
                    if line.startswith(error_marker):
                        current_section = 'error'
                        continue

                    if current_section == 'normal':
                        normal_entries.append(line)
                    elif current_section == 'error':
                        error_logs.append(line)
        except Exception as e:
            # If reading the report fails, just log it and proceed as if the file was empty
            debug_log(f"读取主报告文件失败: {str(e)}", 3)
            normal_entries = []
            error_logs = []


    # 处理新增内容
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if new_entry:
        # Ensure new_entry contains expected keys to prevent errors
        period = new_entry.get('period', 'N/A')
        red = new_entry.get('red', 'N/A')
        blue = new_entry.get('blue', 'N/A')
        prize = new_entry.get('prize', 'N/A')

        entry_block = [
            f"[记录 {timestamp}]",
            f"分析期号(对应开奖期号): {period}", # Clarify which period this refers to
            f"开奖号码: 红{red} 蓝{blue}",
            f"中奖金额: {prize:,}元",
            "-"*40
        ]
        # Insert new block at the beginning
        normal_entries = entry_block + normal_entries

    if new_error:
        error_logs = [f"[错误 {timestamp}] {new_error}"] + error_logs

    # 限制记录数量
    # Since each normal entry is a block of 5 lines (including separator),
    # we keep MAX_NORMAL_RECORDS blocks.
    lines_per_normal_entry = 5 # Lines per block: [记录], 期号, 号码, 金额, ----
    normal_entries_to_keep = MAX_NORMAL_RECORDS * lines_per_normal_entry
    # Find the start of blocks to keep the last MAX_NORMAL_RECORDS blocks
    # We need to find the index of the (MAX_NORMAL_RECORDS)-th block separator from the end
    # This is a bit complex with varying line counts per entry potentially.
    # A simpler approach is to count the block start markers "[记录"
    block_start_indices = [i for i, line in enumerate(normal_entries) if line.startswith("[记录")]
    if len(block_start_indices) > MAX_NORMAL_RECORDS:
        # Keep the latest MAX_NORMAL_RECORDS blocks
        start_index_to_keep = block_start_indices[-MAX_NORMAL_RECORDS]
        normal_entries = normal_entries[start_index_to_keep:]

    # Error logs are simpler, just keep the last MAX_ERROR_LOGS lines
    error_logs = error_logs[:MAX_ERROR_LOGS]


    # 写入文件
    try:
        with open(MAIN_REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write(f"{normal_marker}\n")
            # Ensure we only write the limited number of normal entries
            for entry in normal_entries:
                 f.write(entry + "\n") # Write each line of the block

            f.write(f"\n{error_marker}\n") # Add an extra newline for separation
            # Ensure we only write the limited number of error logs
            for log in error_logs:
                 f.write(log + "\n")

    except Exception as e:
        debug_log(f"写入主报告文件失败: {str(e)}", 3)


def main_process():
    """主处理流程"""
    debug_log("====== 主流程启动 ======", 1)

    # 读取CSV数据
    csv_content = robust_file_read(CSV_FILE)
    if not csv_content:
        manage_report(new_error=f"CSV文件 '{CSV_FILE}' 读取失败")
        return

    # 构建期号数据 (获取period_map和sorted_periods)
    period_map, sorted_periods = get_period_data(csv_content)
    if period_map is None or sorted_periods is None:
        # get_period_data already logged the error
        manage_report(new_error="CSV数据解析失败") # Log generic failure for report
        return

    if len(sorted_periods) < 2:
        manage_report(new_error="CSV数据不足两期，无法确定下一期开奖结果")
        debug_log("处理停止：CSV数据不足两期。", 2)
        return

    # 获取CSV的最新期号 (用于查找对应的分析报告)
    latest_period_in_csv = sorted_periods[-1]
    debug_log(f"CSV最新期号: {latest_period_in_csv}")

    # 查找匹配该最新期号的分析报告
    analysis_file = find_matching_report(latest_period_in_csv)
    if not analysis_file:
        manage_report(new_error=f"未找到处理期号 {latest_period_in_csv} 的分析报告")
        debug_log(f"处理停止：未找到处理期号 {latest_period_in_csv} 的分析报告。", 2)
        return

    # 解析分析报告
    content = robust_file_read(analysis_file)
    if not content:
        manage_report(new_error=f"分析报告 '{analysis_file}' 读取失败")
        debug_log(f"处理停止：分析报告 '{analysis_file}' 读取失败。", 2)
        return

    rec_tickets = parse_recommendations(content)
    complex_red, complex_blue = parse_complex(content)
    complex_tickets = generate_complex_tickets(complex_red, complex_blue)
    all_tickets = rec_tickets + complex_tickets

    if not all_tickets:
        manage_report(new_error=f"未能从分析报告 '{analysis_file}' 中解析出任何投注组合")
        debug_log("处理停止：未能从分析报告中解析出任何投注组合。", 2)
        return

    # 获取下一期开奖数据
    try:
        # 在sorted_periods中找到最新期号的位置
        latest_period_index = sorted_periods.index(latest_period_in_csv)
    except ValueError:
        # This case should theoretically not happen if latest_period_in_csv comes from sorted_periods,
        # but as a safeguard.
        manage_report(new_error=f"内部错误: 未在 sorted_periods 中找到期号 {latest_period_in_csv}")
        debug_log(f"处理停止：内部错误，未在 sorted_periods 中找到期号 {latest_period_in_csv}。", 3)
        return


    # 检查是否有下一期数据
    if latest_period_index + 1 >= len(sorted_periods):
        # CSV中没有下一期数据
        # 计算下一期期号（仅用于提示）
        try:
            expected_next_period = str(int(latest_period_in_csv) + 1).zfill(7) # Assumes 7 digit periods
        except ValueError:
             expected_next_period = "下一期" # Fallback if period is not purely numeric

        manage_report(new_error=f"CSV数据不包含期号 {latest_period_in_csv} 的下一期开奖结果 (预期的下一期为 {expected_next_period})")
        debug_log(f"处理停止：CSV中没有期号 {latest_period_in_csv} 的下一期数据。", 2)
        return

    # 获取下一期期号和开奖数据
    next_period_in_csv = sorted_periods[latest_period_index + 1]
    # Double check if the next period exists in the map (redundant if from sorted_periods, but safe)
    if next_period_in_csv not in period_map:
        manage_report(new_error=f"内部错误: 期号 {next_period_in_csv} 在 sorted_periods 中，但不在 period_map 中")
        debug_log(f"处理停止：内部错误，期号 {next_period_in_csv} 在 sorted_periods 中，但不在 period_map 中。", 3)
        return

    prize_data = period_map[next_period_in_csv]
    debug_log(f"找到对应分析报告 {latest_period_in_csv} 的下一期数据: 期号 {next_period_in_csv}", 1)


    # 计算奖金
    total_prize, breakdown = calculate_prize(all_tickets, prize_data['red'], prize_data['blue'])

    # 保存结果
    manage_report(new_entry={
        'period': next_period_in_csv, # Record the actual next period
        'red': prize_data['red'],
        'blue': prize_data['blue'],
        'prize': total_prize
    })

    debug_log(f"处理完成！分析期号 {latest_period_in_csv} 的预测，中奖金额: {total_prize:,}元 (开奖期号 {next_period_in_csv})", 1)
    debug_log("====== 主流程结束 ======", 1)


if __name__ == "__main__":
    try:
        main_process()
    except Exception as e:
        manage_report(new_error=f"未处理异常: {str(e)}")
        debug_log(f"主流程异常: {str(e)}", 3)

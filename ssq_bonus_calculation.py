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
MAX_NORMAL_RECORDS = 10 # 主报告保留的普通记录数量
MAX_ERROR_LOGS = 20    # 主报告保留的错误日志数量

# 奖金对照表
PRIZE_TABLE = {
    1: 5_000_000, # 一等奖
    2: 500_000,   # 二等奖
    3: 3_000,     # 三等奖
    4: 200,       # 四等奖
    5: 10,        # 五等奖
    6: 5         # 六等奖
}

# 移除了 PeriodIntegrityError 异常

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
        except FileNotFoundError:
            debug_log(f"文件未找到: {file_path}", 3)
            return None
        except Exception as e:
            debug_log(f"读取文件 {file_path} 异常：{str(e)}", 3)
            return None
    debug_log(f"无法使用所有尝试的编码读取文件: {file_path}", 3)
    return None

def get_period_data(csv_content):
    """获取CSV期号数据并按期号排序"""
    period_map = {}
    periods_list = [] # 用于存储读取到的期号
    try:
        reader = csv.reader(csv_content.splitlines())
        # next(reader) # 如果确定有标题行，可以取消注释此行

        for i, row in enumerate(reader):
            if len(row) >= 4 and re.match(r'^\d{7}$', row[0]):
                period = row[0].strip()
                date_str = row[1].strip()
                red_str = row[2].strip()
                blue_str = row[3].strip()

                try:
                    red_balls = sorted(list(map(int, red_str.split(','))))
                    blue_ball = int(blue_str)

                    period_map[period] = {
                        'date': date_str,
                        'red': red_balls,
                        'blue': blue_ball
                    }
                    periods_list.append(period)

                except ValueError:
                    debug_log(f"CSV数据格式错误或数字转换失败，跳过第 {i+1} 行: {row}", 2)
                    continue
                except Exception as e:
                    debug_log(f"处理CSV第 {i+1} 行时发生未知错误: {str(e)}，跳过该行: {row}", 3)
                    continue

        sorted_periods = sorted(periods_list, key=int)

        if not period_map:
             debug_log("从CSV中未解析到任何有效期号数据。", 2)
             return None, None

        debug_log(f"成功从CSV解析到 {len(period_map)} 期数据。", 1)
        return period_map, sorted_periods
    except Exception as e:
        debug_log(f"CSV数据读取或整体解析失败: {str(e)}", 3)
        return None, None


def find_matching_report(target_period):
    """查找匹配指定期号的分析报告"""
    debug_log(f"开始查找处理期号 {target_period} 的分析报告...")
    candidates = []

    report_files = glob.glob(os.path.join('.', REPORT_PATTERN))

    if not report_files:
         debug_log(f"未找到符合模式 '{REPORT_PATTERN}' 的分析报告文件。", 1)
         return None

    for file in report_files:
        content = robust_file_read(file)
        if not content:
            debug_log(f"跳过读取失败的报告文件: {file}", 2)
            continue

        match = re.search(r'数据期数范围:.*?第\s*(\d+)\s*期\s*至\s*第\s*(\d+)\s*期', content)
        if match and match.group(2).strip() == target_period:
            time_match = re.search(r'_(\d{8}_\d{6})\.', os.path.basename(file))
            if time_match:
                try:
                    timestamp = datetime.strptime(time_match.group(1), "%Y%m%d_%H%M%S")
                    candidates.append((timestamp, file))
                except ValueError:
                     debug_log(f"文件名时间戳格式不正确，跳过文件: {file}", 2)
                     continue
            else:
                 debug_log(f"文件名未包含时间戳，无法确定新旧，跳过文件: {file}", 2)


    if not candidates:
        debug_log(f"未找到处理期号 {target_period} 的分析报告", 3)
        return None

    candidates.sort(reverse=True)
    selected = candidates[0][1]
    debug_log(f"找到 {len(candidates)} 个匹配报告，选择最新: {selected}")
    return selected

def parse_recommendations(content):
    """解析推荐组合"""
    debug_log("解析推荐组合...")
    # Adjusted pattern to match the file format: 红球 [...] 蓝球 \d+
    pattern = re.compile(
        r'组合\s*\d+\s*:\s*红球\s*\[([\d\s,]+?)\]\s*蓝球\s*(\d+)',
        re.DOTALL
    )
    parsed_rec = []
    matches = pattern.findall(content)
    debug_log(f"推荐组合 Regex found {len(matches)} matches.", 1) # 添加调试日志
    if not matches:
        debug_log("推荐组合 Regex did not find any matches.", 1) # 添加调试日志


    for red_str_match, blue_str in matches:
        try:
            red_str = red_str_match.replace(' ', '').strip(',')
            if not red_str:
                 debug_log(f"发现空红球字符串，跳过组合。", 2)
                 continue
            red_balls_list = red_str.split(',')
            if len(red_balls_list) != 6:
                 debug_log(f"发现红球数量不是6个 ({len(red_balls_list)}): '{red_str_match}'，跳过组合。", 2)
                 continue

            red_balls = sorted(map(int, red_balls_list))
            blue_ball = int(blue_str)
            parsed_rec.append((red_balls, blue_ball))

        except ValueError:
            debug_log(f"解析推荐组合时发现无效数字: 红球 '{red_str_match}', 蓝球 '{blue_str}'", 2)
            continue
        except Exception as e:
             debug_log(f"解析推荐组合时发生未知错误: {str(e)}，跳过组合。", 3)
             continue

    debug_log(f"成功解析到 {len(parsed_rec)} 个有效推荐组合。", 1) # 修正日志，反映有效组合数
    return parsed_rec[:5]


def parse_complex(content):
    """解析复式组合"""
    debug_log("解析复式组合...")
    # 首先找到红球部分
    match_red = re.search(
        r'7\+7复式选号.*?选择的7个红球:\s*\[([\d\s,]+?)\]',
        content, re.DOTALL
    )

    complex_red = []
    if match_red:
        debug_log("复式红球 Regex found a match.", 1) # 添加调试日志
        try:
            complex_red_str = match_red.group(1).replace(' ', '').strip(',')
            if complex_red_str:
                 complex_red = sorted(map(int, complex_red_str.split(',')))
            else:
                 debug_log("解析到空的复式红球字符串。", 2)
        except ValueError:
            debug_log(f"解析复式红球时发现无效数字: '{match_red.group(1)}'", 2)
        except Exception as e:
             debug_log(f"解析复式红球时发生未知错误: {str(e)}", 3)
    else:
        debug_log("复式红球 Regex did not find a match.", 1) # 添加调试日志
        return [], [] # 如果没找到红球，直接返回空列表


    # 然后找到蓝球部分，从红球匹配结束位置之后开始查找
    complex_blue = []
    # 只有找到了红球部分 (match_red is not None)，才尝试找蓝球
    # search_start_pos = match_red.end() # 从红球匹配结束位置开始
    # 使用 content[match_red.end():] 可以确保蓝球在红球之后
    match_blue = re.search(
        r'选择的7个蓝球:\s*\[([\d\s,]+?)\]',
        content[match_red.end():], # 在红球匹配后的内容中查找
        re.DOTALL
    )

    if match_blue:
        debug_log("复式蓝球 Regex found a match.", 1) # 添加调试日志
        try:
            complex_blue_str = match_blue.group(1).replace(' ', '').strip(',')
            if complex_blue_str:
                 complex_blue = sorted(map(int, complex_blue_str.split(',')))
            else:
                debug_log("解析到空的复式蓝球字符串。", 2)

        except ValueError:
            debug_log(f"解析复式蓝球时发现无效数字: '{match_blue.group(1)}'", 2)
        except Exception as e:
             debug_log(f"解析复式蓝球时发生未知错误: {str(e)}", 3)
    else:
         debug_log("复式蓝球 Regex did not find a match after red.", 1) # 添加调试日志


    # 简单验证一下数量，复式至少需要6个红球和1个蓝球才能生成有效投注
    if len(complex_red) < 6 or len(complex_blue) < 1:
         if complex_red or complex_blue:
             debug_log(f"解析到不完整的复式号码（红球需>=6，蓝球需>=1）: 红球 {len(complex_red)}个, 蓝球 {len(complex_blue)}个", 2)
         # 即使解析到了不完整的号码，如果数量不足以生成投注，也返回空列表
         return [], []

    debug_log(f"成功解析复式号码: 红球 {complex_red}, 蓝球 {complex_blue}", 1)
    return complex_red, complex_blue


def generate_complex_tickets(reds, blues):
    """从复式号码中生成所有可能的 6红+蓝 投注组合"""
    if len(reds) < 6 or not blues:
        debug_log(f"复式号码不足，无法生成投注: 红球 {len(reds)}个, 蓝球 {len(blues)}个", 1)
        return []

    tickets = []
    max_reasonable_reds = 12
    max_reasonable_blues = 16
    max_generated_tickets = 5000 # 硬性限制最大生成的投注数量

    if len(reds) > max_reasonable_reds or len(blues) > max_reasonable_blues:
        debug_log(f"复式号码数量过多 ({len(reds)}红, {len(blues)}蓝)，可能生成大量组合，跳过生成复式投注。", 2)
        return []

    try:
        from math import comb
        possible_combinations_count = comb(len(reds), 6) * len(blues)
    except AttributeError: # Fallback for older Python versions
         def combinations_count(n, k):
             if k < 0 or k > n: return 0
             if k == 0 or k == n: return 1
             if k > n // 2: k = n - k
             res = 1
             for i in range(k):
                 res = res * (n - i) // (i + 1)
             return res
         possible_combinations_count = combinations_count(len(reds), 6) * len(blues)

    if possible_combinations_count > max_generated_tickets:
         debug_log(f"复式号码可生成 {possible_combinations_count} 注，超过硬性限制 {max_generated_tickets} 注，跳过生成复式投注。", 2)
         return []

    try:
        for combo in combinations(reds, 6):
            for blue in blues:
                 tickets.append((sorted(list(combo)), blue))
        debug_log(f"从复式号码中成功生成了 {len(tickets)} 注投注。", 1)
        return tickets
    except Exception as e:
         debug_log(f"生成复式投注时发生异常: {str(e)}", 3)
         return []


def calculate_prize(tickets, prize_red, prize_blue):
    """计算奖金"""
    prize_red_set = set(prize_red)
    breakdown = {k:0 for k in PRIZE_TABLE}
    total_prize = 0

    for red, blue in tickets:
        matched_red = len(set(red) & prize_red_set)
        matched_blue = blue == prize_blue

        level = None

        if matched_red == 6:
            level = 1 if matched_blue else 2
        elif matched_red == 5:
            level = 3 if matched_blue else 4
        elif matched_red == 4:
            level = 4 if matched_blue else 5
        elif matched_red == 3 and matched_blue:
            level = 5
        elif matched_blue:
            level = 6

        if level is not None:
            breakdown[level] += 1

    for level, count in breakdown.items():
        if level in PRIZE_TABLE:
            total_prize += PRIZE_TABLE[level] * count
        else:
            debug_log(f"发现未知中奖级别: {level}", 2)

    debug_log(f"奖金计算明细: {breakdown}", 1)
    return total_prize, breakdown

def manage_report(new_entry=None, new_error=None):
    """维护主报告文件"""
    normal_marker = "==== NORMAL RECORDS ===="
    error_marker = "==== ERROR LOGS ===="

    normal_entries = []
    error_logs = []
    report_content = []

    if os.path.exists(MAIN_REPORT_FILE):
        try:
            # 使用 'r+' 模式以便读写，同时保持文件指针在开头
            with open(MAIN_REPORT_FILE, 'r+', encoding='utf-8') as f:
                report_content = [line.rstrip() for line in f]
                # 清空文件内容以便重写
                f.seek(0)
                f.truncate()
        except Exception as e:
            debug_log(f"读取或清空主报告文件 '{MAIN_REPORT_FILE}' 失败: {str(e)}", 3)
            report_content = []


    normal_start = -1
    error_start = -1
    for i, line in enumerate(report_content):
        if line.startswith(normal_marker):
            normal_start = i
        elif line.startswith(error_marker):
            error_start = i

    if normal_start != -1:
        normal_entries = report_content[normal_start + 1 : error_start if error_start != -1 else len(report_content)]
        normal_entries = [line for line in normal_entries if line.strip()]

    if error_start != -1:
        error_logs = report_content[error_start + 1 :]
        error_logs = [line for line in error_logs if line.strip()]


    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if new_entry:
        period = new_entry.get('period', 'N/A')
        red = new_entry.get('red', 'N/A')
        blue = new_entry.get('blue', 'N/A')
        prize = new_entry.get('prize', 'N/A')

        entry_block = [
            f"[记录 {timestamp}]",
            f"分析期号(对应开奖期号): {period}",
            f"开奖号码: 红{red} 蓝{blue}",
            f"中奖金额: {prize:,}元",
            "-"*40
        ]
        normal_entries = entry_block + normal_entries

    if new_error:
        error_logs = [f"[错误 {timestamp}] {new_error}"] + error_logs

    lines_per_normal_entry_block = 5
    max_normal_lines = MAX_NORMAL_RECORDS * lines_per_normal_entry_block
    record_starts = [i for i, line in enumerate(normal_entries) if line.startswith("[记录 ")]
    if len(record_starts) > MAX_NORMAL_RECORDS:
        start_index_to_keep = record_starts[-MAX_NORMAL_RECORDS]
        normal_entries = normal_entries[start_index_to_keep:]

    error_logs = error_logs[:MAX_ERROR_LOGS]


    try:
        with open(MAIN_REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write(f"{normal_marker}\n")
            if normal_entries:
                f.write("\n".join(normal_entries) + "\n")

            f.write(f"\n{error_marker}\n")
            if error_logs:
                f.write("\n".join(error_logs) + "\n")

    except Exception as e:
        debug_log(f"写入主报告文件 '{MAIN_REPORT_FILE}' 失败: {str(e)}", 3)

def main_process():
    """主处理流程"""
    debug_log("====== 主流程启动 ======", 1)

    csv_content = robust_file_read(CSV_FILE)
    if not csv_content:
        manage_report(new_error=f"CSV文件 '{CSV_FILE}' 读取失败或文件不存在")
        debug_log(f"处理停止：CSV文件 '{CSV_FILE}' 读取失败或文件不存在。", 3)
        return

    period_map, sorted_periods = get_period_data(csv_content)
    if period_map is None or sorted_periods is None:
        manage_report(new_error=f"CSV文件 '{CSV_FILE}' 数据解析失败")
        debug_log(f"处理停止：CSV数据解析失败。", 3)
        return

    if len(sorted_periods) < 2:
        manage_report(new_error=f"CSV数据不足两期 ({len(sorted_periods)} 期)，无法确定下一期开奖结果")
        debug_log(f"处理停止：CSV数据不足两期 ({len(sorted_periods)} 期)。", 2)
        return

    latest_period_in_csv = sorted_periods[-1]
    debug_log(f"CSV最新期号: {latest_period_in_csv}")

    analysis_file = find_matching_report(latest_period_in_csv)
    if not analysis_file:
        manage_report(new_error=f"未找到处理期号 {latest_period_in_csv} 的分析报告")
        debug_log(f"处理停止：未找到处理期号 {latest_period_in_csv} 的分析报告。", 2)
        return

    content = robust_file_read(analysis_file)
    if not content:
        manage_report(new_error=f"分析报告 '{analysis_file}' 读取失败")
        debug_log(f"处理停止：分析报告 '{analysis_file}' 读取失败。", 3)
        return

    rec_tickets = parse_recommendations(content)
    complex_red, complex_blue = parse_complex(content)
    complex_tickets = generate_complex_tickets(complex_red, complex_blue)
    all_tickets = rec_tickets + complex_tickets

    if not all_tickets:
        manage_report(new_error=f"未能从分析报告 '{os.path.basename(analysis_file)}' 中解析出任何投注组合")
        debug_log("处理继续：未能从分析报告中解析出任何投注组合，将计算奖金为0。", 2)
        # 继续执行，计算奖金会是0

    try:
        latest_period_index = sorted_periods.index(latest_period_in_csv)
    except ValueError:
        manage_report(new_error=f"内部错误: 未在 sorted_periods 中找到期号 {latest_period_in_csv}")
        debug_log(f"处理停止：内部错误，未在 sorted_periods 中找到期号 {latest_period_in_csv}。", 3)
        return

    if latest_period_index + 1 >= len(sorted_periods):
        try:
            expected_next_period = str(int(latest_period_in_csv) + 1).zfill(7)
        except ValueError:
             expected_next_period = "下一期"

        manage_report(new_error=f"CSV数据不包含期号 {latest_period_in_csv} 的下一期开奖结果 (预期的下一期期号可能是 {expected_next_period})")
        debug_log(f"处理停止：CSV中没有期号 {latest_period_in_csv} 的下一期数据。", 2)
        return

    next_period_in_csv = sorted_periods[latest_period_index + 1]
    if next_period_in_csv not in period_map:
         manage_report(new_error=f"内部错误: 期号 {next_period_in_csv} 在 sorted_periods 中，但不在 period_map 中")
         debug_log(f"处理停止：内部错误，期号 {next_period_in_csv} 在 sorted_periods 中，但不在 period_map 中。", 3)
         return

    prize_data = period_map[next_period_in_csv]
    debug_log(f"找到对应分析报告期号 {latest_period_in_csv} 的下一期开奖数据: 期号 {next_period_in_csv}", 1)


    total_prize, breakdown = calculate_prize(all_tickets, prize_data['red'], prize_data['blue'])

    manage_report(new_entry={
        'period': next_period_in_csv,
        'red': prize_data['red'],
        'blue': prize_data['blue'],
        'prize': total_prize
    })

    debug_log(f"处理完成！分析期号 {latest_period_in_csv} (开奖期号 {next_period_in_csv}) 的预测，中奖金额: {total_prize:,}元。", 1)
    debug_log("====== 主流程结束 ======", 1)


if __name__ == "__main__":
    try:
        main_process()
    except Exception as e:
        error_message = f"主流程发生未处理异常: {type(e).__name__} - {str(e)}"
        manage_report(new_error=error_message)
        debug_log(error_message, 3)

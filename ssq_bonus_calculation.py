import os
import re
import glob
import csv
from itertools import combinations
from datetime import datetime

# 配置参数
REPORT_PATTERN = "ssq_analysis_output_*.txt"
CSV_FILE = "shuangseqiu.csv"
MAIN_REPORT_FILE = "latest_ssq_calculation.txt"
MAX_NORMAL_RECORDS = 10 # 主报告保留的普通记录数量 (这里指的是记录块的数量)
MAX_ERROR_LOGS = 20    # 主报告保留的错误日志数量

# 日志配置: 1=INFO, 2=WARNING, 3=ERROR
# 设置为 1 将显示所有 INFO, WARNING, ERROR 日志
# 设置为 2 将只显示 WARNING, ERROR 日志
# 设置为 3 将只显示 ERROR 日志
MIN_LOG_LEVEL = 1 # <-- 设置为 1 以恢复 INFO 级别的日志

# 奖金对照表
PRIZE_TABLE = {
    1: 5_000_000, # 一等奖
    2: 500_000,   # 二等奖
    3: 3_000,     # 三等奖
    4: 200,       # 四等奖
    5: 10,        # 五等奖
    6: 5         # 六等奖
}

# 移除了不再使用的 PeriodIntegrityError 异常

class PrizeDataNotFound(Exception):
    """开奖数据缺失异常"""
    def __init__(self, target_period):
        super().__init__(f"找不到期号 {target_period} 的开奖数据")
        self.target_period = target_period

def debug_log(message, level=1):
    """分级调试日志"""
    # 如果消息级别低于最低显示级别，则不打印
    if level < MIN_LOG_LEVEL:
        return

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
    prefix = prefixes.get(level, "[DEBUG]") # 默认仍然是 DEBUG
    color = color_codes.get(level, "\033[90m")
    print(f"{color}{prefix} {datetime.now().strftime('%H:%M:%S')} {message}{reset_code}")

def robust_file_read(file_path):
    """带编码回退的文件读取"""
    encodings = ['utf-8', 'gbk', 'gb2312']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            debug_log(f"成功使用 {encoding} 编码读取文件: {file_path}", level=1)
            return content
        except UnicodeDecodeError:
            debug_log(f"尝试使用 {encoding} 编码读取文件 {file_path} 失败。", level=1)
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
            # 确保行有足够列且期号是7位数字
            if len(row) >= 4 and re.match(r'^\d{7}$', row[0]):
                period = row[0].strip()
                # date_str = row[1].strip() # 日期目前未使用
                red_str = row[2].strip()
                blue_str = row[3].strip()

                try:
                    red_balls = sorted(list(map(int, red_str.split(','))))
                    blue_ball = int(blue_str)

                    # 简单验证红球数量是否为6
                    if len(red_balls) != 6:
                         debug_log(f"CSV中期号 {period} 红球数量不是6个 ({len(red_balls)}): '{red_str}'，跳过该期数据。", 2)
                         continue
                    # 简单验证红球和蓝球范围
                    if not all(1 <= r <= 33 for r in red_balls) or not (1 <= blue_ball <= 16):
                         debug_log(f"CSV中期号 {period} 号码超出有效范围: 红球 {red_balls}, 蓝球 {blue_ball}，跳过该期数据。", 2)
                         continue


                    period_map[period] = {
                        'date': row[1].strip(), # 保留日期， although currently not used
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
    """查找匹配指定期号(作为数据截止期)的分析报告"""
    debug_log(f"开始查找数据截止期为 {target_period} 的分析报告...", 1) # INFO level
    # debug_log(f"find_matching_report target_period: '{target_period}'", level=1) # Removed overly detailed debug
    candidates = []

    report_files = glob.glob(os.path.join('.', REPORT_PATTERN))

    if not report_files:
         debug_log(f"未找到符合模式 '{REPORT_PATTERN}' 的分析报告文件。", 1)
         return None

    # debug_log(f"Found {len(report_files)} files matching pattern: {report_files}", level=1) # Removed overly detailed debug

    for file in report_files:
        # debug_log(f"Reading file: {file}", level=1) # Removed overly detailed debug
        content = robust_file_read(file)
        if not content:
            debug_log(f"跳过读取失败的报告文件: {file}", 2) # WARNING level
            continue

        # 查找数据期数范围的结束期号
        match = re.search(r'数据期数范围:.*?第\s*(\d+)\s*期\s*至\s*第\s*(\d+)\s*期', content)

        if match:
            extracted_start_period = match.group(1).strip()
            extracted_end_period = match.group(2).strip()
            # debug_log(f"  File '{file}': Extracted range '{extracted_start_period}' to '{extracted_end_period}'", level=1) # Removed overly detailed debug

            if extracted_end_period == target_period:
                # debug_log(f"  File '{file}': Extracted end period '{extracted_end_period}' MATCHES target '{target_period}'", level=1) # Removed overly detailed debug
                time_match = re.search(r'_(\d{8}_\d{6})\.', os.path.basename(file))
                if time_match:
                    try:
                        timestamp = datetime.strptime(time_match.group(1), "%Y%m%d_%H%M%S")
                        candidates.append((timestamp, file))
                        # debug_log(f"  File '{file}': Added as candidate with timestamp {timestamp}", level=1) # Removed overly detailed debug
                    except ValueError:
                         debug_log(f"文件名时间戳格式不正确，跳过文件: {file}", 2) # WARNING level
                         continue
                else:
                     debug_log(f"文件名未包含时间戳，无法确定新旧，跳过文件: {file}", 2) # WARNING level
            # else: # No need to log non-matches at INFO level
             #    debug_log(f"  File '{file}': Extracted end period '{extracted_end_period}' DOES NOT MATCH target '{target_period}'", level=1)

        # else: # No need to log regex non-matches at INFO level
            # debug_log(f"  File '{file}': Regex '数据期数范围:.*?第\\s*(\\d+)\\s*期\\s*至\\s*第\\s*(\\d+)\\s*期' did not match content.", level=1)


    if not candidates:
        debug_log(f"未找到数据截止期为 {target_period} 的分析报告", 3) # ERROR level
        return None

    candidates.sort(reverse=True)
    selected = candidates[0][1]
    debug_log(f"找到 {len(candidates)} 个匹配报告，选择最新: {selected}", 1) # INFO level
    return selected

def parse_recommendations(content):
    """解析推荐组合"""
    debug_log("解析推荐组合...", 1) # INFO level
    # Pattern to match: 组合 X: 红球 [...] 蓝球 \d+
    pattern = re.compile(
        r'组合\s*\d+\s*:\s*红球\s*\[([\d\s,]+?)\]\s*蓝球\s*(\d+)',
        re.DOTALL
    )
    parsed_rec = []
    matches = pattern.findall(content)
    # debug_log(f"推荐组合 Regex found {len(matches)} matches.", 1) # Removed overly detailed debug
    if not matches:
        debug_log("推荐组合 Regex did not find any matches.", 1) # INFO level


    for red_str_match, blue_str in matches:
        try:
            red_str = red_str_match.replace(' ', '').strip(',')
            if not red_str:
                 debug_log(f"发现空红球字符串，跳过组合。", 2) # WARNING level
                 continue
            red_balls_list = red_str.split(',')

            red_balls = sorted(map(int, red_balls_list))
            blue_ball = int(blue_str)

            # 验证红球数量和范围
            if len(red_balls) != 6:
                 debug_log(f"发现推荐组合红球数量不是6个 ({len(red_balls)}): '{red_str_match}'，跳过组合。", 2) # WARNING level
                 continue
            if not all(1 <= r <= 33 for r in red_balls) or not (1 <= blue_ball <= 16):
                 debug_log(f"发现推荐组合号码超出有效范围: 红球 {red_balls}, 蓝球 {blue_ball}，跳过组合。", 2) # WARNING level
                 continue

            parsed_rec.append((red_balls, blue_ball))

        except ValueError:
            debug_log(f"解析推荐组合时发现无效数字: 红球 '{red_str_match}', 蓝球 '{blue_str}'", 2) # WARNING level
            continue
        except Exception as e:
             debug_log(f"解析推荐组合时发生未知错误: {str(e)}，跳过组合。", 3) # ERROR level
             continue

    debug_log(f"成功解析到 {len(parsed_rec)} 个有效推荐组合。", 1) # INFO level
    # 返回所有解析到的推荐组合
    return parsed_rec


def parse_complex(content):
    """解析复式组合"""
    debug_log("解析复式组合...", 1) # INFO level
    # 首先找到红球部分
    match_red = re.search(
        r'复式选号.*?选择的.*?个红球:\s*\[([\d\s,]+?)\]',
        content, re.DOTALL
    )

    complex_red = []
    if match_red:
        # debug_log("复式红球 Regex found a match.", 1) # Removed overly detailed debug
        try:
            complex_red_str = match_red.group(1).replace(' ', '').strip(',')
            if complex_red_str:
                 complex_red = sorted(map(int, complex_red_str.split(',')))
            else:
                 debug_log("解析到空的复式红球字符串。", 2) # WARNING level
        except ValueError:
            debug_log(f"解析复式红球时发现无效数字: '{match_red.group(1)}'", 2) # WARNING level
        except Exception as e:
             debug_log(f"解析复式红球时发生未知错误: {str(e)}", 3) # ERROR level
    else:
        debug_log("复式红球 Regex did not find a match.", 1) # INFO level
        return [], []


    # 然后找到蓝球部分，从红球匹配结束位置之后开始查找
    complex_blue = []
    if match_red: # Only look for blue if red was found
        match_blue = re.search(
            r'选择的.*?个蓝球:\s*\[([\d\s,]+?)\]',
            content[match_red.end():],
            re.DOTALL
        )

        if match_blue:
            # debug_log("复式蓝球 Regex found a match.", 1) # Removed overly detailed debug
            try:
                complex_blue_str = match_blue.group(1).replace(' ', '').strip(',')
                if complex_blue_str:
                     complex_blue = sorted(map(int, complex_blue_str.split(',')))
                else:
                    debug_log("解析到空的复式蓝球字符串。", 2) # WARNING level

            except ValueError:
                debug_log(f"解析复式蓝球时发现无效数字: '{match_blue.group(1)}'", 2) # WARNING level
            except Exception as e:
                 debug_log(f"解析复式蓝球时发生未知错误: {str(e)}", 3) # ERROR level
        else:
             debug_log("复式蓝球 Regex did not find a match after red.", 1) # INFO level


    # 简单验证一下数量和范围
    if len(complex_red) < 6 or len(complex_blue) < 1:
         if complex_red or complex_blue: # Only log if some numbers were parsed but not enough
             debug_log(f"解析到不完整的复式号码（红球需>=6，蓝球需>=1）或范围错误，跳过。 红球 {len(complex_red)}个, 蓝球 {len(complex_blue)}个", 2) # WARNING level
         return [], [] # Always return empty if not enough numbers

    if not all(1 <= r <= 33 for r in complex_red) or not all(1 <= b <= 16 for b in complex_blue):
         debug_log(f"解析到复式号码超出有效范围，跳过。 红球 {complex_red}, 蓝球 {complex_blue}", 2) # WARNING level
         return [], []


    debug_log(f"成功解析复式号码: 红球 {complex_red}, 蓝球 {complex_blue}", 1) # INFO level
    return complex_red, complex_blue


def generate_complex_tickets(reds, blues):
    """从复式号码中生成所有可能的 6红+蓝 投注组合"""
    if len(reds) < 6 or not blues:
        debug_log(f"复式号码不足，无法生成投注: 红球 {len(reds)}个, 蓝球 {len(blues)}个", 1) # INFO level
        return []

    tickets = []
    # 可以调整这些限制，但要注意内存和计算时间
    max_reasonable_reds = 15
    max_reasonable_blues = 16
    max_generated_tickets = 20000 # 硬性限制最大生成的投注数量


    if len(reds) > max_reasonable_reds or len(blues) > max_reasonable_blues:
        debug_log(f"复式号码数量过多 ({len(reds)}红, {len(blues)}蓝)，可能生成大量组合，跳过生成复式投注。", 2) # WARNING level
        return []

    try:
        from math import comb
        possible_combinations_count = comb(len(reds), 6) * len(blues)
    except AttributeError: # Fallback for older Python versions (Python < 3.8)
         def combinations_count(n, k):
             if k < 0 or k > n: return 0
             if k == 0 or k == n: return 1
             if k > n // 2: k = n - k
             res = 1
             for i in range(k):
                 res = res * (n - i) // (i + 1)
             return res
         possible_combinations_count = combinations_count(len(reds), 6) * len(blues)
    except Exception as e:
         debug_log(f"计算复式组合数量时发生异常: {str(e)}", 3) # ERROR level
         return []


    if possible_combinations_count > max_generated_tickets:
         debug_log(f"复式号码可生成 {possible_combinations_count:,} 注，超过硬性限制 {max_generated_tickets:,} 注，跳过生成复式投注。", 2) # WARNING level
         return []

    try:
        for combo in combinations(reds, 6):
            for blue in blues:
                 tickets.append((sorted(list(combo)), blue))
        debug_log(f"从复式号码中成功生成了 {len(tickets):,} 注投注。", 1) # INFO level
        return tickets
    except Exception as e:
         debug_log(f"生成复式投注时发生异常: {str(e)}", 3) # ERROR level
         return []


def calculate_prize(tickets, prize_red, prize_blue):
    """
    计算奖金并返回中奖号码详情。
    返回: total_prize, breakdown, winning_tickets_list ([(red_balls, blue_ball, level), ...])
    """
    prize_red_set = set(prize_red)
    breakdown = {k:0 for k in PRIZE_TABLE}
    total_prize = 0
    winning_tickets_list = []

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
            winning_tickets_list.append((red, blue, level))

    for level, count in breakdown.items():
        if level in PRIZE_TABLE:
            total_prize += PRIZE_TABLE[level] * count
        elif count > 0:
             debug_log(f"发现未知中奖级别 {level} 有 {count} 注，请检查 calculate_prize 逻辑或 PRIZE_TABLE", 2) # WARNING level


    # debug_log(f"中奖号码详情: {winning_tickets_list}", 1) # Avoid logging potentially huge list
    debug_log(f"奖金计算明细: {breakdown}", 1) # INFO level

    return total_prize, breakdown, winning_tickets_list

# ====== 修改 format_winning_tickets 以显示匹配号码 ======
def format_winning_tickets(winning_list, winning_red, winning_blue):
    """格式化中奖号码列表为字符串列表，并指示匹配的号码"""
    formatted = []
    winning_red_set = set(winning_red)
    for red, blue, level in winning_list:
        # 找到匹配的红球
        matched_red_balls = sorted(list(set(red) & winning_red_set))
        # 检查蓝球是否匹配
        matched_blue_ball = blue if blue == winning_blue else None

        red_str_parts = []
        for r_ball in sorted(red): # 遍历投注红球，按顺序显示
             if r_ball in matched_red_balls:
                  red_str_parts.append(f"**{r_ball}**") # 匹配的红球加粗或特殊标记 (这里用 **)
             else:
                  red_str_parts.append(str(r_ball))

        blue_str = f"**{blue}**" if matched_blue_ball is not None else str(blue)

        # 命中详情字符串
        matched_details = []
        if matched_red_balls:
             matched_details.append(f"命中红球: {','.join(map(str, matched_red_balls))}")
        if matched_blue_ball is not None:
             matched_details.append(f"命中蓝球: {matched_blue_ball}")

        details_str = ""
        if matched_details:
             details_str = " (" + "; ".join(matched_details) + ")"


        formatted.append(f"  - 红[{', '.join(red_str_parts)}] 蓝{blue_str} ({level}等奖){details_str}")
    return formatted
# ====== 修改 format_winning_tickets 结束 ======


def manage_report(new_entry=None, new_error=None):
    """维护主报告文件"""
    normal_marker = "==== EVALUATION RECORDS ====="
    error_marker = "==== ERROR LOGS ===="

    normal_entries = []
    error_logs = []
    report_content = []

    if os.path.exists(MAIN_REPORT_FILE):
        try:
            content = robust_file_read(MAIN_REPORT_FILE) # Uses debug_log internally
            if content is not None:
                report_content = [line.rstrip() for line in content.splitlines()]

            with open(MAIN_REPORT_FILE, 'w', encoding='utf-8') as f:
                 f.seek(0)
                 f.truncate()

        except Exception as e:
            debug_log(f"读取或清空主报告文件 '{MAIN_REPORT_FILE}' 失败: {str(e)}", 3) # ERROR level
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
        evaluation_period = new_entry.get('evaluation_period', 'N/A')
        report_cutoff_period = new_entry.get('report_cutoff_period', 'N/A')
        winning_red = new_entry.get('winning_red', 'N/A')
        winning_blue = new_entry.get('winning_blue', 'N/A')
        overall_prize = new_entry.get('overall_prize', 'N/A')

        rec_data = new_entry.get('recommended', {})
        complex_data = new_entry.get('complex', {})

        winning_red_str = ','.join(map(str, sorted(winning_red))) if isinstance(winning_red, list) else str(winning_red)
        winning_blue_str = str(winning_blue)

        entry_block = [
            f"[评估记录 {timestamp}]",
            f"评估期号 (实际开奖): {evaluation_period}",
            f"评估报告数据截止期: {report_cutoff_period}",
            f"实际开奖号码: 红[{winning_red_str}] 蓝{winning_blue_str}",
            f"总中奖金额: {overall_prize:,}元",
            ""
        ]

        entry_block.append("--- 普通推荐中奖详情 ---")
        if rec_data and (rec_data.get('prize', 0) > 0 or rec_data.get('winners')):
            entry_block.append(f"普通推荐总奖金: {rec_data.get('prize', 0):,}元")
            breakdown_str = ", ".join([f"{level}等奖: {count}注" for level, count in sorted(rec_data.get('breakdown', {}).items()) if count > 0])
            entry_block.append(f"普通推荐奖金明细: {breakdown_str if breakdown_str else '无中奖'}")
            rec_winners = rec_data.get('winners', [])
            if rec_winners:
                entry_block.append("中奖号码:")
                # ====== 在调用 format_winning_tickets 时传入中奖号码 ======
                entry_block.extend(format_winning_tickets(rec_winners, winning_red, winning_blue))
                # ====== 修改结束 ======
            else:
                entry_block.append("无中奖号码")
        else:
             entry_block.append("无普通推荐投注或未中奖")
        entry_block.append("")


        entry_block.append("--- 复式生成投注中奖详情 ---")
        if complex_data and (complex_data.get('prize', 0) > 0 or complex_data.get('winners')):
            entry_block.append(f"复式生成总奖金: {complex_data.get('prize', 0):,}元")
            breakdown_str = ", ".join([f"{level}等奖: {count}注" for level, count in sorted(complex_data.get('breakdown', {}).items()) if count > 0])
            entry_block.append(f"复式生成奖金明细: {breakdown_str if breakdown_str else '无中奖'}")
            complex_winners = complex_data.get('winners', [])
            if complex_winners:
                entry_block.append("中奖号码:")
                # ====== 在调用 format_winning_tickets 时传入中奖号码 ======
                entry_block.extend(format_winning_tickets(complex_winners, winning_red, winning_blue))
                # ====== 修改结束 ======
            else:
                entry_block.append("无中奖号码")
        else:
             entry_block.append("无复式生成投注或未中奖")
        entry_block.append("")

        entry_block.append("="*60)
        entry_block.append("")

        normal_entries = entry_block + normal_entries

    if new_error:
        error_logs = [f"[错误 {timestamp}] {new_error}"] + error_logs

    record_starts = [i for i, line in enumerate(normal_entries) if line.startswith("[评估记录 ")]

    if len(record_starts) > MAX_NORMAL_RECORDS:
        start_index_to_keep = record_starts[-MAX_LOG_LEVEL] # Should be MAX_NORMAL_RECORDS
        normal_entries = normal_entries[start_index_to_keep:]

    error_logs = error_logs[:MAX_ERROR_LOGS]

    try:
        with open(MAIN_REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write(f"{normal_marker}\n")
            if normal_entries:
                f.write("\n".join(normal_entries).strip() + "\n")

            f.write(f"\n{error_marker}\n")
            if error_logs:
                f.write("\n".join(error_logs).strip() + "\n")

    except Exception as e:
        debug_log(f"写入主报告文件 '{MAIN_REPORT_FILE}' 失败: {str(e)}", 3) # ERROR level

def main_process():
    """主处理流程"""
    debug_log("====== 主流程启动 ======", 1) # INFO level

    csv_content = robust_file_read(CSV_FILE) # Uses debug_log internally
    if not csv_content:
        manage_report(new_error=f"CSV文件 '{CSV_FILE}' 读取失败或文件不存在")
        debug_log(f"处理停止：CSV文件 '{CSV_FILE}' 读取失败或文件不存在。", 3) # ERROR level
        return

    period_map, sorted_periods = get_period_data(csv_content) # Uses debug_log internally
    if period_map is None or sorted_periods is None:
        manage_report(new_error=f"CSV文件 '{CSV_FILE}' 数据解析失败")
        debug_log(f"处理停止：CSV数据解析失败。", 3) # ERROR level
        return

    # ====== 按照“报告截止期 X-1，评估期 X”的逻辑处理 ======

    # 需要至少两期数据：最新一期(X)用于开奖结果，倒数第二期(X-1)用于匹配分析报告
    if len(sorted_periods) < 2:
        manage_report(new_error=f"CSV数据不足两期 ({len(sorted_periods)} 期)。需要至少两期来确定评估期号和报告截止期。")
        debug_log(f"处理停止：CSV数据不足两期 ({len(sorted_periods)} 期)。", 2) # WARNING level
        return

    # CSV最新期号 (期号 X) - 这是用于评估的实际开奖期号
    evaluation_period = sorted_periods[-1] # 例如 2025054
    debug_log(f"将用于评估的实际开奖期号: {evaluation_period}", 1) # INFO level

    # 倒数第二期号 (期号 X-1) - 这是分析报告的数据截止期
    try:
        report_data_cutoff_period = sorted_periods[-2] # 例如 2025053
    except IndexError:
         # 这个检查理论上被 len(sorted_periods) < 2 包含了，但再加一层更安全
         manage_report(new_error=f"内部错误：获取CSV倒数第二期期号失败。")
         debug_log(f"处理停止：内部错误，获取CSV倒数第二期期号失败。", 3) # ERROR level
         return

    debug_log(f"查找数据截止期为 {report_data_cutoff_period} 的分析报告...", 1) # INFO level

    # 查找数据范围截止到期号 X-1 的分析报告
    analysis_file = find_matching_report(report_data_cutoff_period) # Uses debug_log internally
    if not analysis_file:
        # 这里的错误消息明确指出是没找到截止期为 X-1 的报告
        manage_report(new_error=f"未找到数据截止期为 {report_data_cutoff_period} 的分析报告")
        debug_log(f"处理停止：未找到数据截止期为 {report_data_cutoff_period} 的分析报告。", 2) # WARNING level
        return

    content = robust_file_read(analysis_file) # Uses debug_log internally
    if not content:
        manage_report(new_error=f"分析报告 '{analysis_file}' 读取失败")
        debug_log(f"处理停止：分析报告 '{analysis_file}' 读取失败。", 3) # ERROR level
        return

    rec_tickets = parse_recommendations(content) # Uses debug_log internally
    complex_red, complex_blue = parse_complex(content) # Uses debug_log internally
    complex_tickets = generate_complex_tickets(complex_red, complex_blue) # Uses debug_log internally

    # 检查是否有任何投注被解析，如果没有则记录错误并继续（中奖金额为零，报告中显示无投注）
    if not rec_tickets and not complex_tickets:
         # 不停止处理，继续生成报告，报告会显示无中奖
         debug_log("未能从分析报告中解析出任何投注组合，将计算奖金为0。", 2) # WARNING level


    # 获取用于评估的实际开奖期号 (期号 X) 的数据
    if evaluation_period not in period_map:
         # 理论上不会发生，因为 evaluation_period 就是从 period_map 的 key 列表 sorted_periods 中获取的
         manage_report(new_error=f"内部错误: 用于评估的开奖期号 {evaluation_period} 在 sorted_periods 中，但不在 period_map 中")
         debug_log(f"处理停止：内部错误，用于评估的开奖期号 {evaluation_period} 不在 period_map 中。", 3) # ERROR level
         return

    prize_data = period_map[evaluation_period]
    winning_red_balls = prize_data['red']
    winning_blue_ball = prize_data['blue']
    debug_log(f"获取到评估期号 {evaluation_period} ({prize_data['date']}) 的实际开奖数据。", 1) # INFO level


    # ====== 结合详细报告逻辑 ======
    # 计算普通推荐的奖金和中奖详情
    rec_total_prize, rec_breakdown, rec_winning_list = calculate_prize(rec_tickets, winning_red_balls, winning_blue_ball) # Uses debug_log internally
    debug_log(f"普通推荐在期号 {evaluation_period} 中奖详情计算完毕。", 1) # INFO level

    # 计算复式生成投注的奖金和中奖详情
    complex_total_prize, complex_breakdown, complex_winning_list = calculate_prize(complex_tickets, winning_red_balls, winning_blue_ball) # Uses debug_log internally
    debug_log(f"复式生成投注在期号 {evaluation_period} 中奖详情计算完毕。", 1) # INFO level

    # 计算总奖金
    overall_total_prize = rec_total_prize + complex_total_prize

    # 准备传递给 manage_report 的数据
    report_entry_data = {
        'evaluation_period': evaluation_period, # 评估期号 (实际开奖期号 X)
        'report_cutoff_period': report_data_cutoff_period, # 报告数据截止期 (X-1)
        'winning_red': winning_red_balls, # <-- 传递实际开奖红球到报告管理函数
        'winning_blue': winning_blue_ball, # <-- 传递实际开奖蓝球到报告管理函数
        'overall_prize': overall_total_prize,
        'recommended': {
            'prize': rec_total_prize,
            'breakdown': rec_breakdown,
            'winners': rec_winning_list
        },
        'complex': {
            'prize': complex_total_prize,
            'breakdown': complex_breakdown,
            'winners': complex_winning_list
        }
    }

    # 调用 manage_report 记录详细结果
    manage_report(new_entry=report_entry_data) # Uses debug_log internally


    # 在所有处理完成后，检查是否有任何投注被解析，如果都没有则添加一条错误日志到报告文件（如果之前没添加过的话）
    if not rec_tickets and not complex_tickets:
         manage_report(new_error=f"未能从分析报告 '{os.path.basename(analysis_file)}' (数据截止 {report_data_cutoff_period}) 中解析出任何投注组合") # ERROR level


    debug_log(f"处理完成！评估报告 (数据截止 {report_data_cutoff_period}) 在期号 {evaluation_period} 开奖结果上的表现，总中奖金额: {overall_total_prize:,}元。", 1) # INFO level
    debug_log("====== 主流程结束 ======", 1) # INFO level


if __name__ == "__main__":
    try:
        main_process()
    except Exception as e:
        error_message = f"主流程发生未处理异常: {type(e).__name__} - {str(e)}"
        manage_report(new_error=error_message) # Uses debug_log internally
        debug_log(error_message, 3) # ERROR level

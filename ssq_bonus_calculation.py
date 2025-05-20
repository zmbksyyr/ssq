import os
import re
import glob
import csv
from itertools import combinations
from datetime import datetime
import traceback # For detailed error reporting

# 配置参数
REPORT_PATTERN = "ssq_analysis_log_*.txt"
CSV_FILE = "shuangseqiu.csv" # Ensure this file is in the CWD or provide an absolute path
MAIN_REPORT_FILE = "latest_ssq_calculation.txt"
MAX_NORMAL_RECORDS = 10
MAX_ERROR_LOGS = 20

# 日志配置: 1=INFO, 2=WARNING, 3=ERROR
MIN_LOG_LEVEL = 1
USE_COLOR_LOGS = False # Set to True to enable ANSI color codes in console output

# 奖金对照表
PRIZE_TABLE = {
    1: 5_000_000, 2: 500_000, 3: 3_000,
    4: 200, 5: 10, 6: 5
}

class PrizeDataNotFound(Exception):
    """开奖数据缺失异常"""
    def __init__(self, target_period):
        super().__init__(f"找不到期号 {target_period} 的开奖数据")
        self.target_period = target_period

def debug_log(message, level=1):
    """分级调试日志"""
    if level < MIN_LOG_LEVEL:
        return

    prefixes = {1: "[INFO]", 2: "[WARNING]", 3: "[ERROR]"}
    
    color_code_map = {}
    reset_code_str = ""
    if USE_COLOR_LOGS: # This flag now controls color output
        color_code_map = {
            1: "\033[94m", 2: "\033[93m", 3: "\033[91m"
        }
        reset_code_str = "\033[0m"

    prefix = prefixes.get(level, "[DEBUG]") # Default to [DEBUG] if level is not 1,2, or 3
    color = color_code_map.get(level, "") # Default to no color
    
    print(f"{color}{prefix} {datetime.now().strftime('%H:%M:%S')} {message}{reset_code_str}")

def robust_file_read(file_path):
    """带编码回退的文件读取"""
    # It's good practice to work with absolute paths internally if there's any ambiguity
    absolute_path = os.path.abspath(file_path)
    if not os.path.exists(absolute_path):
        debug_log(f"文件未找到: {absolute_path}", level=3)
        return None

    encodings = ['utf-8', 'gbk', 'gb2312']
    for encoding in encodings:
        try:
            with open(absolute_path, 'r', encoding=encoding) as f:
                content = f.read()
            debug_log(f"成功使用 {encoding} 编码读取文件: {absolute_path}", level=1)
            return content
        except UnicodeDecodeError:
            debug_log(f"尝试使用 {encoding} 编码读取文件 {absolute_path} 失败。", level=1)
            continue
        except FileNotFoundError: # Should be caught by os.path.exists, but defensive
            debug_log(f"文件未找到错误 (robust_file_read): {absolute_path}", level=3)
            return None
        except Exception as e:
            debug_log(f"读取文件 {absolute_path} 异常 (编码 {encoding})：{str(e)}", level=3)
            return None
    debug_log(f"无法使用所有尝试的编码读取文件: {absolute_path}", level=3)
    return None

def get_period_data(csv_content):
    """获取CSV期号数据并按期号排序"""
    period_map = {}
    periods_list = []
    if not csv_content:
        debug_log("get_period_data: 接收到空的CSV内容。", level=2)
        return None, None
    try:
        reader = csv.reader(csv_content.splitlines())
        for i, row in enumerate(reader):
            if len(row) >= 4 and re.match(r'^\d{7}$', row[0]):
                period, red_str, blue_str = row[0].strip(), row[2].strip(), row[3].strip()
                try:
                    red_balls = sorted(list(map(int, red_str.split(','))))
                    blue_ball = int(blue_str)
                    if len(red_balls) != 6:
                         debug_log(f"CSV期号 {period}: 红球数量 {len(red_balls)} 不是6个. 跳过.", 2)
                         continue
                    if not all(1 <= r <= 33 for r in red_balls) or not (1 <= blue_ball <= 16):
                         debug_log(f"CSV期号 {period}: 号码超出范围. 红: {red_balls}, 蓝: {blue_ball}. 跳过.", 2)
                         continue
                    period_map[period] = {'date': row[1].strip(), 'red': red_balls, 'blue': blue_ball}
                    periods_list.append(period)
                except ValueError:
                    debug_log(f"CSV数据格式错误或数字转换失败，跳过第 {i+1} 行: {row}", 2)
                except Exception as e:
                    debug_log(f"处理CSV第 {i+1} 行 ({row}) 时发生未知错误: {str(e)}，跳过该行.", 3)
        
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
    debug_log(f"开始查找数据截止期为 {target_period} 的分析报告...", 1)
    candidates = []
    
    # Assumes REPORT_PATTERN is relative to the Current Working Directory (CWD)
    # If script is not run from the dir containing reports, this might need SCRIPT_DIR logic
    report_files = glob.glob(os.path.join('.', REPORT_PATTERN)) 

    if not report_files:
         debug_log(f"未找到符合模式 '{REPORT_PATTERN}' 的分析报告文件 (CWD: {os.getcwd()})。", 1)
         return None
    
    debug_log(f"发现 {len(report_files)} 个潜在报告文件。", 1)

    for file_name in report_files:
        content = robust_file_read(file_name) # robust_file_read handles abspath
        if not content:
            debug_log(f"跳过读取失败的报告文件: {file_name}", 2)
            continue

        match = re.search(r'数据范围:\s*(\d{7})\s*-\s*(\d{7})', content)
        if match:
            _, extracted_end_period = match.group(1).strip(), match.group(2).strip()
            if extracted_end_period == target_period:
                time_match = re.search(r'_(\d{8}_\d{6})\.', os.path.basename(file_name))
                if time_match:
                    try:
                        timestamp = datetime.strptime(time_match.group(1), "%Y%m%d_%H%M%S")
                        candidates.append((timestamp, os.path.abspath(file_name))) # Store absolute path
                    except ValueError:
                         debug_log(f"文件名 {file_name} 时间戳格式不正确. 跳过.", 2)
                else:
                     debug_log(f"文件名 {file_name} 未包含时间戳. 跳过.", 2)

    if not candidates:
        debug_log(f"未找到数据截止期为 {target_period} 的分析报告", 3)
        return None

    candidates.sort(reverse=True)
    selected_path = candidates[0][1]
    debug_log(f"找到 {len(candidates)} 个匹配报告，选择最新: {selected_path}", 1)
    return selected_path # Return absolute path

def parse_recommendations(content):
    """解析推荐组合"""
    debug_log("解析推荐组合...", 1)
    pattern = re.compile(r'组合\s*\d+\s*:\s*红球\s*\[([\d\s,]+?)\]\s*蓝球\s*(\d+)', re.DOTALL)
    parsed_rec = []
    matches = pattern.findall(content)
    if not matches:
        debug_log("未找到普通推荐组合。", 1)

    for red_str_match, blue_str in matches:
        try:
            red_str = red_str_match.replace(' ', '').strip(',')
            if not red_str:
                 debug_log("发现空红球字符串 (普通推荐). 跳过.", 2)
                 continue
            red_balls = sorted(map(int, red_str.split(',')))
            blue_ball = int(blue_str)
            if len(red_balls) != 6:
                 debug_log(f"普通推荐红球数量 {len(red_balls)}!=6: '{red_str_match}'. 跳过.", 2)
                 continue
            if not all(1 <= r <= 33 for r in red_balls) or not (1 <= blue_ball <= 16):
                 debug_log(f"普通推荐号码超出范围. 红:{red_balls}, 蓝:{blue_ball}. 跳过.", 2)
                 continue
            parsed_rec.append((red_balls, blue_ball))
        except ValueError:
            debug_log(f"解析普通推荐时发现无效数字: 红球 '{red_str_match}', 蓝球 '{blue_str}'. 跳过.", 2)
        except Exception as e:
             debug_log(f"解析普通推荐时发生未知错误: {str(e)}. 跳过.", 3)

    debug_log(f"成功解析到 {len(parsed_rec)} 个有效普通推荐组合。", 1)
    return parsed_rec

def parse_complex(content):
    """解析复式组合"""
    debug_log("解析复式组合...", 1)
    complex_red, complex_blue = [], []

    match_red = re.search(r'推荐\d+红球:\s*\[([\d\s,]+?)\]', content, re.DOTALL)
    if match_red:
        try:
            red_str = match_red.group(1).replace(' ', '').strip(',')
            if red_str: complex_red = sorted(map(int, red_str.split(',')))
            else: debug_log("解析到空的复式红球字符串。", 2)
        except ValueError: debug_log(f"解析复式红球时发现无效数字: '{match_red.group(1)}'", 2)
        except Exception as e: debug_log(f"解析复式红球时发生未知错误: {str(e)}", 3)
    else:
        debug_log("未找到复式红球 (推荐X红球).", 1)
        return [], [] 

    if match_red: 
        search_content_for_blue = content[match_red.end():]
        match_blue = re.search(r'推荐\d+蓝球:\s*\[([\d\s,]+?)\]', search_content_for_blue, re.DOTALL)
        if match_blue:
            try:
                blue_str = match_blue.group(1).replace(' ', '').strip(',')
                if blue_str: complex_blue = sorted(map(int, blue_str.split(',')))
                else: debug_log("解析到空的复式蓝球字符串。", 2)
            except ValueError: debug_log(f"解析复式蓝球时发现无效数字: '{match_blue.group(1)}'", 2)
            except Exception as e: debug_log(f"解析复式蓝球时发生未知错误: {str(e)}", 3)
        else:
            debug_log("未找到复式蓝球 (推荐X蓝球) 在红球之后.", 1)

    if not complex_red and not complex_blue: return [], []
    if len(complex_red) < 6 or len(complex_blue) < 1:
        if complex_red or complex_blue:
            debug_log(f"解析到不完整的复式号码 (红球需>=6，蓝球需>=1). R:{len(complex_red)}, B:{len(complex_blue)}. 跳过.", 2)
        return [], []
    if not all(1 <= r <= 33 for r in complex_red) or not all(1 <= b <= 16 for b in complex_blue):
        debug_log(f"解析到复式号码超出有效范围. R:{complex_red}, B:{complex_blue}. 跳过.", 2)
        return [], []

    debug_log(f"成功解析复式号码: 红球 {complex_red}, 蓝球 {complex_blue}", 1)
    return complex_red, complex_blue

def generate_complex_tickets(reds, blues):
    """从复式号码中生成所有可能的 6红+蓝 投注组合"""
    if len(reds) < 6 or not blues:
        debug_log(f"复式号码不足 (R:{len(reds)}, B:{len(blues)}). 无法生成投注.", 1)
        return []
    tickets = []
    max_r, max_b, max_tickets_limit = 15, 16, 20000
    if len(reds) > max_r or len(blues) > max_b:
        debug_log(f"复式号码数量过多 (R:{len(reds)}, B:{len(blues)}). 跳过生成.", 2)
        return []
    try:
        from math import comb
        num_combs = comb(len(reds), 6) * len(blues)
    except ImportError: # math.comb might not be available in older Python (before 3.8)
        def combinations_count(n, k):
            if k < 0 or k > n: return 0
            if k == 0 or k == n: return 1
            if k > n // 2: k = n - k
            res = 1
            for i in range(k): res = res * (n - i) // (i + 1)
            return res
        try:
            num_combs = combinations_count(len(reds), 6) * len(blues)
        except Exception as e_comb_count: # Catch any error during combinations_count
            debug_log(f"计算复式组合数量时发生异常: {str(e_comb_count)}", 3)
            return []
    except Exception as e_comb: # Catch other errors related to comb (though less likely)
            debug_log(f"计算复式组合数量时发生未知异常: {str(e_comb)}", 3)
            return []


    if num_combs > max_tickets_limit:
        debug_log(f"复式号码可生成 {num_combs:,} 注，超过限制 {max_tickets_limit:,}. 跳过生成.", 2)
        return []
    try:
        for combo in combinations(reds, 6):
            for blue in blues: tickets.append((sorted(list(combo)), blue))
        debug_log(f"从复式号码中成功生成了 {len(tickets):,} 注投注。", 1)
        return tickets
    except Exception as e:
        debug_log(f"生成复式投注时发生异常: {str(e)}", 3)
        return []

def calculate_prize(tickets, prize_red, prize_blue):
    """计算奖金"""
    prize_red_set = set(prize_red)
    breakdown = {k:0 for k in PRIZE_TABLE}
    total_prize = 0
    winning_tickets_list = []

    for red, blue in tickets:
        matched_red = len(set(red) & prize_red_set)
        matched_blue = blue == prize_blue
        level = None
        
        # Determine prize level based on matches
        if matched_red == 6 and matched_blue: level = 1
        elif matched_red == 6 and not matched_blue: level = 2
        elif matched_red == 5 and matched_blue: level = 3
        elif matched_red == 5 and not matched_blue: level = 4
        elif matched_red == 4 and matched_blue: level = 4
        elif matched_red == 4 and not matched_blue: level = 5
        elif matched_red == 3 and matched_blue: level = 5
        elif (matched_red == 2 and matched_blue) or \
             (matched_red == 1 and matched_blue) or \
             (matched_red == 0 and matched_blue): level = 6
        
        if level is not None and level in PRIZE_TABLE:
            breakdown[level] += 1
            winning_tickets_list.append((red, blue, level))

    for level_won, count in breakdown.items():
        if count > 0: # No need to check PRIZE_TABLE again if level was set from it
            total_prize += PRIZE_TABLE[level_won] * count
            
    # debug_log(f"奖金计算明细: {breakdown}, 总金额: {total_prize}", 1) # Can be verbose
    return total_prize, breakdown, winning_tickets_list

def format_winning_tickets(winning_list, winning_red, winning_blue):
    """格式化中奖号码"""
    formatted = []
    winning_red_set = set(winning_red)
    for red, blue, level in winning_list:
        matched_r_balls = sorted(list(set(red) & winning_red_set))
        matched_b_ball = blue if blue == winning_blue else None
        red_str_parts = [f"**{r}**" if r in matched_r_balls else str(r) for r in sorted(red)]
        blue_str = f"**{blue}**" if matched_b_ball is not None else str(blue)
        details = []
        if matched_r_balls: details.append(f"命中红球: {','.join(map(str, matched_r_balls))}")
        if matched_b_ball is not None: details.append(f"命中蓝球: {matched_b_ball}")
        details_str = f" ({'; '.join(details)})" if details else ""
        formatted.append(f"  - 红[{', '.join(red_str_parts)}] 蓝{blue_str} ({level}等奖){details_str}")
    return formatted

def manage_report(new_entry=None, new_error=None):
    """维护主报告文件"""
    normal_marker, error_marker = "==== EVALUATION RECORDS =====", "==== ERROR LOGS ===="
    report_content, normal_entries_from_file, error_logs_from_file = [], [], []
    
    absolute_main_report_path = os.path.abspath(MAIN_REPORT_FILE)

    if os.path.exists(absolute_main_report_path):
        content_read = robust_file_read(absolute_main_report_path) # robust_file_read uses abspath
        if content_read is not None:
            report_content = [line.rstrip() for line in content_read.splitlines()]
    
    normal_start, error_start = -1, -1
    for i, line in enumerate(report_content):
        if line.startswith(normal_marker): normal_start = i
        elif line.startswith(error_marker): error_start = i

    if normal_start != -1:
        end_slice = error_start if error_start != -1 and error_start > normal_start else len(report_content)
        normal_entries_from_file = [line for line in report_content[normal_start + 1 : end_slice] if line.strip() or line == ""]
    if error_start != -1:
        error_logs_from_file = [line for line in report_content[error_start + 1 :] if line.strip()]

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    final_normal_entries = []

    if new_entry:
        eval_p, report_p = new_entry.get('evaluation_period', 'N/A'), new_entry.get('report_cutoff_period', 'N/A')
        win_r, win_b = new_entry.get('winning_red', []), new_entry.get('winning_blue', 'N/A')
        overall_prize = new_entry.get('overall_prize', 0)
        rec_data, complex_data = new_entry.get('recommended', {}), new_entry.get('complex', {})
        win_r_str = ','.join(map(str, sorted(win_r))) if isinstance(win_r, list) and win_r else 'N/A'
        
        entry_block = [f"[评估记录 {timestamp}]", f"评估期号 (实际开奖): {eval_p}", f"评估报告数据截止期: {report_p}",
                       f"实际开奖号码: 红[{win_r_str}] 蓝{win_b}", f"总中奖金额: {overall_prize:,}元", ""]
        
        entry_block.append("--- 普通推荐中奖详情 ---")
        if rec_data and (rec_data.get('prize', 0) > 0 or rec_data.get('winners')):
            entry_block.append(f"普通推荐总奖金: {rec_data.get('prize', 0):,}元")
            bd = ", ".join([f"{lvl}等奖:{cnt}注" for lvl,cnt in sorted(rec_data.get('breakdown',{}).items()) if cnt>0])
            entry_block.append(f"普通推荐奖金明细: {bd if bd else '无中奖'}")
            if rec_data.get('winners'): entry_block.extend(["中奖号码:"] + format_winning_tickets(rec_data['winners'], win_r, win_b))
            else: entry_block.append("无中奖号码")
        else: entry_block.append("无普通推荐投注或未中奖")
        entry_block.append("")

        entry_block.append("--- 复式生成投注中奖详情 ---")
        if complex_data and (complex_data.get('prize', 0) > 0 or complex_data.get('winners')):
            entry_block.append(f"复式生成总奖金: {complex_data.get('prize', 0):,}元")
            bd = ", ".join([f"{lvl}等奖:{cnt}注" for lvl,cnt in sorted(complex_data.get('breakdown',{}).items()) if cnt>0])
            entry_block.append(f"复式生成奖金明细: {bd if bd else '无中奖'}")
            if complex_data.get('winners'): entry_block.extend(["中奖号码:"] + format_winning_tickets(complex_data['winners'], win_r, win_b))
            else: entry_block.append("无中奖号码")
        else: entry_block.append("无复式生成投注或未中奖")
        entry_block.extend(["", "="*60, ""]) # Separator and blank lines
        final_normal_entries.extend(entry_block)

    final_normal_entries.extend(normal_entries_from_file)
    
    # Trim normal entries based on MAX_NORMAL_RECORDS (blocks)
    block_indices = [i for i, line in enumerate(final_normal_entries) if line.startswith("[评估记录 ")]
    if len(block_indices) > MAX_NORMAL_RECORDS:
        # Keep the MAX_NORMAL_RECORDS newest blocks. Since new entries are prepended,
        # these are the first MAX_NORMAL_RECORDS blocks in block_indices.
        # We need to find the end of the MAX_NORMAL_RECORDS'th block.
        # The (MAX_NORMAL_RECORDS+1)'th block starts at block_indices[MAX_NORMAL_RECORDS].
        # So, slice up to that point.
        start_of_block_to_cut = block_indices[MAX_NORMAL_RECORDS]
        final_normal_entries = final_normal_entries[:start_of_block_to_cut]

    final_error_logs = error_logs_from_file # Start with existing errors
    if new_error:
        final_error_logs.insert(0, f"[错误 {timestamp}] {new_error}") # Prepend new error
    final_error_logs = final_error_logs[:MAX_ERROR_LOGS] # Trim to max error logs

    try:
        with open(absolute_main_report_path, 'w', encoding='utf-8') as f:
            f.write(f"{normal_marker}\n")
            if final_normal_entries: 
                # Remove trailing blank lines from the block, then add one final newline
                f.write("\n".join(final_normal_entries).strip('\n') + "\n") 
            
            f.write(f"\n{error_marker}\n") # Ensure newline before error marker
            if final_error_logs: 
                f.write("\n".join(final_error_logs).strip('\n') + "\n")
        debug_log(f"主报告已更新: {absolute_main_report_path}", 1)
    except Exception as e:
        debug_log(f"写入主报告文件 '{absolute_main_report_path}' 失败: {str(e)}", 3)

def main_process():
    """主处理流程"""
    debug_log("====== 主流程启动 ======", 1)
    
    csv_abs_path = os.path.abspath(CSV_FILE) # For error messages
    csv_content = robust_file_read(CSV_FILE) 
    if not csv_content:
        err_msg = f"CSV文件 '{CSV_FILE}' (路径: {csv_abs_path}) 读取失败或文件不存在/内容为空."
        manage_report(new_error=err_msg)
        debug_log(f"处理停止: {err_msg}", 3)
        return

    period_map, sorted_periods = get_period_data(csv_content)
    if not period_map or not sorted_periods:
        err_msg = f"CSV文件 '{CSV_FILE}' 数据解析失败或无有效数据."
        manage_report(new_error=err_msg)
        debug_log(f"处理停止: {err_msg}", 3)
        return

    if len(sorted_periods) < 2:
        err_msg = f"CSV数据不足两期 ({len(sorted_periods)}期). 需要至少两期以进行评估."
        manage_report(new_error=err_msg)
        debug_log(f"处理停止: {err_msg}", 2)
        return

    evaluation_period = sorted_periods[-1]
    report_data_cutoff_period = sorted_periods[-2]
    debug_log(f"评估期号 (最新CSV期): {evaluation_period}, 报告数据截止期 (CSV倒数第二期): {report_data_cutoff_period}", 1)

    analysis_file_abs_path = find_matching_report(report_data_cutoff_period) # Returns absolute path
    if not analysis_file_abs_path:
        err_msg = f"未找到数据截止期为 {report_data_cutoff_period} 的分析报告."
        manage_report(new_error=err_msg)
        debug_log(f"处理停止: {err_msg}", 2)
        return

    analysis_content = robust_file_read(analysis_file_abs_path)
    if not analysis_content:
        err_msg = f"分析报告 '{analysis_file_abs_path}' 读取失败或内容为空."
        manage_report(new_error=err_msg)
        debug_log(f"处理停止: {err_msg}", 3)
        return

    rec_tickets = parse_recommendations(analysis_content)
    complex_red, complex_blue = parse_complex(analysis_content)
    complex_tickets = generate_complex_tickets(complex_red, complex_blue)

    if not rec_tickets and not complex_tickets:
         debug_log("未能从分析报告中解析出任何投注组合，将计算奖金为0。", 2)

    if evaluation_period not in period_map:
         err_msg = f"内部错误: 评估期号 {evaluation_period} 在 sorted_periods 中，但不在 period_map 中."
         manage_report(new_error=err_msg)
         debug_log(f"处理停止: {err_msg}", 3)
         return

    prize_data = period_map[evaluation_period]
    winning_r, winning_b = prize_data['red'], prize_data['blue']
    debug_log(f"获取到评估期号 {evaluation_period} ({prize_data['date']}) 的实际开奖数据: 红{winning_r} 蓝{winning_b}", 1)

    rec_prize, rec_bd, rec_wl = calculate_prize(rec_tickets, winning_r, winning_b)
    complex_prize, complex_bd, complex_wl = calculate_prize(complex_tickets, winning_r, winning_b)
    overall_prize = rec_prize + complex_prize

    report_entry = {
        'evaluation_period': evaluation_period, 'report_cutoff_period': report_data_cutoff_period,
        'winning_red': winning_r, 'winning_blue': winning_b, 'overall_prize': overall_prize,
        'recommended': {'prize': rec_prize, 'breakdown': rec_bd, 'winners': rec_wl},
        'complex': {'prize': complex_prize, 'breakdown': complex_bd, 'winners': complex_wl}
    }
    manage_report(new_entry=report_entry)

    if not rec_tickets and not complex_tickets: # Log this specific failure if it occurs
         manage_report(new_error=f"未能从分析报告 '{os.path.basename(analysis_file_abs_path)}' (数据截止 {report_data_cutoff_period}) 中解析出任何投注组合.")
    
    debug_log(f"处理完成！评估报告 (数据截止 {report_data_cutoff_period}) 在期号 {evaluation_period} 开奖结果上的表现，总中奖金额: {overall_prize:,}元。", 1)
    debug_log("====== 主流程结束 ======", 1)

if __name__ == "__main__":
    try:
        main_process()
    except Exception as e:
        # Capture full traceback for better debugging
        tb_str = traceback.format_exc()
        error_message = f"主流程发生未处理异常: {type(e).__name__} - {str(e)}\n详细追溯:\n{tb_str}"
        try:
            manage_report(new_error=error_message)
        except Exception as report_e:
            # If manage_report itself fails, print critical error to console
            print(f"[CRITICAL ERROR] {datetime.now().strftime('%H:%M:%S')} Failed to write final error to report file: {str(report_e)}")
            print(f"[CRITICAL ERROR] {datetime.now().strftime('%H:%M:%S')} Original unhandled error: {error_message}")
        debug_log(error_message, 3) # Ensure it's printed to console as well

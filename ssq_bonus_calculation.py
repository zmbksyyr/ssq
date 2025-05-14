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

class PeriodIntegrityError(Exception):
    """期号连续性异常"""
    def __init__(self, missing_period):
        self.missing_period = missing_period
        super().__init__(f"期号链断裂于 {missing_period}")

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
    try:
        reader = csv.reader(csv_content.splitlines())
        for row in reader:
            if len(row) >=4 and re.match(r'\d{7}', row[0]):
                period_map[row[0]] = {
                    'date': row[1],
                    'red': sorted(list(map(int, row[2].split(',')))),
                    'blue': int(row[3])
                }
        return period_map
    except Exception as e:
        debug_log(f"CSV数据解析失败: {str(e)}", 3)
        return None

def build_period_chain(sorted_periods):
    """构建期号链并检查连续性"""
    period_chain = {}
    for i in range(len(sorted_periods)-1):
        current = sorted_periods[i]
        next_p = sorted_periods[i+1]
        
        if int(next_p) - int(current) != 1:
            raise PeriodIntegrityError(current)
        
        period_chain[current] = next_p
    return period_chain

def find_matching_report(target_period):
    """查找匹配指定期号的分析报告"""
    debug_log(f"开始查找处理期号 {target_period} 的分析报告...")
    candidates = []
    
    for file in glob.glob(REPORT_PATTERN):
        content = robust_file_read(file)
        if not content:
            continue
        
        match = re.search(r'数据期数范围:.*?第\s*(\d+)\s*期\s*至\s*第\s*(\d+)\s*期', content)
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
    pattern = re.compile(
        r'组合\s*\d+\s*:\s*红球\s*$$([\d\s,]+)$$\s*蓝球\s*(\d+)',
        re.DOTALL
    )
    return [
        (sorted(map(int, red.split(','))), int(blue)) 
        for red, blue in pattern.findall(content)[:5]
        if len(red.split(',')) == 6
    ]

def parse_complex(content):
    """解析复式组合"""
    debug_log("解析复式组合...")
    match = re.search(
        r'7\+7复式选号.*?红球:\s*$$([\d\s,]+)$$.*?蓝球:\s*$$([\d\s,]+)$$',
        content, re.DOTALL
    )
    if not match:
        return [], []
    
    return (
        sorted(map(int, match.group(1).split(','))),
        sorted(map(int, match.group(2).split(',')))
    )

def generate_complex_tickets(reds, blues):
    """生成复式投注"""
    if len(reds) <6 or not blues:
        return []
    return [
        (sorted(combo), blue)
        for combo in combinations(reds, 6)
        for blue in blues
    ]

def calculate_prize(tickets, prize_red, prize_blue):
    """计算奖金"""
    prize_red_set = set(prize_red)
    breakdown = {k:0 for k in PRIZE_TABLE}
    
    for red, blue in tickets:
        matched_red = len(set(red) & prize_red_set)
        matched_blue = blue == prize_blue
        
        if matched_red ==6:
            level =1 if matched_blue else2
        elif matched_red ==5:
            level =3 if matched_blue else4
        elif matched_red ==4:
            level =4 if matched_blue else5
        elif matched_red ==3 and matched_blue:
            level =5
        elif matched_blue:
            level =6
        else:
            continue
        
        breakdown[level] +=1
    
    total = sum(PRIZE_TABLE[k]*v for k,v in breakdown.items())
    return total, breakdown

def manage_report(new_entry=None, new_error=None):
    """维护主报告文件"""
    normal_marker = "==== NORMAL RECORDS ===="
    error_marker = "==== ERROR LOGS ===="
    
    # 读取现有内容
    normal_entries = []
    error_logs = []
    if os.path.exists(MAIN_REPORT_FILE):
        with open(MAIN_REPORT_FILE, 'r', encoding='utf-8') as f:
            current_section = None
            for line in f:
                if line.startswith(normal_marker):
                    current_section = 'normal'
                    continue
                if line.startswith(error_marker):
                    current_section = 'error'
                    continue
                
                if current_section == 'normal' and line.strip():
                    normal_entries.append(line.strip())
                elif current_section == 'error' and line.strip():
                    error_logs.append(line.strip())

    # 处理新增内容
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if new_entry:
        entry = [
            f"[记录 {timestamp}]",
            f"分析期号: {new_entry['period']}",
            f"开奖号码: 红{new_entry['red']} 蓝{new_entry['blue']}",
            f"中奖金额: {new_entry['prize']:,}元",
            "-"*40
        ]
        normal_entries = entry + normal_entries[: (MAX_NORMAL_RECORDS-1)*5]
    
    if new_error:
        error_logs = [f"[错误 {timestamp}] {new_error}"] + error_logs[:MAX_ERROR_LOGS-1]

    # 写入文件
    with open(MAIN_REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"{normal_marker}\n")
        f.write("\n".join(normal_entries[:MAX_NORMAL_RECORDS*5]))
        
        f.write(f"\n\n{error_marker}\n")
        f.write("\n".join(error_logs[:MAX_ERROR_LOGS]))

def main_process():
    """主处理流程"""
    debug_log("====== 主流程启动 ======", 1)
    
    # 读取CSV数据
    csv_content = robust_file_read(CSV_FILE)
    if not csv_content:
        manage_report(new_error="CSV文件读取失败")
        return
    
    # 构建期号数据
    period_map = get_period_data(csv_content)
    if not period_map:
        manage_report(new_error="CSV数据解析失败")
        return
    
    sorted_periods = sorted(period_map.keys())
    if len(sorted_periods) <2:
        manage_report(new_error="CSV数据不足两期")
        return
    
    # 构建期号链
    try:
        period_chain = build_period_chain(sorted_periods)
    except PeriodIntegrityError as e:
        manage_report(new_error=f"期号不连续: {e.missing_period}")
        return
    
    # 获取最新期号
    latest_period = sorted_periods[-1]
    debug_log(f"CSV最新期号: {latest_period}")
    
    # 查找匹配的分析报告
    analysis_file = find_matching_report(latest_period)
    if not analysis_file:
        manage_report(new_error=f"未找到处理期号 {latest_period} 的分析报告")
        return
    
    # 解析分析报告
    content = robust_file_read(analysis_file)
    if not content:
        manage_report(new_error="分析报告读取失败")
        return
    
    rec_tickets = parse_recommendations(content)
    complex_red, complex_blue = parse_complex(content)
    complex_tickets = generate_complex_tickets(complex_red, complex_blue)
    all_tickets = rec_tickets + complex_tickets
    
    # 获取下一期开奖数据
    next_period = period_chain.get(latest_period)
    if not next_period or next_period not in period_map:
        manage_report(new_error=f"找不到下一期数据: {latest_period}→{next_period}")
        return
    
    prize_data = period_map[next_period]
    
    # 计算奖金
    total_prize, breakdown = calculate_prize(all_tickets, prize_data['red'], prize_data['blue'])
    
    # 保存结果
    manage_report(new_entry={
        'period': next_period,
        'red': prize_data['red'],
        'blue': prize_data['blue'],
        'prize': total_prize
    })
    
    debug_log(f"处理完成！中奖金额: {total_prize:,}元", 1)

if __name__ == "__main__":
    try:
        main_process()
    except Exception as e:
        manage_report(new_error=f"未处理异常: {str(e)}")
        debug_log(f"主流程异常: {str(e)}", 3)

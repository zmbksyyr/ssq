import os
import re
import glob
import csv
from itertools import combinations
from datetime import datetime

# 配置参数
REPORT_PATTERN = "ssq_analysis_output_*.txt"
CSV_FILE = "shuangseqiu.csv"
OUTPUT_FILE = "latest_ssq_analysis.txt" # Ensure this path is correct for your environment

# 奖金对照表
PRIZE_TABLE = {
    1: 5_000_000,
    2: 500_000,
    3: 3_000,
    4: 200,
    5: 10,
    6: 5
}

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

def find_analysis_file():
    """智能查找分析报告文件"""
    debug_log("启动文件搜索...")
    try:
        candidates = []
        for file in glob.glob(REPORT_PATTERN):
            match = re.search(r'_(\d{8}_\d{6})\.', file)
            if match:
                timestamp = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
                candidates.append((timestamp, file))
        
        if not candidates:
            debug_log("未找到任何分析报告文件", 3)
            return None
        
        # 按时间降序排列 (newest first)
        candidates.sort(reverse=True)
        debug_log(f"找到 {len(candidates)} 个候选文件，最新文件：{candidates[0][0].strftime('%Y-%m-%d %H:%M')}")
        
        # 修改点 1: 恢复为选择第二新的文件（如果存在至少两个文件），否则选择最新的/唯一的。
        target_index = 1 if len(candidates) >= 2 else 0
        
        selected = candidates[target_index][1]
        debug_log(f"选择文件：{selected} (规则: {'第二新' if target_index == 1 else '最新/唯一'})")
        return selected
    
    except Exception as e:
        debug_log(f"文件搜索失败：{str(e)}", 3)
        return None

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

def parse_recommendations(content):
    """改进的推荐组合解析"""
    debug_log("解析推荐组合...")
    pattern = re.compile(
        r'组合\s*\d+\s*:\s*红球\s*\[([\d\s,]+)\]\s*蓝球\s*(\d+)',
        re.DOTALL
    )
    matches = pattern.findall(content)
    debug_log(f"原始匹配结果（推荐组合）：{matches[:5]}")

    recommendations = []
    for i, (red_str, blue) in enumerate(matches[:5], 1): # Process up to 5 recommendations
        try:
            red = sorted(list(map(int, red_str.replace(' ', '').split(','))))
            if len(red) !=6:
                debug_log(f"无效红球数量：{red}，跳过推荐组合 {i}", 2)
                continue
            recommendations.append((red, int(blue)))
            debug_log(f"成功解析推荐组合 {i}: 红球{red} 蓝球{blue}")
        except Exception as e:
            debug_log(f"推荐组合 {i} 解析失败：{str(e)}", 2)
    
    return recommendations

def parse_complex(content):
    """增强型复式组合解析"""
    debug_log("解析复式组合...")
    section_pattern = re.compile(
        r'7\+7复式选号.*?选择的7个红球:\s*\[([\d\s,]+)\].*?选择的7个蓝球:\s*\[([\d\s,]+)\]',
        re.DOTALL | re.IGNORECASE
    )
    match = section_pattern.search(content)
    if not match:
        debug_log("未找到复式组合区块", 2)
        return [], []
    
    try:
        reds_str = match.group(1).replace(' ', '')
        blues_str = match.group(2).replace(' ', '')
        
        reds = sorted(list(map(int, reds_str.split(','))))
        blues = sorted(list(map(int, blues_str.split(','))))
        
        debug_log(f"解析到复式红球({len(reds)}个)：{reds}")
        debug_log(f"解析到复式蓝球({len(blues)}个)：{blues}")
        return reds, blues
    except Exception as e:
        debug_log(f"复式组合解析异常：{str(e)}", 3)
        return [], []

def parse_analysis_file(file_path):
    """综合解析函数"""
    debug_log(f"开始解析文件：{file_path}")
    content = robust_file_read(file_path)
    if not content:
        return None, ([], []) 
    
    sample_content = content[:500] + ("..." if len(content)>500 else "")
    debug_log(f"文件内容预览：\n{sample_content}")
    
    recommendations = parse_recommendations(content)
    complex_red, complex_blue = parse_complex(content)
    return recommendations, (complex_red, complex_blue)

def get_latest_result():
    """增强的CSV解析"""
    debug_log(f"读取CSV文件：{CSV_FILE}")
    try:
        content = robust_file_read(CSV_FILE)
        if not content:
            return None, None
        
        reader = csv.reader(content.splitlines())
        all_data_rows = [row for row in reader if len(row) >= 4 and re.match(r'\d{7}', row[0])]

        if not all_data_rows:
            debug_log("CSV文件中没有有效数据行", 3)
            return None, None
        
        last_row = all_data_rows[-1]
        debug_log(f"最新记录原始数据：{last_row}")
        
        try:
            prize_red = sorted(list(map(int, last_row[2].split(','))))
            prize_blue = int(last_row[3])
            return prize_red, prize_blue
        except Exception as e:
            debug_log(f"CSV数据转换失败：{str(e)}", 3)
            return None, None
    
    except Exception as e:
        debug_log(f"CSV解析异常：{str(e)}", 3)
        return None, None

def generate_complex_tickets(reds, blues):
    """安全组合生成"""
    debug_log("生成复式投注...")
    if len(reds) <6:
        debug_log(f"红球不足6个（当前{len(reds)}），无法生成复式组合", 2)
        return []
    if not blues:
        debug_log("蓝球列表为空，无法生成复式组合", 2)
        return []
    
    try:
        red_combos = list(combinations(reds, 6))
        debug_log(f"从{len(reds)}个红球中生成 {len(red_combos)} 种6红球组合")
        tickets = [ (sorted(list(r)), b) for r in red_combos for b in blues ]
        debug_log(f"总生成复式投注 {len(tickets)} 注")
        return tickets
    except Exception as e:
        debug_log(f"复式组合生成异常：{str(e)}", 3)
        return []

def calculate_prize(tickets, prize_red, prize_blue):
    """奖金计算"""
    debug_log("开始奖金计算...")
    breakdown = {k:0 for k in PRIZE_TABLE}
    total_winnings = 0
    prize_red_set = set(prize_red)
    
    if not tickets:
        debug_log("没有投注组合进行奖金计算。", 2)
        return 0, breakdown

    for i, (red, blue) in enumerate(tickets, 1):
        matched_red_count = len(set(red) & prize_red_set)
        matched_blue = (blue == prize_blue)
        
        level = 0
        if matched_red_count ==6:
            level =1 if matched_blue else 2
        elif matched_red_count ==5:
            level =3 if matched_blue else 4
        elif matched_red_count ==4:
            level =4 if matched_blue else 5
        elif matched_red_count ==3 and matched_blue:
            level =5
        elif matched_blue:
            if matched_red_count < 3:
                 level = 6
        
        if level in PRIZE_TABLE:
            total_winnings += PRIZE_TABLE[level]
            breakdown[level] +=1
            # debug_log(f"中奖！组合: 红{red} 蓝{blue} -> 匹配红球{matched_red_count}, 匹配蓝球:{matched_blue}, 奖级:{level}, 金额:{PRIZE_TABLE[level]}", 1)
        # elif i <= 10 or i == len(tickets):
             # debug_log(f"未中奖。组合: 红{red} 蓝{blue} -> 匹配红球{matched_red_count}, 匹配蓝球:{matched_blue}", 1)

    debug_log(f"奖金计算完成. 总奖金: {total_winnings}, 明细: {breakdown}")
    return total_winnings, breakdown

def save_report(result):
    """带格式的报表生成并追加到文件"""
    debug_log(f"准备生成最终报告并追加到: {OUTPUT_FILE}...")
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_content_list = [
            f"\n{'='*60}", # Start with a newline for separation from previous reports
            f"双色球收益报告 {timestamp}",
            f"分析文件: {result.get('analysis_file', 'N/A')}",
            f"开奖号码：红球{result['prize_red']} 蓝球{result['prize_blue']}",
            f"总投注数: {result['total_bets_count']} 注 (推荐{result['rec_count']}注 + 复式{result['complex_count']}注)",
            '-'*60
        ]
        
        if result['total_winnings'] >0:
            report_content_list.append("中奖详情:")
            for level in sorted(PRIZE_TABLE.keys()):
                count = result['breakdown'].get(level, 0)
                if count > 0:
                    report_content_list.append(f"  【{level}等奖】{count}注 × {PRIZE_TABLE[level]:,}元 = {count * PRIZE_TABLE[level]:,}元")
        else:
            report_content_list.append("未中奖")
        
        report_content_list.extend([
            f"\n总奖金：{result['total_winnings']:,}元",
            '='*60 + '\n' # End with a newline
        ])
        
        # Appending to the file. 'a' mode creates the file if it doesn't exist.
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write('\n'.join(report_content_list))
        
        debug_log(f"报告已成功追加保存至 {OUTPUT_FILE}")
    except Exception as e:
        debug_log(f"报告保存失败：{str(e)}", 3)

def main():
    debug_log("====== 程序启动 ======", 1)
    
    analysis_file_path = find_analysis_file()
    if not analysis_file_path:
        debug_log("未能找到分析文件，程序终止。", 3)
        return
    
    # Ensure rec_tickets is initialized even if parse_analysis_file returns None for it
    rec_tickets = [] 
    complex_red_list, complex_blue_list = [], []

    parsed_data = parse_analysis_file(analysis_file_path)
    if parsed_data and parsed_data[0] is not None : # Check if parsing didn't completely fail
        rec_tickets, (complex_red_list, complex_blue_list) = parsed_data
    else:
        debug_log("解析分析文件返回空数据或失败，将使用空投注列表。", 2)
        # rec_tickets, complex_red_list, complex_blue_list remain []
        
    complex_tickets = generate_complex_tickets(complex_red_list, complex_blue_list)
    
    prize_red_numbers, prize_blue_number = get_latest_result()
    if not prize_red_numbers:
        debug_log("未能获取最新开奖结果，程序终止。", 3)
        return
    
    # Ensure rec_tickets is a list before concatenation
    all_tickets = (rec_tickets if rec_tickets is not None else []) + \
                  (complex_tickets if complex_tickets is not None else [])
    total_bets_count = len(all_tickets)
    debug_log(f"总投注数：{total_bets_count}")
    
    if total_bets_count > 0:
        total_winnings_amount, prize_breakdown = calculate_prize(all_tickets, prize_red_numbers, prize_blue_number)
    else:
        total_winnings_amount, prize_breakdown = 0, {k:0 for k in PRIZE_TABLE}
        debug_log("没有投注，无需计算奖金。", 1)

    save_report({
        'analysis_file': analysis_file_path,
        'prize_red': prize_red_numbers,
        'prize_blue': prize_blue_number,
        'rec_count': len(rec_tickets if rec_tickets is not None else []),
        'complex_count': len(complex_tickets if complex_tickets is not None else []),
        'total_bets_count': total_bets_count,
        'breakdown': prize_breakdown,
        'total_winnings': total_winnings_amount
    })
    
    print("\n最终结果：")
    print(f"使用的分析文件: {analysis_file_path}")
    print(f"红球开奖：{prize_red_numbers}")
    print(f"蓝球开奖：{prize_blue_number}")
    print(f"总投注数：{total_bets_count}注")
    print(f"中奖金额：{total_winnings_amount:,}元")

if __name__ == "__main__":
    main()

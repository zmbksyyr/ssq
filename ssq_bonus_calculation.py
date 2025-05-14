import os
import re
import glob
import csv
from itertools import combinations
from datetime import datetime

# 配置参数
REPORT_PATTERN = "ssq_analysis_output_*.txt"
CSV_FILE = "shuangseqiu.csv"
OUTPUT_FILE = "latest_ssq_analysis.txt"

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
        
        # 按时间降序排列
        candidates.sort(reverse=True)
        debug_log(f"找到 {len(candidates)} 个候选文件，最新文件：{candidates[0][0].strftime('%Y-%m-%d %H:%M')}")
        
        # 修改点 1: 选择最新的文件
        target_index = 0 # Was: 1 if len(candidates) >=2 else 0
        selected = candidates[target_index][1]
        debug_log(f"选择文件：{selected}")
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
            debug_log(f"成功使用 {encoding} 编码读取文件")
            return content
        except UnicodeDecodeError:
            continue
        except Exception as e:
            debug_log(f"读取异常：{str(e)}", 3)
            return None
    debug_log("无法识别文件编码", 3)
    return None

def parse_recommendations(content):
    """改进的推荐组合解析"""
    debug_log("解析推荐组合...")
    # 修改点 2: 更新正则表达式以匹配方括号 []
    pattern = re.compile(
        r'组合\s*\d+\s*:\s*红球\s*\[([\d\s,]+)\]\s*蓝球\s*(\d+)', # Was: $$([\d\s,]+)$$
        re.DOTALL
    )
    matches = pattern.findall(content)
    debug_log(f"原始匹配结果（推荐组合）：{matches[:5]}") # 打印前五个匹配项

    recommendations = []
    for i, (red_str, blue) in enumerate(matches[:5], 1): # Process up to 5 recommendations
        try:
            red = sorted(list(map(int, red_str.replace(' ', '').split(',')))) # Sort red balls
            if len(red) !=6:
                debug_log(f"无效红球数量：{red}，跳过组合 {i}", 2)
                continue
            recommendations.append((red, int(blue)))
            debug_log(f"成功解析推荐组合 {i}: 红球{red} 蓝球{blue}")
        except Exception as e:
            debug_log(f"推荐组合 {i} 解析失败：{str(e)}", 2)
    
    return recommendations

def parse_complex(content):
    """增强型复式组合解析"""
    debug_log("解析复式组合...")
    # 修改点 3: 更新正则表达式以匹配 "选择的7个红球: [numbers]" 和 "选择的7个蓝球: [numbers]"
    section_pattern = re.compile(
        r'7\+7复式选号.*?选择的7个红球:\s*\[([\d\s,]+)\].*?选择的7个蓝球:\s*\[([\d\s,]+)\]',
        # Was: r'7\+7复式选号.*?红球\s*:\s*$$([\d\s,]+)$$.*?蓝球\s*:\s*$$([\d\s,]+)$$'
        re.DOTALL | re.IGNORECASE # Added re.IGNORECASE for flexibility
    )
    match = section_pattern.search(content)
    if not match:
        debug_log("未找到复式组合区块", 2)
        return [], []
    
    try:
        reds_str = match.group(1).replace(' ', '')
        blues_str = match.group(2).replace(' ', '')
        
        reds = sorted(list(map(int, reds_str.split(',')))) # Sort red balls
        blues = sorted(list(map(int, blues_str.split(',')))) # Sort blue balls
        
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
        return None, ([], []) # Return empty lists if content is None
    
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
        valid_rows = []
        # Read all rows first to find the actual last valid row
        all_data_rows = [row for row in reader if len(row) >= 4 and re.match(r'\d{7}', row[0])] # Assuming issue number is 7 digits like 2025053

        if not all_data_rows:
            debug_log("CSV文件中没有有效数据行", 3)
            return None, None
        
        last_row = all_data_rows[-1] # Get the very last valid row
        debug_log(f"最新记录原始数据：{last_row}")
        
        try:
            prize_red = sorted(list(map(int, last_row[2].split(',')))) # Sort red balls
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
    if not blues: # Check if blues list is empty
        debug_log("蓝球列表为空，无法生成复式组合", 2)
        return []
    
    try:
        red_combos = list(combinations(reds, 6))
        debug_log(f"从{len(reds)}个红球中生成 {len(red_combos)} 种6红球组合")
        tickets = [ (sorted(list(r)), b) for r in red_combos for b in blues ] # Sort red balls in tickets
        debug_log(f"总生成复式投注 {len(tickets)} 注")
        return tickets
    except Exception as e:
        debug_log(f"复式组合生成异常：{str(e)}", 3)
        return []

def calculate_prize(tickets, prize_red, prize_blue):
    """带进度显示的奖金计算"""
    debug_log("开始奖金计算...")
    breakdown = {k:0 for k in PRIZE_TABLE}
    total_winnings = 0 # Renamed from total to avoid confusion
    prize_red_set = set(prize_red) # prize_red should already be sorted
    
    if not tickets:
        debug_log("没有投注组合进行奖金计算。", 2)
        return 0, breakdown

    for i, (red, blue) in enumerate(tickets, 1):
        if i % 1000 ==0 and len(tickets) > 1000 : # Only log progress for many tickets
            debug_log(f"正在计算第 {i}/{len(tickets)} 注...")
        
        # Ensure ticket red balls are sorted for consistent comparison if needed elsewhere,
        # though set operations don't require it.
        # red_set = set(red) # red is already sorted from generation/parsing
        
        matched_red_count = len(set(red) & prize_red_set)
        matched_blue = (blue == prize_blue)
        
        level = 0
        if matched_red_count ==6:
            level =1 if matched_blue else 2
        elif matched_red_count ==5:
            level =3 if matched_blue else 4
        elif matched_red_count ==4:
            level =4 if matched_blue else 5 # Corrected: 4+0 is 5th prize (10 yuan)
        elif matched_red_count ==3 and matched_blue:
            level =5
        elif matched_blue: # This covers 0+1, 1+1, 2+1
            if matched_red_count < 3: # Only award 6th prize if red balls < 3 and blue matches
                 level = 6
        
        if level in PRIZE_TABLE:
            total_winnings += PRIZE_TABLE[level]
            breakdown[level] +=1
            debug_log(f"中奖！组合: 红{red} 蓝{blue} -> 匹配红球{matched_red_count}, 匹配蓝球:{matched_blue}, 奖级:{level}, 金额:{PRIZE_TABLE[level]}", 1)
        elif i <= 10 or i == len(tickets): # Log details for first few and last ticket if no win
             debug_log(f"未中奖。组合: 红{red} 蓝{blue} -> 匹配红球{matched_red_count}, 匹配蓝球:{matched_blue}", 1)


    debug_log("奖金计算完成")
    return total_winnings, breakdown

def save_report(result):
    """带格式的报表生成"""
    debug_log("生成最终报告...")
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_content = [ # Renamed to avoid conflict with 'report' module if ever imported
            f"\n{'='*60}",
            f"双色球收益报告 {timestamp}",
            f"分析文件: {result.get('analysis_file', 'N/A')}", # Added analysis file to report
            f"开奖号码：红球{result['prize_red']} 蓝球{result['prize_blue']}",
            f"总投注数: {result['total_bets_count']} 注 (推荐{result['rec_count']}注 + 复式{result['complex_count']}注)",
            '-'*60
        ]
        
        if result['total_winnings'] >0:
            report_content.append("中奖详情:")
            for level in sorted(PRIZE_TABLE.keys()): # Iterate through sorted prize levels
                count = result['breakdown'].get(level, 0)
                if count > 0:
                    report_content.append(f"  【{level}等奖】{count}注 × {PRIZE_TABLE[level]:,}元 = {count * PRIZE_TABLE[level]:,}元")
        else:
            report_content.append("未中奖")
        
        report_content.extend([
            f"\n总奖金：{result['total_winnings']:,}元",
            '='*60 + '\n'
        ])
        
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        debug_log(f"报告已保存至 {OUTPUT_FILE}")
    except Exception as e:
        debug_log(f"报告保存失败：{str(e)}", 3)

def main():
    debug_log("====== 程序启动 ======", 1)
    
    analysis_file_path = find_analysis_file() # Renamed for clarity
    if not analysis_file_path:
        debug_log("未能找到分析文件，程序终止。", 3)
        return
    
    rec_tickets, (complex_red_list, complex_blue_list) = parse_analysis_file(analysis_file_path) # Renamed for clarity
    if rec_tickets is None: # Check if parsing failed
        debug_log("解析分析文件失败，程序终止。", 3)
        return
        
    complex_tickets = generate_complex_tickets(complex_red_list, complex_blue_list)
    
    prize_red_numbers, prize_blue_number = get_latest_result() # Renamed for clarity
    if not prize_red_numbers:
        debug_log("未能获取最新开奖结果，程序终止。", 3)
        return
    
    all_tickets = (rec_tickets if rec_tickets else []) + (complex_tickets if complex_tickets else [])
    total_bets_count = len(all_tickets)
    debug_log(f"总投注数：{total_bets_count}")
    
    # Ensure all_tickets is not None before calculating prize
    if total_bets_count > 0:
        total_winnings_amount, prize_breakdown = calculate_prize(all_tickets, prize_red_numbers, prize_blue_number) # Renamed for clarity
    else:
        total_winnings_amount, prize_breakdown = 0, {k:0 for k in PRIZE_TABLE}
        debug_log("没有投注，无需计算奖金。", 1)

    save_report({
        'analysis_file': analysis_file_path, # Added for report
        'prize_red': prize_red_numbers,
        'prize_blue': prize_blue_number,
        'rec_count': len(rec_tickets if rec_tickets else []),
        'complex_count': len(complex_tickets if complex_tickets else []),
        'total_bets_count': total_bets_count,
        'breakdown': prize_breakdown,
        'total_winnings': total_winnings_amount
    })
    
    print("\n最终结果：")
    print(f"分析文件: {analysis_file_path}")
    print(f"红球开奖：{prize_red_numbers}")
    print(f"蓝球开奖：{prize_blue_number}")
    print(f"总投注数：{total_bets_count}注")
    print(f"中奖金额：{total_winnings_amount:,}元")

if __name__ == "__main__":
    main()

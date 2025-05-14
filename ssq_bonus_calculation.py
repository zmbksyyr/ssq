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
        
        target_index = 1 if len(candidates) >=2 else 0
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
    pattern = re.compile(
        r'组合\s*\d+\s*:\s*红球\s*$$([\d\s,]+)$$\s*蓝球\s*(\d+)',
        re.DOTALL
    )
    matches = pattern.findall(content)
    debug_log(f"原始匹配结果：{matches[:2]}")  # 打印前两个匹配项
    
    recommendations = []
    for i, (red_str, blue) in enumerate(matches[:5], 1):
        try:
            red = list(map(int, red_str.replace(' ', '').split(',')))
            if len(red) !=6:
                debug_log(f"无效红球数量：{red}，跳过组合 {i}", 2)
                continue
            recommendations.append((red, int(blue)))
            debug_log(f"成功解析组合 {i}: 红球{red} 蓝球{blue}")
        except Exception as e:
            debug_log(f"组合 {i} 解析失败：{str(e)}", 2)
    
    return recommendations

def parse_complex(content):
    """增强型复式组合解析"""
    debug_log("解析复式组合...")
    section_pattern = re.compile(
        r'7\+7复式选号.*?红球\s*:\s*$$([\d\s,]+)$$.*?蓝球\s*:\s*$$([\d\s,]+)$$',
        re.DOTALL
    )
    match = section_pattern.search(content)
    if not match:
        debug_log("未找到复式组合区块", 2)
        return [], []
    
    try:
        reds = list(map(int, match.group(1).replace(' ', '').split(',')))
        blues = list(map(int, match.group(2).replace(' ', '').split(',')))
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
        return None, (None, None)
    
    # 显示关键区域内容
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
        for row in reader:
            if len(row) >=4 and re.match(r'\d{6}', row[0]):
                valid_rows.append(row)
        
        if not valid_rows:
            debug_log("没有有效数据行", 3)
            return None, None
        
        last_row = valid_rows[-1]
        debug_log(f"最新记录原始数据：{last_row}")
        
        try:
            prize_red = list(map(int, last_row[2].split(',')))
            prize_blue = int(last_row[3])
            return prize_red, prize_blue
        except Exception as e:
            debug_log(f"数据转换失败：{str(e)}", 3)
            return None, None
    
    except Exception as e:
        debug_log(f"CSV解析异常：{str(e)}", 3)
        return None, None

def generate_complex_tickets(reds, blues):
    """安全组合生成"""
    debug_log("生成复式投注...")
    if len(reds) <6:
        debug_log(f"红球不足6个（当前{len(reds)}），无法生成组合", 2)
        return []
    if len(blues) <1:
        debug_log("蓝球数量为0，无法生成组合", 2)
        return []
    
    try:
        red_combos = list(combinations(reds, 6))
        debug_log(f"生成 {len(red_combos)} 种红球组合")
        tickets = [ (list(r), b) for r in red_combos for b in blues ]
        debug_log(f"总生成 {len(tickets)} 注")
        return tickets
    except Exception as e:
        debug_log(f"组合生成异常：{str(e)}", 3)
        return []

def calculate_prize(tickets, prize_red, prize_blue):
    """带进度显示的奖金计算"""
    debug_log("开始奖金计算...")
    breakdown = {k:0 for k in PRIZE_TABLE}
    total = 0
    prize_red_set = set(prize_red)
    
    for i, (red, blue) in enumerate(tickets, 1):
        # 进度提示
        if i % 1000 ==0:
            debug_log(f"正在计算第 {i} 注...")
        
        # 计算匹配
        matched_red = len(set(red) & prize_red_set)
        matched_blue = blue == prize_blue
        
        # 确定奖级
        level = 0
        if matched_red ==6:
            level =1 if matched_blue else 2
        elif matched_red ==5:
            level =3 if matched_blue else 4
        elif matched_red ==4:
            level =4 if matched_blue else 5
        elif matched_red ==3 and matched_blue:
            level =5
        elif matched_blue:
            level =6
        
        if level in PRIZE_TABLE:
            total += PRIZE_TABLE[level]
            breakdown[level] +=1
    
    debug_log("奖金计算完成")
    return total, breakdown

def save_report(result):
    """带格式的报表生成"""
    debug_log("生成最终报告...")
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = [
            f"\n{'='*60}",
            f"双色球收益报告 {timestamp}",
            f"开奖号码：红球{result['prize_red']} 蓝球{result['prize_blue']}",
            f"投注组合：推荐{result['rec_count']}注 + 复式{result['complex_count']}注",
            '-'*60
        ]
        
        if result['total'] >0:
            for level in sorted(PRIZE_TABLE):
                if count := result['breakdown'].get(level, 0):
                    report.append(f"【{level}等奖】{count}注 × {PRIZE_TABLE[level]:,}元")
        else:
            report.append("未中奖")
        
        report.extend([
            f"\n总奖金：{result['total']:,}元",
            '='*60 + '\n'
        ])
        
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        debug_log(f"报告已保存至 {OUTPUT_FILE}")
    except Exception as e:
        debug_log(f"报告保存失败：{str(e)}", 3)

def main():
    debug_log("====== 程序启动 ======", 1)
    
    # 文件处理
    analysis_file = find_analysis_file()
    if not analysis_file:
        return
    
    # 数据解析
    rec_tickets, (complex_red, complex_blue) = parse_analysis_file(analysis_file)
    complex_tickets = generate_complex_tickets(complex_red, complex_blue)
    
    # 获取开奖结果
    prize_red, prize_blue = get_latest_result()
    if not prize_red:
        return
    
    # 合并投注
    all_tickets = rec_tickets + complex_tickets
    debug_log(f"总投注数：{len(all_tickets)}")
    
    # 计算奖金
    total, breakdown = calculate_prize(all_tickets, prize_red, prize_blue)
    
    # 保存结果
    save_report({
        'prize_red': prize_red,
        'prize_blue': prize_blue,
        'rec_count': len(rec_tickets),
        'complex_count': len(complex_tickets),
        'breakdown': breakdown,
        'total': total
    })
    
    # 控制台输出
    print("\n最终结果：")
    print(f"红球开奖：{prize_red}")
    print(f"蓝球开奖：{prize_blue}")
    print(f"总投注数：{len(all_tickets)}注")
    print(f"中奖金额：{total:,}元")

if __name__ == "__main__":
    main()

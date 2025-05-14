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
    """调试信息输出"""
    prefix = {
        1: "[INFO]",
        2: "[WARNING]",
        3: "[ERROR]"
    }.get(level, "[DEBUG]")
    print(f"{prefix} {datetime.now().strftime('%H:%M:%S')} {message}")

def find_analysis_file():
    """查找分析报告文件"""
    debug_log("开始查找分析报告文件...")
    try:
        files = []
        for f in glob.glob(REPORT_PATTERN):
            timestamp = re.search(r'_(\d{8}_\d{6})\.', f)
            if timestamp:
                files.append((timestamp.group(1), f))
        
        if not files:
            debug_log("未找到任何分析报告文件", 3)
            return None
        
        files.sort(reverse=True)
        debug_log(f"找到 {len(files)} 个报告文件，时间戳范围: {files[-1][0]} ~ {files[0][0]}")
        
        if len(files) >= 2:
            target_file = files[1][1]
            debug_log(f"选择第二新的文件: {target_file}")
            return target_file
        
        debug_log("不足两个文件，返回最新文件", 2)
        return files[0][1]
    
    except Exception as e:
        debug_log(f"文件查找失败: {str(e)}", 3)
        return None

def parse_analysis_file(file_path):
    """解析分析报告"""
    debug_log(f"开始解析分析报告: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取推荐组合
        rec_matches = re.findall(
            r'组合 \d+: 红球 $$([\d, ]+)$$ 蓝球 (\d+)',
            content
        )
        debug_log(f"找到 {len(rec_matches)} 组推荐组合")
        
        recommendations = []
        for i, (red, blue) in enumerate(rec_matches[:5], 1):
            red_balls = list(map(int, red.split(', ')))
            recommendations.append((red_balls, int(blue)))
            debug_log(f"推荐组合 #{i}: 红球{red_balls} 蓝球{blue}")
        
        # 提取复式组合
        complex_match = re.search(
            r'7\+7复式选号.*?红球: $$([\d, ]+)$$.*?蓝球: $$([\d, ]+)$$', 
            content, re.DOTALL
        )
        if complex_match:
            complex_red = list(map(int, complex_match.group(1).split(', ')))
            complex_blue = list(map(int, complex_match.group(2).split(', ')))
            debug_log(f"复式组合解析成功 - 红球{complex_red} 蓝球{complex_blue}")
        else:
            debug_log("未找到复式组合", 2)
            complex_red, complex_blue = [], []
        
        return recommendations, (complex_red, complex_blue)
    
    except Exception as e:
        debug_log(f"报告解析失败: {str(e)}", 3)
        return None, (None, None)

def get_latest_result():
    """获取最新开奖结果"""
    debug_log(f"开始读取CSV文件: {CSV_FILE}")
    try:
        # 自动检测编码
        encodings = ['utf-8', 'gbk', 'gb2312']
        content = None
        for encoding in encodings:
            try:
                with open(CSV_FILE, 'r', encoding=encoding) as f:
                    content = list(csv.reader(f))
                debug_log(f"成功使用 {encoding} 编码读取文件")
                break
            except UnicodeDecodeError:
                continue
        
        if not content:
            debug_log("无法解码CSV文件", 3)
            return None, None
        
        # 过滤空行
        valid_rows = [row for row in content if len(row) >=4]
        debug_log(f"找到 {len(valid_rows)} 条有效记录")
        
        if not valid_rows:
            debug_log("CSV文件中没有有效数据", 3)
            return None, None
        
        last_row = valid_rows[-1]
        debug_log(f"最新开奖记录: {last_row}")
        
        try:
            prize_red = list(map(int, last_row[2].split(',')))
            prize_blue = int(last_row[3])
            debug_log(f"解析成功 - 红球: {prize_red} 蓝球: {prize_blue}")
            return prize_red, prize_blue
        except (ValueError, IndexError) as e:
            debug_log(f"数据格式错误: {str(e)}", 3)
            return None, None
    
    except FileNotFoundError:
        debug_log("CSV文件不存在", 3)
        return None, None

def generate_tickets(complex_red, complex_blue):
    """生成复式组合"""
    debug_log("开始生成复式组合...")
    try:
        if len(complex_red) < 6:
            debug_log("红球数量不足，无法生成组合", 2)
            return []
        
        red_combos = list(combinations(complex_red, 6))
        debug_log(f"生成 {len(red_combos)} 种红球组合")
        
        tickets = []
        for r in red_combos:
            for b in complex_blue:
                tickets.append((list(r), b))
        
        debug_log(f"共生成 {len(tickets)} 注复式组合")
        return tickets
    
    except Exception as e:
        debug_log(f"组合生成失败: {str(e)}", 3)
        return []

def calculate_prize(tickets, prize_red, prize_blue):
    """计算奖金"""
    debug_log("开始计算奖金...")
    breakdown = {k:0 for k in PRIZE_TABLE}
    total = 0
    
    for i, (red, blue) in enumerate(tickets, 1):
        red_match = len(set(red) & set(prize_red))
        blue_match = (blue == prize_blue)
        
        level = 0
        if red_match == 6:
            level = 1 if blue_match else 2
        elif red_match == 5:
            level = 3 if blue_match else 4
        elif red_match == 4:
            level = 4 if blue_match else 5
        elif red_match == 3 and blue_match:
            level = 5
        elif blue_match:
            level = 6
        
        if level in PRIZE_TABLE:
            total += PRIZE_TABLE[level]
            breakdown[level] += 1
        
        if i % 100 == 0:
            debug_log(f"已计算 {i}/{len(tickets)} 注...")
    
    debug_log("奖金计算完成")
    return total, breakdown

def save_report(result, total):
    """保存结果报告"""
    debug_log("正在生成最终报告...")
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = [
            f"\n{'='*40} 双色球收益报告 {timestamp} {'='*40}",
            f"开奖号码：红球{result['prize_red']} 蓝球{result['prize_blue']}",
            f"投注组合：推荐{result['rec_count']}注 + 复式{result['complex_count']}注",
            "-"*85
        ]
        
        if total > 0:
            for level in sorted(PRIZE_TABLE):
                if count := result['breakdown'].get(level, 0):
                    report.append(f"【{level}等奖】{count}注 × {PRIZE_TABLE[level]:,}元")
        else:
            report.append("本次未中奖")
        
        report.extend([
            f"\n总奖金：{total:,}元",
            "="*100 + "\n"
        ])
        
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write("\n".join(report))
        
        debug_log(f"报告已保存至 {OUTPUT_FILE}")
    
    except Exception as e:
        debug_log(f"报告保存失败: {str(e)}", 3)

def main():
    debug_log("====== 双色球收益计算器启动 ======")
    
    # 获取分析报告
    analysis_file = find_analysis_file()
    if not analysis_file:
        debug_log("无法获取分析报告文件，程序终止", 3)
        return
    
    # 解析推荐组合
    rec_tickets, (complex_red, complex_blue) = parse_analysis_file(analysis_file)
    if not rec_tickets:
        debug_log("没有有效的推荐组合", 2)
    
    # 生成复式组合
    complex_tickets = generate_tickets(complex_red, complex_blue)
    
    # 获取开奖结果
    prize_red, prize_blue = get_latest_result()
    if not prize_red:
        debug_log("无法获取开奖结果，程序终止", 3)
        return
    
    # 合并所有组合
    all_tickets = rec_tickets + complex_tickets
    debug_log(f"总投注数：{len(all_tickets)}注")
    
    # 计算奖金
    total, breakdown = calculate_prize(all_tickets, prize_red, prize_blue)
    
    # 保存结果
    result_data = {
        'prize_red': prize_red,
        'prize_blue': prize_blue,
        'rec_count': len(rec_tickets),
        'complex_count': len(complex_tickets),
        'breakdown': breakdown
    }
    save_report(result_data, total)
    
    # 控制台输出
    print("\n最终结果：")
    print(f"开奖号码：红球{prize_red} 蓝球{prize_blue}")
    print(f"总投注：{len(all_tickets)}注（推荐{len(rec_tickets)} + 复式{len(complex_tickets)}）")
    if total > 0:
        print("中奖明细：")
        for level in sorted(breakdown):
            if count := breakdown[level]:
                print(f"  {level}等奖 × {count}注")
    print(f"总奖金：{total:,}元")

if __name__ == "__main__":
    main()

import os
import re
import glob
import csv
from itertools import combinations
from datetime import datetime

# 双色球奖金对照表（单位：元）
PRIZE_TABLE = {
    1: 5_000_000,
    2: 500_000,
    3: 3_000,
    4: 200,
    5: 10,
    6: 5
}

def find_second_newest_file(pattern):
    """查找第二新的分析报告文件"""
    files = []
    for file in glob.glob(pattern):
        try:
            # 提取文件名中的时间戳
            timestamp_str = re.search(r'_(\d{8}_\d{6})\.', file).group(1)
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            files.append((timestamp, file))
        except (AttributeError, ValueError):
            continue
    
    if len(files) < 2:
        raise FileNotFoundError("需要至少两个有效分析报告文件")
    
    # 按时间戳降序排序
    files.sort(reverse=True)
    return files[1][1]

def extract_combinations(file_path):
    """从分析报告中提取投注组合"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取5组推荐组合
    recommendations = []
    rec_section = re.search(r'生成最终推荐(.*?)推荐完成', content, re.DOTALL)
    if rec_section:
        matches = re.finditer(
            r'红球 $$([\d,\s]+)$$ 蓝球 (\d+)',
            rec_section.group(1)
        )
        for i, match in enumerate(matches):
            if i >= 5:
                break
            red = list(map(int, match.group(1).split(', ')))
            blue = int(match.group(2))
            recommendations.append((red, blue))
    
    # 提取7+7复式组合
    complex_section = re.search(r'7\+7复式选号(.*?)选择完成', content, re.DOTALL)
    complex_red = []
    complex_blue = []
    if complex_section:
        red_match = re.search(r'红球: $$([\d,\s]+)$$', complex_section.group(1))
        blue_match = re.search(r'蓝球: $$([\d,\s]+)$$', complex_section.group(1))
        if red_match:
            complex_red = list(map(int, red_match.group(1).split(', ')))
        if blue_match:
            complex_blue = list(map(int, blue_match.group(1).split(', ')))
    
    return recommendations, (complex_red, complex_blue)

def get_latest_result(csv_path):
    """从CSV文件获取最新开奖结果"""
    try:
        with open(csv_path, 'r', encoding='gbk') as f:
            reader = csv.reader(f)
            rows = [row for row in reader if len(row) >= 4]
    except UnicodeDecodeError:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = [row for row in reader if len(row) >= 4]
    
    if not rows:
        raise ValueError("CSV文件为空或格式错误")
    
    last_row = rows[-1]
    try:
        prize_red = list(map(int, last_row[2].split(',')))
        prize_blue = int(last_row[3])
    except (IndexError, ValueError) as e:
        raise ValueError("CSV格式错误") from e
    
    return prize_red, prize_blue

def generate_complex_tickets(reds, blues):
    """生成复式组合的所有单式投注"""
    return [
        (list(combo), blue)
        for combo in combinations(reds, 6)
        for blue in blues
    ]

def calculate_prize(tickets, prize_red, prize_blue):
    """计算奖金"""
    breakdown = {k:0 for k in PRIZE_TABLE}
    total = 0
    
    for red, blue in tickets:
        red_match = len(set(red) & set(prize_red))
        blue_matched = (blue == prize_blue)
        
        # 判断奖级
        if red_match == 6:
            level = 1 if blue_matched else 2
        elif red_match == 5:
            level = 3 if blue_matched else 4
        elif red_match == 4:
            level = 4 if blue_matched else 5
        elif red_match == 3 and blue_matched:
            level = 5
        elif blue_matched and red_match < 3:
            level = 6
        else:
            continue
        
        if level in PRIZE_TABLE:
            total += PRIZE_TABLE[level]
            breakdown[level] += 1
    
    return total, breakdown

def save_report(total, breakdown, prize_red, prize_blue, ticket_counts):
    """保存分析报告"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = [
        f"\n{'='*40} 双色球收益报告 {timestamp} {'='*40}",
        f"开奖号码：红球{prize_red} 蓝球{prize_blue}",
        f"投注组合：推荐组合{ticket_counts[0]}注 + 复式组合{ticket_counts[1]}注",
        "-"*100
    ]
    
    # 添加中奖明细
    has_prize = False
    for level in sorted(breakdown):
        if breakdown[level] > 0:
            has_prize = True
            report.append(
                f"【{level}等奖】{breakdown[level]}注 × {PRIZE_TABLE[level]:,}元"
            )
    
    if not has_prize:
        report.append("本次投注未中奖")
    
    report.extend([
        f"\n总奖金：{total:,}元",
        "="*100 + "\n"
    ])
    
    # 追加写入文件
    with open("latest_ssq_analysis.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(report))

def main():
    try:
        # 1. 获取输入数据
        analysis_file = find_second_newest_file("ssq_analysis_output_*.txt")
        rec_tickets, (complex_red, complex_blue) = extract_combinations(analysis_file)
        prize_red, prize_blue = get_latest_result("shuangseqiu.csv")
        
        # 2. 生成所有投注组合
        complex_tickets = generate_complex_tickets(complex_red, complex_blue)
        all_tickets = rec_tickets + complex_tickets
        
        # 3. 计算奖金
        total, breakdown = calculate_prize(all_tickets, prize_red, prize_blue)
        
        # 4. 保存报告
        save_report(total, breakdown, prize_red, prize_blue, 
                   (len(rec_tickets), len(complex_tickets)))
        
        # 5. 控制台输出
        print(f"\n最新开奖结果：红球 {prize_red} 蓝球 {prize_blue}")
        print(f"总投注数：{len(all_tickets)}注（推荐{len(rec_tickets)} + 复式{len(complex_tickets)}）")
        print("中奖明细：")
        for level in sorted(breakdown):
            if count := breakdown[level]:
                print(f"  {level}等奖 × {count}注 = {count * PRIZE_TABLE[level]:,}元")
        print(f"\n总奖金：{total:,}元")
        print("\n收益报告已追加至 latest_ssq_analysis.txt")
    
    except Exception as e:
        print(f"运行出错：{str(e)}")

if __name__ == "__main__":
    main()

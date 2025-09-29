import pandas as pd
import datetime
import os
import glob
import re
from math import comb

# --- 1. 全局常量定义 ---

# 奖金规则与名称定义
PRIZE_RULES = {
    (6, 1): 5000000, (6, 0): 100000, (5, 1): 3000, (5, 0): 200,
    (4, 1): 200, (4, 0): 10, (3, 1): 10, (2, 1): 5, (1, 1): 5, (0, 1): 5
}
PRIZE_NAMES = {
    (6, 1): "一等奖", (6, 0): "二等奖", (5, 1): "三等奖", 
    (5, 0): "四等奖", (4, 1): "四等奖", (4, 0): "五等奖", 
    (3, 1): "五等奖", (2, 1): "六等奖", (1, 1): "六等奖", (0, 1): "六等奖"
}

# --- 2. 核心功能函数 ---

def find_matching_report(target_issue):
    """
    在当前目录查找所有报告文件，并返回与目标期号匹配的报告文件路径。
    """
    report_files = glob.glob("ssq_analysis_output_*.txt")
    if not report_files:
        return None, "错误: 当前目录未找到任何 'ssq_analysis_output_*.txt' 报告文件。"

    for report_file in sorted(report_files, reverse=True): # 从最新的开始找
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith("Prediction_Target_Issue:"):
                        prediction_target = line.strip().split(": ")[1]
                        if str(prediction_target) == str(target_issue):
                            return report_file, None # 找到了匹配的文件
                        else:
                            break # 元数据不匹配，检查下一个文件
        except Exception as e:
            print(f"警告: 读取文件 {report_file} 时出错: {e}")
            continue
            
    return None, f"错误: 未找到预测目标期号为 '{target_issue}' 的报告文件。"

def parse_report_bets(filepath):
    """
    解析指定的报告文件，提取单式和复式投注。
    """
    single_bets = []
    duplex_bet = {'red': [], 'blue': []}
    
    in_single_section = False
    in_duplex_section = False

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if "【单式推荐 (10组)】" in line:
                in_single_section = True
                in_duplex_section = False
                continue
            if "【7+7 复式推荐 (1组)】" in line:
                in_single_section = False
                in_duplex_section = True
                continue

            if in_single_section and "红球" in line:
                try:
                    reds_str = re.search(r'\[(.*?)\]', line).group(1)
                    blue_str = re.search(r'蓝球 \[(.*?)\]', line).group(1)
                    reds = [int(r) for r in reds_str.split(', ')]
                    blue = int(blue_str)
                    single_bets.append({'red': reds, 'blue': blue})
                except Exception:
                    continue # 忽略格式不正确的行
            
            if in_duplex_section:
                try:
                    if "红球:" in line:
                        reds_str = re.search(r'\[(.*?)\]', line).group(1)
                        duplex_bet['red'] = [int(r) for r in reds_str.split(', ')]
                    if "蓝球:" in line:
                        blues_str = re.search(r'\[(.*?)\]', line).group(1)
                        duplex_bet['blue'] = [int(b) for b in blues_str.split(', ')]
                        in_duplex_section = False # 解析完毕
                except Exception:
                    continue

    return single_bets, duplex_bet

def calculate_single_prize(bet_reds, bet_blue, winning_reds, winning_blue):
    """计算单式票的奖金和中奖等级。"""
    red_hits = len(set(bet_reds) & winning_reds)
    blue_hit = 1 if bet_blue == winning_blue else 0
    hit_key = (red_hits, blue_hit)
    
    prize = PRIZE_RULES.get(hit_key, 0)
    prize_name = PRIZE_NAMES.get(hit_key, "未中奖")
    
    return prize, prize_name, f"命中{red_hits}+{blue_hit}"

def calculate_duplex_prize(bet_reds, bet_blues, winning_reds, winning_blue):
    """使用组合数学计算复式票的总奖金和奖项构成。"""
    total_prize = 0
    prize_breakdown = {}
    
    red_hits = len(set(bet_reds) & winning_reds)
    red_misses = len(bet_reds) - red_hits
    blue_hit = 1 if winning_blue in set(bet_blues) else 0

    for (r_needed, b_needed), prize_value in PRIZE_RULES.items():
        if r_needed > red_hits:
            continue
        
        # 计算红球组合数
        red_combos = comb(red_hits, r_needed) * comb(red_misses, 6 - r_needed)
        
        # 检查蓝球是否匹配
        if blue_hit == b_needed:
            prize_name = PRIZE_NAMES.get((r_needed, b_needed))
            prize_breakdown[prize_name] = prize_breakdown.get(prize_name, 0) + red_combos
            total_prize += red_combos * prize_value
            
    summary = f"总计命中 {red_hits} 个红球, {blue_hit} 个蓝球"
    return total_prize, prize_breakdown, summary

# --- 3. 主执行逻辑 ---

if __name__ == '__main__':
    # 1. 获取最新开奖结果
    try:
        ssq_df = pd.read_csv("shuangseqiu.csv", header=0)
        latest_draw = ssq_df.iloc[-1]
        target_issue = latest_draw['期号']
        winning_reds = set([int(r) for r in latest_draw['红球'].split(',')])
        winning_blue = int(latest_draw['蓝球'])
    except Exception as e:
        print(f"读取 ssq.csv 文件失败: {e}")
        exit()

    # 2. 查找匹配的报告
    report_filepath, error_msg = find_matching_report(target_issue)
    if error_msg:
        print(error_msg)
        exit()
        
    # 3. 解析报告中的投注
    single_bets, duplex_bet = parse_report_bets(report_filepath)
    if not single_bets or not duplex_bet['red']:
        print(f"错误: 未能从报告 {report_filepath} 中成功解析出投注号码。")
        exit()

    # 4. 计算奖金
    total_single_bonus = 0
    single_details = []
    for i, bet in enumerate(single_bets, 1):
        prize, prize_name, summary = calculate_single_prize(bet['red'], bet['blue'], winning_reds, winning_blue)
        total_single_bonus += prize
        single_details.append(f"  组合 {i:>2}: {str(bet['red']):<24} 蓝球 [{bet['blue']:02d}] -> {summary}, {prize_name}, 奖金: {prize} 元")

    duplex_prize, duplex_breakdown, duplex_summary = calculate_duplex_prize(duplex_bet['red'], duplex_bet['blue'], winning_reds, winning_blue)

    # 5. 构建并输出报告
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("          双色球推荐核对报告")
    report_lines.append("="*70)
    report_lines.append(f"\n报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"核对报告文件: {os.path.basename(report_filepath)}")
    report_lines.append(f"核对开奖期数: {target_issue}")
    report_lines.append(f"官方开奖号码: 红球 {sorted(list(winning_reds))}  蓝球 [{winning_blue}]")

    report_lines.append("\n--- 1. 单式推荐核对详情 ---")
    report_lines.extend(single_details)
    report_lines.append(f"\n单式推荐总奖金: {total_single_bonus} 元")

    report_lines.append("\n--- 2. 复式推荐核对详情 ---")
    report_lines.append(f"  红球: {duplex_bet['red']}")
    report_lines.append(f"  蓝球: {duplex_bet['blue']}")
    report_lines.append(f"  核对结果: {duplex_summary}")
    if not duplex_breakdown:
        report_lines.append("  奖项构成: 未中任何奖项。")
    else:
        report_lines.append("  奖项构成:")
        for name, count in sorted(duplex_breakdown.items(), key=lambda item: list(PRIZE_NAMES.values()).index(item[0])):
             report_lines.append(f"    - {name}: {count} 注")
    report_lines.append(f"\n复式推荐总奖金: {duplex_prize} 元")
    
    report_lines.append("\n" + "-"*70)
    report_lines.append(f"总计中奖金额: {total_single_bonus + duplex_prize} 元")
    report_lines.append("="*70)

    final_report_string = "\n".join(report_lines)
    print("\n" + final_report_string)

    # 6. 写入文件
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ssq_bonus_check_{timestamp}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(final_report_string)
        print(f"\n核对报告已成功保存到文件: {filename}")
    except Exception as e:
        print(f"\n写入核对报告文件失败: {e}")

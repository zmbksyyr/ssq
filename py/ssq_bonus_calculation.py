import pandas as pd
import datetime
import os
import glob
import re
from math import comb
from ssq_core import (
    PRIZE_NAMES, PRIZE_RULES, atomic_write_text, parse_blue_ball,
    parse_blue_balls, parse_issue, parse_red_balls,
)

# --- 动态路径设置 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
CSV_PATH = os.path.join(root_dir, 'shuangseqiu.csv')
REPORT_DIR = os.path.join(root_dir, 'report')

# --- 2. 核心功能函数 ---

def find_matching_report(target_issue):
    """
    在 report/ 目录查找所有报告文件，并返回与目标期号匹配的报告文件路径。
    """
    report_pattern = os.path.join(REPORT_DIR, "ssq_analysis_output_*.txt")
    report_files = glob.glob(report_pattern)
    
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

SINGLE_HEADER_PATTERN = re.compile(r'^【单式推荐 \((\d+)组\)】$', re.MULTILINE)
DUPLEX_HEADER_PATTERN = re.compile(r'^【7\+N 复式推荐 \(1组\)】$', re.MULTILINE)
SINGLE_BET_PATTERN = re.compile(
    r'^组合\s+\d+:\s+红球\s+\[(.*?)\]\s+蓝球\s+\[(.*?)\]$'
)


def parse_single_bet_line(line):
    match = SINGLE_BET_PATTERN.fullmatch(line)
    if match is None:
        raise ValueError("字段格式不完整")
    return {
        'red': parse_red_balls(match.group(1)),
        'blue': parse_blue_ball(match.group(2)),
    }


def parse_duplex_section(section):
    red_match = re.search(r'^\s*红球:\s*\[(.*?)\]\s*$', section, re.MULTILINE)
    blue_match = re.search(r'^\s*蓝球:\s*\[(.*?)\]\s*$', section, re.MULTILINE)
    if red_match is None or blue_match is None:
        raise ValueError("复式投注内容不完整")
    return {
        'red': parse_red_balls(red_match.group(1), expected_count=7),
        'blue': parse_blue_balls(blue_match.group(1)),
    }


def validate_parsed_bets(expected_count, single_bets):
    if len(single_bets) != expected_count:
        raise ValueError(
            f"单式投注数量不完整: 声明 {expected_count} 注，"
            f"实际解析 {len(single_bets)} 注"
        )
    unique_singles = {
        (tuple(bet['red']), bet['blue']) for bet in single_bets
    }
    if len(unique_singles) != len(single_bets):
        raise ValueError("报告包含重复单式投注")


def parse_report_bets(filepath):
    """Parse a complete set of single and duplex bets from an analysis report."""
    with open(filepath, 'r', encoding='utf-8') as report_file:
        content = report_file.read()
    single_header = SINGLE_HEADER_PATTERN.search(content)
    if single_header is None:
        raise ValueError("报告缺少或无法识别单式推荐标题")
    duplex_header = DUPLEX_HEADER_PATTERN.search(content, single_header.end())
    if duplex_header is None:
        raise ValueError("报告缺少或无法识别复式推荐标题")

    single_section = content[single_header.end():duplex_header.start()]
    single_lines = [
        line.strip() for line in single_section.splitlines()
        if line.strip().startswith('组合')
    ]
    try:
        single_bets = [parse_single_bet_line(line) for line in single_lines]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"单式投注解析失败: {exc}") from exc
    validate_parsed_bets(int(single_header.group(1)), single_bets)

    try:
        duplex_bet = parse_duplex_section(content[duplex_header.end():])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"复式投注解析失败: {exc}") from exc
    return single_bets, duplex_bet

def calculate_single_prize(bet_reds, bet_blue, winning_reds, winning_blue):
    """计算单式票的奖金和中奖等级。"""
    bet_red_set = set(parse_red_balls(bet_reds))
    winning_red_set = set(parse_red_balls(winning_reds))
    bet_blue = parse_blue_ball(bet_blue)
    winning_blue = parse_blue_ball(winning_blue)
    red_hits = len(bet_red_set & winning_red_set)
    blue_hit = 1 if bet_blue == winning_blue else 0
    hit_key = (red_hits, blue_hit)
    
    prize = PRIZE_RULES.get(hit_key, 0)
    prize_name = PRIZE_NAMES.get(hit_key, "未中奖")
    
    return prize, prize_name, f"命中{red_hits}+{blue_hit}"

def calculate_duplex_prize(bet_reds, bet_blues, winning_reds, winning_blue):
    """使用组合数学计算复式票的总奖金和奖项构成。"""
    bet_reds = parse_red_balls(bet_reds, expected_count=7)
    bet_blues = parse_blue_balls(bet_blues)
    winning_reds = set(parse_red_balls(winning_reds))
    winning_blue = parse_blue_ball(winning_blue)
    total_prize = 0
    prize_breakdown = {}
    
    red_hits = len(set(bet_reds) & winning_reds)
    red_misses = len(bet_reds) - red_hits
    unique_blues = set(bet_blues)
    blue_hit = 1 if winning_blue in unique_blues else 0

    for (r_needed, b_needed), prize_value in PRIZE_RULES.items():
        if r_needed > red_hits:
            continue
        
        # 计算红球组合数
        red_combos = comb(red_hits, r_needed) * comb(red_misses, 6 - r_needed)
        
        if b_needed == 1:
            blue_combos = blue_hit
        else:
            blue_combos = len(unique_blues) - blue_hit
        winning_tickets = red_combos * blue_combos
        if winning_tickets:
            prize_name = PRIZE_NAMES.get((r_needed, b_needed))
            prize_breakdown[prize_name] = prize_breakdown.get(prize_name, 0) + winning_tickets
            total_prize += winning_tickets * prize_value
            
    summary = f"总计命中 {red_hits} 个红球, {blue_hit} 个蓝球"
    return total_prize, prize_breakdown, summary


def load_latest_draw(filepath=CSV_PATH):
    frame = pd.read_csv(filepath, header=0)
    required = {'期号', '日期', '红球', '蓝球'}
    if not required.issubset(frame.columns):
        raise ValueError(f"开奖数据缺少字段: {sorted(required - set(frame.columns))}")
    frame['期号'] = frame['期号'].apply(parse_issue)
    if frame['期号'].duplicated().any():
        raise ValueError("开奖数据存在重复期号")
    if frame.empty:
        raise ValueError("开奖数据为空")
    frame['_parsed_date'] = pd.to_datetime(
        frame['日期'], format='%Y-%m-%d', errors='raise'
    )
    if ((frame['期号'] // 1000) != frame['_parsed_date'].dt.year).any():
        raise ValueError("期号年份与开奖日期不一致")
    ordered = frame.sort_values('期号')
    if not ordered['_parsed_date'].is_monotonic_increasing:
        raise ValueError("期号与开奖日期顺序不一致")
    latest = ordered.iloc[-1]
    return {
        'issue': int(latest['期号']),
        'red': set(parse_red_balls(latest['红球'])),
        'blue': parse_blue_ball(latest['蓝球']),
    }

# --- 3. 主执行逻辑 ---

if __name__ == '__main__':
    # 1. 获取最新开奖结果
    try:
        latest_draw = load_latest_draw()
        target_issue = latest_draw['issue']
        winning_reds = latest_draw['red']
        winning_blue = latest_draw['blue']
    except Exception as e:
        print(f"读取 {CSV_PATH} 文件失败: {e}")
        raise SystemExit(1)

    # 2. 查找匹配的报告
    report_filepath, error_msg = find_matching_report(target_issue)
    if error_msg:
        print(error_msg)
        raise SystemExit(1)
        
    # 3. 解析报告中的投注
    try:
        single_bets, duplex_bet = parse_report_bets(report_filepath)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"错误: 无法解析报告 {report_filepath}: {exc}")
        raise SystemExit(1)
    if not single_bets or not duplex_bet['red'] or not duplex_bet['blue']:
        print(f"错误: 未能从报告 {report_filepath} 中成功解析出投注号码。")
        raise SystemExit(1)

    # 4. 计算奖金
    total_single_bonus = 0
    single_details = []
    for i, bet in enumerate(single_bets, 1):
        prize, prize_name, summary = calculate_single_prize(bet['red'], bet['blue'], winning_reds, winning_blue)
        total_single_bonus += prize
        single_details.append(f"  组合 {i:>2}: {str(bet['red']):<24} 蓝球 [{bet['blue']:02d}] -> {summary}, {prize_name}, 参考奖金: {prize} 元")

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
    report_lines.append("奖金说明: 使用固定参考金额估算；一等奖、二等奖实际金额以官方派奖为准。")

    report_lines.append("\n--- 1. 单式推荐核对详情 ---")
    report_lines.extend(single_details)
    report_lines.append(f"\n单式推荐参考奖金: {total_single_bonus} 元")

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
    report_lines.append(f"\n复式推荐参考奖金: {duplex_prize} 元")
    
    report_lines.append("\n" + "-"*70)
    report_lines.append(f"总计参考奖金: {total_single_bonus + duplex_prize} 元")
    report_lines.append("="*70)

    final_report_string = "\n".join(report_lines)
    print("\n" + final_report_string)

    # 6. 写入文件
    try:
        os.makedirs(REPORT_DIR, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ssq_bonus_check_{timestamp}.txt"
        
        filepath = os.path.join(REPORT_DIR, filename)
        
        atomic_write_text(filepath, final_report_string)
        print(f"\n核对报告已成功保存到文件: {filepath}")
    except Exception as e:
        raise SystemExit(f"\n写入核对报告文件失败: {e}")

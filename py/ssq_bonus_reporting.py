"""Formatting for recommendation prize-check reports."""

import os

from ssq_core import PRIZE_NAMES
from ssq_prizes import calculate_duplex_prize, calculate_single_prize


def build_bonus_report(
    report_filepath,
    target_issue,
    winning_reds,
    winning_blue,
    single_bets,
    duplex_bet,
    generated_at,
):
    """Build one deterministic report from validated draws and bets."""
    total_single_bonus = 0
    single_details = []
    for index, bet in enumerate(single_bets, 1):
        prize, prize_name, summary = calculate_single_prize(
            bet['red'], bet['blue'], winning_reds, winning_blue
        )
        total_single_bonus += prize
        single_details.append(
            f"  组合 {index:>2}: {bet['red']!s:<24} 蓝球 [{bet['blue']:02d}] "
            f'-> {summary}, {prize_name}, 参考奖金: {prize} 元'
        )

    duplex_prize, duplex_breakdown, duplex_summary = calculate_duplex_prize(
        duplex_bet['red'], duplex_bet['blue'], winning_reds, winning_blue
    )
    lines = [
        '=' * 70,
        '          双色球推荐核对报告',
        '=' * 70,
        f"\n报告生成时间: {generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f'核对报告文件: {os.path.basename(report_filepath)}',
        f'核对开奖期数: {target_issue}',
        f'官方开奖号码: 红球 {sorted(winning_reds)}  蓝球 [{winning_blue}]',
        '奖金说明: 使用固定参考金额估算；一等奖、二等奖实际金额以官方派奖为准。',
        '\n--- 1. 单式推荐核对详情 ---',
        *single_details,
        f'\n单式推荐参考奖金: {total_single_bonus} 元',
        '\n--- 2. 复式推荐核对详情 ---',
        f"  红球: {duplex_bet['red']}",
        f"  蓝球: {duplex_bet['blue']}",
        f'  核对结果: {duplex_summary}',
    ]
    if not duplex_breakdown:
        lines.append('  奖项构成: 未中任何奖项。')
    else:
        prize_order = {name: index for index, name in enumerate(PRIZE_NAMES.values())}
        lines.append('  奖项构成:')
        for name, count in sorted(
            duplex_breakdown.items(), key=lambda item: prize_order[item[0]]
        ):
            lines.append(f'    - {name}: {count} 注')
    lines.extend([
        f'\n复式推荐参考奖金: {duplex_prize} 元',
        '\n' + '-' * 70,
        f'总计参考奖金: {total_single_bonus + duplex_prize} 元',
        '=' * 70,
    ])
    return '\n'.join(lines)

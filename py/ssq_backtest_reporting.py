"""Formatting for strategy settings and rolling backtest metrics."""

from ssq_core import PRIZE_NAMES
from ssq_rank_bands import RANK_BAND_NAMES

PRIZE_DISPLAY_ORDER = (
    '一等奖', '二等奖', '三等奖', '四等奖', '五等奖', '六等奖',
)


def format_strategy_parameters(data):
    config = data.config
    lines = ["\n--- 1. 策略参数与回测 ---"]
    mode_desc = "加载已固化的参数" if data.params_loaded else "使用内置的默认参数"
    lines.append(f"模式: {mode_desc}")
    middle_count = config.pool_size_red - config.high_count - config.low_count
    lines.append(f"  - red_pool_size       : {config.pool_size_red}")
    lines.append(
        '  - mixed_pool_bands    : '
        f'{config.high_count} high + {middle_count} middle + '
        f'{config.low_count} low'
    )
    lines.append(f"  - blue_count          : {config.blue_count}")
    lines.append(f"  - recommendation_count: {config.recommendation_count}")
    lines.append(f"  - max_shared_red_balls: {config.max_shared_red_balls}")
    lines.append(f"  - anti_crowding_size  : {config.rejection_lib_size}")
    lines.append(
        f"  - anti_crowding_seed  : {config.random_seed} -> "
        f"{data.rejection_seed} (目标期派生)"
    )
    for key, value in data.params.items():
        lines.append(f"  - {key:<20}: {value}")
    return lines


def format_backtest_metrics(data):
    backtest = data.backtest
    config = data.config
    lines = [
        (
            f"\n单式策略滚动回测 ({backtest.periods}期，每期最多"
            f"{config.recommendation_count}注，不含复式):"
        )
    ]
    lines.append(f"  - 候选池模式: {data.pool_mode}")
    lines.append(f"  - 成功建模评估期数: {backtest.evaluated_periods}")
    lines.append(f"  - 实际投注期数: {backtest.active_periods}")
    lines.append(f"  - 投注注数: {backtest.tickets}")
    lines.append(f"  - 候选池平均覆盖红球: {backtest.average_pool_red_hits:.2f}/6")
    lines.append(f"  - 单注平均命中红球: {backtest.average_ticket_red_hits:.3f}/6")
    lines.append(
        f"  - 候选全集平均命中红球: {backtest.average_candidate_red_hits:.3f}/6，"
        f"最终排序增益 {backtest.ranking_red_hit_delta:+.3f}"
    )
    lines.append(
        f"  - 命中至少3个红球: {backtest.three_plus_red_tickets} 注 "
        f"({backtest.three_plus_red_rate:.2%})"
    )
    lines.append(
        f"  - 候选全集3+红比例: {backtest.candidate_three_plus_red_rate:.2%}，"
        f"最终排序增益 {backtest.ranking_three_plus_delta:+.2%}"
    )
    lines.append(
        f"  - 最高分蓝球命中: {backtest.blue_hit_periods}/"
        f"{backtest.evaluated_periods} ({backtest.blue_hit_rate:.2%})"
    )
    lines.append(f"  - 总投入: {backtest.cost:.2f} 元")
    lines.append(f"  - 固定参考奖金: {backtest.winnings:.2f} 元")
    lines.append(f"  - 参考净收益: {backtest.profit:.2f} 元")
    lines.append(f"  - 参考回报率: {backtest.roi:.2%}")
    return lines


def format_pool_comparison(backtests):
    if len(backtests) <= 1:
        return []
    lines = ["  - 候选池对照:"]
    for name, result in backtests.items():
        lines.append(
            f"    {name:<6} 投入 {result.cost:>6.0f} 元，参考奖金 {result.winnings:>6.0f} 元，"
            f"参考净收益 {result.profit:>7.0f} 元，参考回报率 {result.roi:>7.2%}，"
            f"池覆盖 {result.average_pool_red_hits:.2f}/6，"
            f"单注红球 {result.average_ticket_red_hits:.3f}/6 "
            f"({result.ranking_red_hit_delta:+.3f})，"
            f"3+红 {result.three_plus_red_rate:.2%} "
            f"({result.ranking_three_plus_delta:+.2%})"
        )
    return lines


def format_window_stability(backtest, backtests):
    if not backtest.windows:
        return []
    earlier_periods = backtest.windows['earlier'].periods
    recent_periods = backtest.windows['recent'].periods
    lines = [
        (
            f"  - 候选池分段稳定性 (较早{earlier_periods}期 / "
            f"最近{recent_periods}期验证):"
        )
    ]
    for name, result in backtests.items():
        earlier = result.windows['earlier']
        recent = result.windows['recent']
        lines.append(
            f"    {name:<6} 池覆盖 {earlier.average_pool_red_hits:.2f} -> "
            f"{recent.average_pool_red_hits:.2f}/6，单注红球 "
            f"{earlier.average_ticket_red_hits:.3f} -> "
            f"{recent.average_ticket_red_hits:.3f}/6，3+红 "
            f"{earlier.three_plus_red_rate:.2%} -> "
            f"{recent.three_plus_red_rate:.2%}，排序增益 "
            f"{earlier.ranking_red_hit_delta:+.3f} -> "
            f"{recent.ranking_red_hit_delta:+.3f}"
        )
    return lines


def format_rank_band_distribution(data):
    backtest = data.backtest
    lines = ["  - 实际红球在模型评分排名中的分布:"]
    total_rank_width = sum(data.rank_band_widths.values())
    for band in RANK_BAND_NAMES:
        label = data.rank_band_labels[band]
        band_width = data.rank_band_widths[band]
        lines.append(
            f"    {label:<13}: {backtest.rank_band_hits[band]:>3} 个 "
            f"(占比 {backtest.rank_band_rate(band):.2%}，"
            f"随机基线 {band_width / total_rank_width:.2%}，"
            f"相对 {backtest.rank_band_lift(band):.2f}x)"
        )
    return lines


def format_prize_counts(backtest):
    lines = ["中奖详情如下："]
    aggregated_counts = {name: 0 for name in set(PRIZE_NAMES.values())}
    for hit, count in backtest.prize_counts.items():
        if count > 0:
            aggregated_counts[PRIZE_NAMES[hit]] += count

    if sum(aggregated_counts.values()) == 0:
        lines.append("  - 未中任何奖项。")
    else:
        for prize_name in PRIZE_DISPLAY_ORDER:
            count = aggregated_counts.get(prize_name, 0)
            if count > 0:
                lines.append(f"  - {prize_name:<5}: {count} 次")
    return lines


def format_backtest_report(data):
    lines = format_strategy_parameters(data)
    lines.extend(format_backtest_metrics(data))
    lines.extend(format_pool_comparison(data.backtests))
    lines.extend(format_window_stability(data.backtest, data.backtests))
    lines.extend(format_rank_band_distribution(data))
    lines.extend(format_prize_counts(data.backtest))
    return lines

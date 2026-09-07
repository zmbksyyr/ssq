"""Pure text formatting for analysis reports."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ssq_core import PRIZE_NAMES
from ssq_rules import FILTER_NAMES, HARD_FILTER_NAMES

RANK_BAND_NAMES = ('high', 'middle', 'low', 'other')
PRIZE_DISPLAY_ORDER = (
    '一等奖', '二等奖', '三等奖', '四等奖', '五等奖', '六等奖',
)


def format_audit_window(actual_periods, requested_periods):
    if actual_periods == requested_periods:
        return f'最近 {actual_periods} 期'
    return f'实际 {actual_periods} 期，请求 {requested_periods} 期'


@dataclass(frozen=True)
class AnalysisReportData:
    latest_issue: str
    target_issue: int
    generated_at: datetime
    params_loaded: bool
    params: dict
    config: Any
    rejection_seed: int
    backtest: Any
    backtests: dict[str, Any]
    pool_mode: str
    rank_band_widths: dict[str, int]
    rank_band_labels: dict[str, str]
    pipeline_stats: list[dict]
    rule_coverage: dict
    hard_pipeline_coverage: dict
    rule_audit_periods: int
    selection: Any
    recommended_blues: list[int]
    best_7_reds: list


def format_backtest_report(data):
    backtest = data.backtest
    config = data.config
    lines = ["\n--- 1. 策略参数与回测 ---"]
    mode_desc = "加载已固化的参数" if data.params_loaded else "使用内置的默认参数"
    lines.append(f"模式: {mode_desc}")
    lines.append(f"  - anti_crowding_size  : {config.rejection_lib_size}")
    lines.append(
        f"  - anti_crowding_seed  : {config.random_seed} -> "
        f"{data.rejection_seed} (目标期派生)"
    )
    for key, value in data.params.items():
        lines.append(f"  - {key:<20}: {value}")

    lines.append(
        f"\n单式策略滚动回测 ({backtest.periods}期，每期最多"
        f"{config.recommendation_count}注，不含复式):"
    )
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
    if len(data.backtests) > 1:
        lines.append("  - 候选池对照:")
        for name, result in data.backtests.items():
            lines.append(
                f"    {name:<6} 投入 {result.cost:>6.0f} 元，参考奖金 {result.winnings:>6.0f} 元，"
                f"参考净收益 {result.profit:>7.0f} 元，参考回报率 {result.roi:>7.2%}，"
                f"池覆盖 {result.average_pool_red_hits:.2f}/6，"
                f"单注红球 {result.average_ticket_red_hits:.3f}/6 "
                f"({result.ranking_red_hit_delta:+.3f})，"
                f"3+红 {result.three_plus_red_rate:.2%} "
                f"({result.ranking_three_plus_delta:+.2%})"
            )
    if backtest.windows:
        earlier_periods = backtest.windows['earlier'].periods
        recent_periods = backtest.windows['recent'].periods
        lines.append(
            f"  - 候选池分段稳定性 (较早{earlier_periods}期 / "
            f"最近{recent_periods}期验证):"
        )
        for name, result in data.backtests.items():
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
    lines.append("  - 实际红球在模型评分排名中的分布:")
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
    lines.append("中奖详情如下：")
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


def format_rule_audit_report(data):
    lines = ["\n硬规则过滤统计 (按流水线累计):"]
    for item in data.pipeline_stats:
        lines.append(
            f"  - {item['rule']:<22}: {item['before']} -> {item['remaining']} "
            f"(remove {item['removed']})"
        )
    inactive_rules = [
        item['rule'] for item in data.pipeline_stats if item['removed'] == 0
    ]
    aggressive_rules = [
        item['rule'] for item in data.pipeline_stats
        if item['before'] and item['removed'] / item['before'] >= 0.5
    ]
    if inactive_rules:
        lines.append(f"  提示：本轮未淘汰组合的规则: {', '.join(inactive_rules)}")
    if aggressive_rules:
        lines.append(f"  提示：淘汰比例达到或超过50%的规则: {', '.join(aggressive_rules)}")

    independent_total = next(iter(data.rule_coverage.values()), {}).get('total', 0)
    independent_window = format_audit_window(
        independent_total,
        data.rule_audit_periods,
    )
    lines.append(
        f"\n真实开奖规则覆盖率 ({independent_window}，逐条独立统计):"
    )
    for name in FILTER_NAMES:
        result = data.rule_coverage[name]
        rule_type = '硬' if name in HARD_FILTER_NAMES else '软'
        lines.append(
            f"  - [{rule_type}] {name:<22}: {result['passed']}/"
            f"{result['total']} ({result['rate']:.2%})"
        )

    pipeline_total = data.hard_pipeline_coverage['total']
    pipeline_window = format_audit_window(
        pipeline_total,
        data.rule_audit_periods,
    )
    lines.append(
        f"\n真实开奖硬规则累计覆盖率 ({pipeline_window}，不含随机撞号):"
    )
    for item in data.hard_pipeline_coverage['stages']:
        lines.append(
            f"  - {item['rule']:<22}: {item['before']} -> {item['remaining']} "
            f"(新增排除 {item['removed']} 期)"
        )
    lines.append(
        f"  - 合计保留: {data.hard_pipeline_coverage['passed']}/"
        f"{data.hard_pipeline_coverage['total']} "
        f"({data.hard_pipeline_coverage['rate']:.2%})"
    )
    return lines


def format_recommendations_report(data):
    lines = ["\n--- 2. 推荐组合 ---"]
    top_blue = data.recommended_blues[0] if data.recommended_blues else None
    single_combos = data.selection.recommendations if top_blue is not None else ()
    lines.append(f"\n【单式推荐 ({len(single_combos)}组)】")
    if single_combos:
        for index, combo in enumerate(single_combos, 1):
            lines.append(
                f"  组合 {index:>2}: 红球 {list(combo)!s:<24} 蓝球 [{top_blue:02d}]"
            )
    else:
        lines.append("  - 未能生成足够的单式组合。")

    lines.append("\n【7+N 复式推荐 (1组)】")
    if data.best_7_reds and data.recommended_blues:
        lines.append(f"  红球: {list(data.best_7_reds[0][0])}")
        lines.append(f"  蓝球: {data.recommended_blues}")
    else:
        lines.append("  - 未能生成足够的复式组合。")
    return lines


def build_analysis_report(data):
    lines = [
        "=" * 60,
        "          双色球策略分析与推荐报告 (高级过滤版)",
        "=" * 60,
        "\n--- 0. 报告元数据 ---",
        f"Data_Basis_Issue: {data.latest_issue}",
        f"Prediction_Target_Issue: {data.target_issue}",
        f"报告生成时间: {data.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    lines.extend(format_backtest_report(data))
    lines.extend(format_rule_audit_report(data))
    lines.extend(format_recommendations_report(data))
    lines.append("\n" + "=" * 60 + "\n报告结束。祝您好运！\n" + "=" * 60)
    return "\n".join(lines)

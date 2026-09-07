"""Formatting for current and historical rule audit results."""

from ssq_rule_registry import FILTER_NAMES, HARD_FILTER_NAMES


def format_audit_window(actual_periods, requested_periods):
    if actual_periods == requested_periods:
        return f'最近 {actual_periods} 期'
    return f'实际 {actual_periods} 期，请求 {requested_periods} 期'


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

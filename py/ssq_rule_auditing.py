"""Leakage-free historical coverage audits for selection rules."""

from collections import Counter

from ssq_config import RULE_AUDIT_PERIODS
from ssq_modeling import get_omission
from ssq_rules import FILTER_NAMES, RED_RULES, RuleContext, explain_filter_failures


def historical_rule_context(full_df, index):
    """Build rule inputs using only draws before the audited issue."""
    history = full_df.iloc[:index]
    recent = [set(draw) for draw in history.iloc[-10:]['红球']]
    return RuleContext(
        omission_values=get_omission(history),
        recent_draws=recent,
        last_draw=recent[-1],
        previous_draw=recent[-2],
    )


def audit_historical_rule_coverage(full_df, periods=RULE_AUDIT_PERIODS):
    """Measure how often each strategy rule accepts actual historical draws."""
    if periods <= 0 or len(full_df) < 11:
        return {
            name: {'passed': 0, 'total': 0, 'rate': 0.0}
            for name in FILTER_NAMES
        }
    start = max(10, len(full_df) - periods)
    passed_counts = Counter()
    total = 0
    for index in range(start, len(full_df)):
        combo = tuple(full_df.iloc[index]['红球'])
        failures = set(explain_filter_failures(
            combo,
            historical_rule_context(full_df, index),
        ))
        for name in FILTER_NAMES:
            if name not in failures:
                passed_counts[name] += 1
        total += 1
    return {
        name: {
            'passed': passed_counts[name],
            'total': total,
            'rate': passed_counts[name] / total if total else 0.0,
        }
        for name in FILTER_NAMES
    }


def audit_historical_hard_pipeline(full_df, periods=RULE_AUDIT_PERIODS):
    """Measure cumulative survival of actual draws through ordered hard rules."""
    hard_rules = [rule for rule in RED_RULES if rule.hard]
    if periods <= 0 or len(full_df) < 11:
        return {
            'total': 0,
            'passed': 0,
            'rate': 0.0,
            'stages': [
                {'rule': rule.name, 'before': 0, 'removed': 0, 'remaining': 0}
                for rule in hard_rules
            ],
        }

    start = max(10, len(full_df) - periods)
    total = len(full_df) - start
    remaining_counts = Counter()
    for index in range(start, len(full_df)):
        combo = tuple(full_df.iloc[index]['红球'])
        context = historical_rule_context(full_df, index)
        for rule in hard_rules:
            if not rule.evaluator(combo, context):
                break
            remaining_counts[rule.name] += 1

    stages = []
    before = total
    for rule in hard_rules:
        remaining = remaining_counts[rule.name]
        stages.append({
            'rule': rule.name,
            'before': before,
            'removed': before - remaining,
            'remaining': remaining,
        })
        before = remaining
    return {
        'total': total,
        'passed': before,
        'rate': before / total if total else 0.0,
        'stages': stages,
    }

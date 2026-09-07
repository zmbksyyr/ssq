"""Execution and diagnostics for a configured red-ball rule registry."""


def passes_red_filters(combo, context, rule_definitions, rejection_set=None):
    passes_rules = all(
        not rule.hard or rule.evaluator(combo, context)
        for rule in rule_definitions
    )
    return passes_rules and (rejection_set is None or combo not in rejection_set)


def explain_filter_failures(combo, context, rule_definitions, rejection_set=None):
    failures = [
        rule.name for rule in rule_definitions
        if not rule.evaluator(combo, context)
    ]
    if rejection_set is not None and combo in rejection_set:
        failures.append('anti_crowding')
    return failures


def filter_pipeline_stats(combos, context, rule_definitions, rejection_set=None):
    checks = [
        (rule.name, lambda combo, current=rule: current.evaluator(combo, context))
        for rule in rule_definitions if rule.hard
    ]
    checks.append((
        'anti_crowding',
        lambda combo: rejection_set is None or combo not in rejection_set,
    ))
    remaining = list(combos)
    stats = []
    for name, check in checks:
        before = len(remaining)
        remaining = [combo for combo in remaining if check(combo)]
        stats.append({
            'rule': name,
            'before': before,
            'removed': before - len(remaining),
            'remaining': len(remaining),
        })
    return stats

"""Structural and score-budget validation for rule registries."""

from math import isclose, isfinite
from numbers import Real


def validate_signal_weight(signal_weight):
    if (
        isinstance(signal_weight, bool)
        or not isinstance(signal_weight, Real)
        or not isfinite(signal_weight)
        or not 0 <= signal_weight <= 1
    ):
        raise ValueError('基础排名信号权重必须为 0 到 1 之间的有限数值')


def validate_rule_names(rule_definitions):
    names = [rule.name for rule in rule_definitions]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError('规则名称必须为非空字符串')
    if len(names) != len(set(names)):
        raise ValueError('规则名称不能重复')


def validate_rule_definition(rule):
    if not isinstance(rule.hard, bool):
        raise TypeError(f'规则 {rule.name} 的 hard 标记必须为布尔值')
    if not callable(rule.evaluator):
        raise TypeError(f'规则 {rule.name} 的 evaluator 必须可调用')
    weight = rule.score_weight
    if (
        isinstance(weight, bool)
        or not isinstance(weight, Real)
        or not isfinite(weight)
        or weight < 0
    ):
        raise ValueError(f'规则 {rule.name} 的评分权重必须为有限非负数')
    if rule.scorer is None and weight != 0:
        raise ValueError(f'规则 {rule.name} 有评分权重但缺少 scorer')
    if rule.scorer is not None and (not callable(rule.scorer) or weight == 0):
        raise ValueError(f'规则 {rule.name} 的 scorer 与评分权重不一致')
    if not rule.hard and rule.scorer is None:
        raise ValueError(f'软规则 {rule.name} 必须提供 scorer')


def validate_rule_registry(rule_definitions, signal_weight):
    """Validate rule identity, behavior, and the normalized score budget."""
    rule_definitions = tuple(rule_definitions)
    validate_signal_weight(signal_weight)
    validate_rule_names(rule_definitions)
    for rule in rule_definitions:
        validate_rule_definition(rule)

    score_weight_total = sum(
        (rule.score_weight for rule in rule_definitions),
        start=signal_weight,
    )
    if not isclose(score_weight_total, 1.0):
        raise ValueError(f'组合评分总权重必须为 1，实际为 {score_weight_total}')
    return tuple(rule_definitions)

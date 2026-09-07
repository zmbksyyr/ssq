"""Formatting for single-bet and duplex recommendations."""

from itertools import combinations


def format_recommendations_report(data):
    lines = ["\n--- 2. 推荐组合 ---"]
    top_blue = data.recommended_blues[0] if data.recommended_blues else None
    single_combos = data.selection.recommendations if top_blue is not None else ()
    lines.append(f"\n【单式推荐 ({len(single_combos)}组)】")
    if single_combos:
        max_overlap = max(
            (
                len(set(left) & set(right))
                for left, right in combinations(single_combos, 2)
            ),
            default=0,
        )
        lines.append(f'  实际任意两注最大重合红球数: {max_overlap}')
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

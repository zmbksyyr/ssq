"""Evaluation of generated candidates against historical draw results."""

from ssq_candidates import generate_candidates
from ssq_core import PRIZE_RULES
from ssq_selection_models import CandidateGenerationRequest


def evaluate_backtest_mode(
    mode,
    current,
    issue,
    selection_inputs,
    additional_accumulators=(),
):
    """Evaluate one pool mode for one historical issue."""
    selection = generate_candidates(CandidateGenerationRequest(
        red_scores=selection_inputs.red_scores,
        context=selection_inputs.context,
        rejection_set=selection_inputs.rejection_set,
        config=selection_inputs.config,
        mode=mode,
    ))
    red_hits_by_combo = None
    for accumulator in (current, *additional_accumulators):
        red_hits_by_combo = record_backtest_selection(
            accumulator,
            selection,
            issue,
            red_hits_by_combo,
        )
    return selection


def record_backtest_selection(
    current,
    selection,
    issue,
    red_hits_by_combo=None,
):
    """Accumulate one already-generated selection into a result window."""
    current.evaluated_periods += 1
    current.pool_red_hits += len(set(selection.red_pool) & issue.actual_reds)
    current.rank_band_hits.update(issue.rank_band_hits)
    if issue.recommended_blue == issue.actual_blue:
        current.blue_hit_periods += 1

    if not selection.passed_combos:
        return {}

    if red_hits_by_combo is None:
        red_hits_by_combo = {
            combo: len(set(combo) & issue.actual_reds)
            for combo in selection.passed_combos
        }
    current.candidate_tickets += len(selection.passed_combos)
    current.candidate_red_hit_counts.update(red_hits_by_combo.values())
    current.active_periods += 1
    current.tickets += len(selection.recommendations)
    current.cost += len(selection.recommendations) * 2
    blue_hits = int(issue.recommended_blue == issue.actual_blue)
    for combo in selection.recommendations:
        red_hits = red_hits_by_combo[combo]
        current.ticket_red_hit_counts[red_hits] += 1
        hit_key = (red_hits, blue_hits)
        prize = PRIZE_RULES.get(hit_key, 0)
        if prize > 0:
            current.winnings += prize
            current.prize_counts[hit_key] += 1
    return red_hits_by_combo

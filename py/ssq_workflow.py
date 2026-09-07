"""Application workflow for loading data, running analysis, and saving reports."""

import json
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import pandas as pd
from ssq_backtesting import (
    BacktestRequest,
    run_backtest,
)
from ssq_config import (
    COUNTDOWN_SECONDS,
    INTERACTIVE_THRESHOLD,
    LoadedStrategyParams,
    parse_cli_options,
    validate_strategy_params,
)
from ssq_core import (
    atomic_write_text,
    infer_next_issue,
    local_now,
)
from ssq_draw_data import (
    fingerprint_draw_frame,
    normalize_draw_frame,
    validate_draw_dates_not_future,
)
from ssq_modeling import (
    FEATURE_COLUMNS,
    MODEL_TRAINING_PARAMS,
    feature_engineer,
    get_omission,
    run_strategy_and_get_scores,
    train_prediction_models,
    validate_model_sets,
)
from ssq_rank_bands import build_rank_band_labels, build_rank_band_widths
from ssq_reporting import AnalysisReportData, build_analysis_report
from ssq_rule_auditing import (
    audit_historical_hard_pipeline,
    audit_historical_rule_coverage,
)
from ssq_rules import RuleContext, filter_pipeline_stats
from ssq_selection import (
    CandidateGenerationRequest,
    DuplexSelectionRequest,
    generate_candidates,
    make_rejection_set,
    rank_duplex_candidates,
    rejection_seed_for_issue,
)

try:
    import msvcrt
except ImportError:
    import select

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CSV_PATH = os.path.join(PROJECT_ROOT, 'shuangseqiu.csv')
PARAMS_JSON_PATH = os.path.join(PROJECT_ROOT, 'best_params.json')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report')
RUNTIME_PACKAGES = ('numpy', 'pandas', 'lightgbm', 'scikit-learn')


@dataclass(frozen=True)
class PreparedHistory:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    latest_issue: str
    target_issue: int
    sha256: str


@dataclass(frozen=True)
class HistoricalEvaluation:
    loaded_params: LoadedStrategyParams
    rule_coverage: dict
    hard_pipeline_coverage: dict
    backtests: dict[str, Any]
    selected_backtest: Any


@dataclass(frozen=True)
class CurrentSelection:
    red_scores: dict[int, float]
    recommended_blues: list[int]
    rejection_seed: int
    rule_context: RuleContext
    candidate_selection: Any
    pipeline_stats: list[dict]


def collect_runtime_versions():
    return {
        'python': platform.python_version(),
        **{package: version(package) for package in RUNTIME_PACKAGES},
    }


def load_strategy_params(filepath=PARAMS_JSON_PATH):
    try:
        with open(filepath, encoding='utf-8') as handle:
            values = validate_strategy_params(json.load(handle))
    except FileNotFoundError:
        return LoadedStrategyParams(validate_strategy_params({}), False)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f'参数文件 {filepath} 无效: {exc}') from exc
    return LoadedStrategyParams(values, True)


def load_and_preprocess_data(filepath=CSV_PATH):
    """Load, validate, parse, and chronologically order draw history."""
    try:
        frame = pd.read_csv(filepath, header=0)
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        print(f"错误: 无法加载数据文件 '{filepath}': {exc}")
        return None

    try:
        normalized = normalize_draw_frame(frame)
        validate_draw_dates_not_future(normalized)
        return normalized
    except (TypeError, ValueError) as exc:
        print(f"错误: 数据文件 '{filepath}' 校验失败: {exc}")
        return None


def is_confirmation_input(value):
    """Return whether terminal input explicitly confirms the prompt."""
    if isinstance(value, bytes):
        value = value.decode(errors='ignore')
    return value.strip().lower() == 'y'


def get_user_input_with_timeout(timeout):
    """Wait up to ``timeout`` seconds and return whether the user confirmed."""
    prompt = (
        f"\n发现大量高质量组合。输入 'y' 并回车可在 {timeout} 秒内查看全部，"
        '否则将仅输出随机推荐...\n'
    )
    sys.stdout.write(prompt)
    sys.stdout.flush()

    confirmed = False
    if 'msvcrt' in sys.modules:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if msvcrt.kbhit() and is_confirmation_input(msvcrt.getch()):
                confirmed = True
                break
            time.sleep(0.1)
    else:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            confirmed = is_confirmation_input(sys.stdin.readline())

    sys.stdout.write('\n倒计时结束。\n')
    sys.stdout.flush()
    return confirmed


def prepare_history():
    """Load historical draws and derive the immutable inputs for one run."""
    full_df = load_and_preprocess_data()
    if full_df is None or len(full_df) < 50:
        raise SystemExit('错误: 历史数据加载失败或数据量过少（至少需要50期），程序终止。')
    history_sha256 = fingerprint_draw_frame(full_df)
    full_df = feature_engineer(full_df)
    latest_issue = str(full_df.iloc[-1]['期号'])
    try:
        target_issue = infer_next_issue(latest_issue, full_df.iloc[-1]['日期'])
    except (TypeError, ValueError) as exc:
        raise SystemExit(f'错误: 无法推导下一期期号: {exc}') from exc
    return PreparedHistory(
        frame=full_df,
        feature_columns=FEATURE_COLUMNS,
        latest_issue=latest_issue,
        target_issue=target_issue,
        sha256=history_sha256,
    )


def evaluate_history(history, options):
    """Load strategy parameters, audit rules, and run rolling backtests."""
    try:
        loaded_params = load_strategy_params()
    except ValueError as exc:
        raise SystemExit(f'错误: {exc}') from exc
    if not loaded_params.loaded_from_file:
        print(f'警告: 未找到参数文件 {PARAMS_JSON_PATH}，将使用内置的默认参数。')

    rule_coverage = audit_historical_rule_coverage(
        history.frame,
        options.rule_audit_periods,
    )
    hard_pipeline_coverage = audit_historical_hard_pipeline(
        history.frame,
        options.rule_audit_periods,
    )
    backtests = run_backtest(
        history.frame,
        BacktestRequest(
            params=loaded_params.values,
            feature_columns=history.feature_columns,
            num_periods=options.backtest_periods,
            pool_modes=options.backtest_pool_modes,
            config=options.strategy_config,
        ),
    )
    return HistoricalEvaluation(
        loaded_params=loaded_params,
        rule_coverage=rule_coverage,
        hard_pipeline_coverage=hard_pipeline_coverage,
        backtests=backtests,
        selected_backtest=backtests[options.pool_mode],
    )


def train_final_models(history):
    """Train and validate the models used for the target issue."""
    models = train_prediction_models(
        history.frame.iloc[5:].copy(),
        history.feature_columns,
        show_progress=True,
    )
    try:
        validate_model_sets(*models)
    except ValueError as exc:
        raise SystemExit(f'错误: {exc}') from exc
    return models


def select_current_issue(history, options, params, models):
    """Score the target issue and apply anti-crowding and hard rules."""
    red_scores, blue_scores = run_strategy_and_get_scores(
        history.frame,
        params,
        *models,
        history.feature_columns,
    )
    recommended_blues = sorted(
        blue_scores,
        key=blue_scores.get,
        reverse=True,
    )[:options.strategy_config.blue_count]

    print('\n[阶段 5/8] 正在从大底中生成组合并应用硬规则过滤...')
    config = options.strategy_config
    rejection_seed = rejection_seed_for_issue(config.random_seed, history.target_issue)
    rejection_set = make_rejection_set(
        config.rejection_lib_size,
        random.Random(rejection_seed),
    )
    recent_draws = [set(draw) for draw in history.frame.iloc[-10:]['红球'].tolist()]
    context = RuleContext(
        omission_values=get_omission(history.frame),
        recent_draws=recent_draws,
        last_draw=recent_draws[-1],
        previous_draw=recent_draws[-2],
    )
    selection = generate_candidates(CandidateGenerationRequest(
        red_scores=red_scores,
        context=context,
        rejection_set=rejection_set,
        config=config,
        mode=options.pool_mode,
        show_progress=True,
    ))
    print(
        f'已根据ML评分选出 {config.pool_size_red} 个红球大底: '
        f'{list(selection.red_pool)}'
    )
    print(f'过滤完成！共有 {len(selection.passed_combos)} 组号码通过硬规则检验。')
    return CurrentSelection(
        red_scores=red_scores,
        recommended_blues=recommended_blues,
        rejection_seed=rejection_seed,
        rule_context=context,
        candidate_selection=selection,
        pipeline_stats=filter_pipeline_stats(
            selection.potential_combos,
            context,
            rejection_set,
        ),
    )


def display_passed_combinations(passed_combos, non_interactive):
    """Display candidate combinations according to the existing CLI policy."""
    if 0 < len(passed_combos) < INTERACTIVE_THRESHOLD:
        print(
            f'\n通过检验的组合数量为 {len(passed_combos)} '
            f'(低于{INTERACTIVE_THRESHOLD})，全部输出如下：'
        )
        for index, combo in enumerate(passed_combos, 1):
            print(f"  组合 {index:>2}: {' '.join(f'{number:02d}' for number in combo)}")
    elif (
        len(passed_combos) >= INTERACTIVE_THRESHOLD
        and not non_interactive
        and get_user_input_with_timeout(COUNTDOWN_SECONDS)
    ):
        print('\n根据您的确认，输出所有通过检验的组合：')
        for index, combo in enumerate(passed_combos, 1):
            print(f"  组合 {index:>3}: {' '.join(f'{number:02d}' for number in combo)}")


def save_analysis_report(data):
    """Build, print, and atomically persist an analysis report."""
    report = build_analysis_report(data)
    print('\n\n' + report)
    try:
        os.makedirs(REPORT_DIR, exist_ok=True)
        timestamp = data.generated_at.strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(REPORT_DIR, f'ssq_analysis_output_{timestamp}.txt')
        atomic_write_text(filepath, report)
    except OSError as exc:
        raise SystemExit(f'\n\n写入报告文件失败: {exc}') from exc
    print(f'\n\n报告已成功保存到文件: {filepath}')
    return filepath


def run_analysis(options):
    """Execute one complete analysis run and return the saved report path."""
    config = options.strategy_config

    print('=' * 70)
    print('         双色球策略分析器 v7.0')
    print('=' * 70)

    print('\n[阶段 1/8] 正在加载和处理历史数据...')
    history = prepare_history()
    print('数据加载与特征工程完成。')

    print('\n[阶段 2/8] 正在执行历史规则审计与滚动回测...')
    evaluation = evaluate_history(history, options)

    print('\n[阶段 3/8] 正在使用全部历史数据，训练用于最终预测的模型...')
    models = train_final_models(history)

    print('\n[阶段 4/8] 正在为下一期号码进行机器学习评分...')
    current = select_current_issue(
        history,
        options,
        evaluation.loaded_params.values,
        models,
    )

    print('\n[阶段 6/8] 正在整理通过硬规则的候选组合...')
    display_passed_combinations(
        current.candidate_selection.passed_combos,
        options.non_interactive,
    )

    print('\n[阶段 7/8] 正在从最终组合中，生成高重合度的7红球大底...')
    best_7_reds = rank_duplex_candidates(DuplexSelectionRequest(
        passed_combos=current.candidate_selection.passed_combos,
        red_pool=current.candidate_selection.red_pool,
        red_scores=current.red_scores,
        context=current.rule_context,
    ))

    print('\n[阶段 8/8] 正在生成最终推荐报告...')
    generated_at = local_now()
    report_data = AnalysisReportData(
        latest_issue=history.latest_issue,
        target_issue=history.target_issue,
        generated_at=generated_at,
        params_loaded=evaluation.loaded_params.loaded_from_file,
        params=evaluation.loaded_params.values,
        config=config,
        rejection_seed=current.rejection_seed,
        backtest=evaluation.selected_backtest,
        backtests=evaluation.backtests,
        pool_mode=options.pool_mode,
        rank_band_widths=build_rank_band_widths(config),
        rank_band_labels=build_rank_band_labels(config),
        pipeline_stats=current.pipeline_stats,
        rule_coverage=evaluation.rule_coverage,
        hard_pipeline_coverage=evaluation.hard_pipeline_coverage,
        rule_audit_periods=options.rule_audit_periods,
        selection=current.candidate_selection,
        recommended_blues=current.recommended_blues,
        best_7_reds=best_7_reds,
        runtime_versions=collect_runtime_versions(),
        history_sha256=history.sha256,
        model_features=history.feature_columns,
        model_training_params=MODEL_TRAINING_PARAMS,
    )
    return save_analysis_report(report_data)


def main(argv=None):
    """Parse command-line arguments and run the analyzer workflow."""
    return run_analysis(parse_cli_options(argv))

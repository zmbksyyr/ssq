# --- 核心库导入 ---
import json
import os
import random
import sys
import time

import pandas as pd
from ssq_backtesting import (  # noqa: F401 - compatibility exports
    FILTER_NAMES,
    BacktestAccumulator,
    BacktestResult,
    audit_historical_hard_pipeline,
    audit_historical_rule_coverage,
    evaluate_backtest_mode,
    historical_rule_context,
    run_full_backtest,
)
from ssq_config import (  # noqa: F401 - compatibility exports
    BACKTEST_PERIODS,
    COUNTDOWN_SECONDS,
    DEFAULT_PARAMS,
    DEFAULT_STRATEGY_CONFIG,
    INTERACTIVE_THRESHOLD,
    MAX_SHARED_RED_BALLS,
    NUM_BLUE_BALLS,
    NUM_RECOMMENDATIONS,
    POOL_SIZE_RED,
    RANDOM_SEED,
    RED_HIGH_COUNT,
    RED_LOW_COUNT,
    RED_POOL_MODES,
    REJECTION_LIB_SIZE,
    REJECTION_SEED_MULTIPLIER,
    RULE_AUDIT_PERIODS,
    TOTAL_RED_COMBINATIONS,
    AnalyzerOptions,
    LoadedStrategyParams,
    StrategyConfig,
    build_argument_parser,
    parse_cli_options,
    validate_strategy_params,
)
from ssq_core import (
    atomic_write_text,
    infer_next_issue,
    local_now,
    parse_blue_ball,
    parse_issue,
    parse_red_balls,
)
from ssq_modeling import (  # noqa: F401 - compatibility exports
    apply_red_score_adjustments,
    feature_engineer,
    get_omission,
    get_weighted_frequency,
    predict_positive_probability,
    run_strategy_and_get_scores,
    train_ball_models,
    train_prediction_models,
    validate_model_sets,
)
from ssq_reporting import AnalysisReportData, build_analysis_report
from ssq_rules import RuleContext, filter_pipeline_stats
from ssq_selection import (  # noqa: F401 - compatibility exports
    RANK_BAND_WIDTHS,
    RANK_BANDS,
    RedCandidateSelection,
    build_red_pool,
    count_actual_reds_by_rank_band,
    find_best_7_red_combinations,
    generate_red_candidates,
    make_rejection_set,
    passes_red_filters,
    rejection_seed_for_issue,
    select_recommendations,
)

# --- 平台特定模块导入, 用于实现非阻塞的键盘输入监听 ---
try:
    # 尝试导入 msvcrt, 这是 Windows 平台专用的库
    import msvcrt
except ImportError:
    # 如果导入失败 (说明不是 Windows 平台), 则导入 select, 适用于 Linux/Mac
    import select

# --- 文件路径设置 ---
# 获取当前脚本文件所在的目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 假设脚本在 'py' 这样的子目录中, 数据和报告文件都位于其上一级目录
root_dir = os.path.dirname(script_dir) 
# 构造历史数据 CSV 文件的绝对路径 (例如: .../ssq/shuangseqiu.csv)
CSV_PATH = os.path.join(root_dir, 'shuangseqiu.csv')
# 构造机器学习优化参数 JSON 文件的绝对路径 (例如: .../ssq/best_params.json)
PARAMS_JSON_PATH = os.path.join(root_dir, 'best_params.json')
# 构造报告输出目录的绝对路径 (例如: .../ssq/report/)
REPORT_DIR = os.path.join(root_dir, 'report')

# --- 2. 辅助函数库 ---


def load_strategy_params(filepath=PARAMS_JSON_PATH):
    try:
        with open(filepath, encoding='utf-8') as handle:
            values = validate_strategy_params(json.load(handle))
    except FileNotFoundError:
        return LoadedStrategyParams(validate_strategy_params({}), False)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"参数文件 {filepath} 无效: {exc}") from exc
    return LoadedStrategyParams(values, True)


def load_and_preprocess_data(filepath=CSV_PATH):
    """
    从CSV文件加载并预处理双色球历史数据。

    Args:
        filepath (str): CSV文件的完整路径。

    Returns:
        DataFrame: 处理好并按期号升序排列的数据。如果加载或处理失败，返回None。
    """
    try:
        # 尝试使用 pandas 读取 CSV 文件，假设第一行是表头
        df = pd.read_csv(filepath, header=0)
        # 为了代码的健壮性，强制重命名列
        df.columns = ['期号', '日期', '红球', '蓝球']
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as e:
        # 如果文件读取失败（例如文件不存在、格式错误），打印错误信息并中断程序
        print(f"错误: 无法加载数据文件 '{filepath}': {e}")
        return None
    
    try:
        df['期号'] = df['期号'].apply(parse_issue)
        if df['期号'].duplicated().any():
            raise ValueError("存在重复期号")
        df['_parsed_date'] = pd.to_datetime(
            df['日期'], format='%Y-%m-%d', errors='raise'
        )
        df['红球'] = df['红球'].apply(parse_red_balls)
        df['蓝球'] = df['蓝球'].apply(parse_blue_ball)
    except (TypeError, ValueError) as exc:
        print(f"错误: 数据文件 '{filepath}' 校验失败: {exc}")
        return None

    df = df.sort_values('期号').reset_index(drop=True)
    if ((df['期号'] // 1000) != df['_parsed_date'].dt.year).any():
        print(f"错误: 数据文件 '{filepath}' 校验失败: 期号年份与开奖日期不一致")
        return None
    if not df['_parsed_date'].is_monotonic_increasing:
        print(f"错误: 数据文件 '{filepath}' 校验失败: 期号与开奖日期顺序不一致")
        return None
    return df.drop(columns=['_parsed_date'])

# --- 3. 交互式输入模块 ---


def is_confirmation_input(value):
    """Return whether terminal input explicitly confirms the prompt."""
    if isinstance(value, bytes):
        value = value.decode(errors='ignore')
    return value.strip().lower() == 'y'


def get_user_input_with_timeout(timeout):
    """Wait up to ``timeout`` seconds and return whether the user confirmed."""
    prompt = f"\n发现大量高质量组合。输入 'y' 并回车可在 {timeout} 秒内查看全部，否则将仅输出随机推荐...\n"
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
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            confirmed = is_confirmation_input(sys.stdin.readline())

    sys.stdout.write("\n倒计时结束。\n")
    sys.stdout.flush()
    return confirmed

# --- 5. 主执行逻辑 ---
if __name__ == '__main__':
    options = parse_cli_options()
    config = options.strategy_config

    print("="*70)
    print("         双色球策略分析器 v7.0")
    print("="*70)

    # --- [阶段 1/8] 加载与特征工程 ---
    print("\n[阶段 1/8] 正在加载和处理历史数据...")
    full_df = load_and_preprocess_data()
    if full_df is None or len(full_df) < 50:
        raise SystemExit("错误: 历史数据加载失败或数据量过少（至少需要50期），程序终止。")
    full_df = feature_engineer(full_df)
    feature_columns = [col for col in full_df.columns if col not in ['期号', '日期', '红球', '蓝球']]
    rule_coverage = audit_historical_rule_coverage(
        full_df, options.rule_audit_periods
    )
    hard_pipeline_coverage = audit_historical_hard_pipeline(
        full_df, options.rule_audit_periods
    )
    latest_issue = str(full_df.iloc[-1]['期号'])
    try:
        target_issue = infer_next_issue(latest_issue, full_df.iloc[-1]['日期'])
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"错误: 无法推导下一期期号: {exc}")
    print("数据加载与特征工程完成。")

    # --- [阶段 2/8] 执行严谨的历史回测 ---
    try:
        loaded_params = load_strategy_params()
    except ValueError as exc:
        raise SystemExit(f"错误: {exc}") from exc
    params = loaded_params.values
    params_loaded = loaded_params.loaded_from_file
    if not params_loaded:
        print(f"警告: 未找到参数文件 {PARAMS_JSON_PATH}，将使用内置的默认参数。")
    # 执行回测并捕获其返回的统计结果
    backtests = run_full_backtest(
        full_df, params, feature_columns, options.backtest_periods,
        pool_modes=options.backtest_pool_modes, config=config,
    )
    backtest = backtests[options.pool_mode]
    
    # --- [阶段 3/8] 训练最终预测模型 ---
    print("\n[阶段 3/8] 正在使用全部历史数据，训练用于最终预测的模型...")
    ml_training_df = full_df.iloc[5:].copy()
    final_ml_models_red, final_ml_models_blue = train_prediction_models(
        ml_training_df, feature_columns, show_progress=True
    )
    try:
        validate_model_sets(final_ml_models_red, final_ml_models_blue)
    except ValueError as exc:
        raise SystemExit(f"错误: {exc}")
    
    # --- [阶段 4/8] 执行对下一期的预测 ---
    print("\n[阶段 4/8] 正在为下一期号码进行机器学习评分...")
    red_scores, blue_scores = run_strategy_and_get_scores(
        full_df, params, final_ml_models_red, final_ml_models_blue,
        feature_columns,
    )
    recommended_blues = sorted(
        blue_scores, key=blue_scores.get, reverse=True
    )[:config.blue_count]

    # --- [阶段 5/8] 规则过滤 ---
    print("\n[阶段 5/8] 正在从大底中生成组合并应用硬规则过滤...")
    
    # 同一期可复现，不同期变化；与滚动回测采用完全相同的派生规则。
    rejection_seed = rejection_seed_for_issue(config.random_seed, target_issue)
    rejection_set = make_rejection_set(
        config.rejection_lib_size, random.Random(rejection_seed)
    )

    # 提前计算过滤所需的历史数据
    omission_values = get_omission(full_df)
    last_10_draws_sets = [set(d) for d in full_df.iloc[-10:]['红球'].tolist()]
    last_draw_set = last_10_draws_sets[-1]
    last_2_draw_set = last_10_draws_sets[-2]
    context = RuleContext(
        omission_values=omission_values,
        recent_draws=last_10_draws_sets,
        last_draw=last_draw_set,
        previous_draw=last_2_draw_set,
    )
    
    selection = generate_red_candidates(
        red_scores,
        context,
        rejection_set,
        config=config,
        mode=options.pool_mode,
        show_progress=True,
    )
    red_pool = selection.red_pool
    potential_combos = selection.potential_combos
    passed_combos_tuples = selection.passed_combos

    print(f"已根据ML评分选出 {config.pool_size_red} 个红球大底: {list(red_pool)}")
    print(f"过滤完成！共有 {len(passed_combos_tuples)} 组号码通过硬规则检验。")
    pipeline_stats = filter_pipeline_stats(
        potential_combos, context, rejection_set
    )

    # --- [阶段 6/8] 交互式输出 ---
    if 0 < len(passed_combos_tuples) < INTERACTIVE_THRESHOLD:
        print(f"\n通过检验的组合数量为 {len(passed_combos_tuples)} (低于{INTERACTIVE_THRESHOLD})，全部输出如下：")
        for i, combo in enumerate(passed_combos_tuples, 1): 
            print(f"  组合 {i:>2}: {' '.join(f'{n:02d}' for n in combo)}")
    elif (
        len(passed_combos_tuples) >= INTERACTIVE_THRESHOLD
        and not options.non_interactive
    ):
        if get_user_input_with_timeout(COUNTDOWN_SECONDS):
            print("\n根据您的确认，输出所有通过检验的组合：")
            for i, combo in enumerate(passed_combos_tuples, 1): 
                print(f"  组合 {i:>3}: {' '.join(f'{n:02d}' for n in combo)}")

    # --- [阶段 7/8] 高级推荐 ---
    print("\n[阶段 7/8] 正在从最终组合中，生成高重合度的7红球大底...")
    best_7_reds = find_best_7_red_combinations(
        passed_combos_tuples, red_pool, red_scores, last_draw_set, last_2_draw_set
    )

    # --- [阶段 8/8] 最终报告 ---
    print("\n--- 正在生成最终推荐报告 ---")
    report_data = AnalysisReportData(
        latest_issue=latest_issue,
        target_issue=target_issue,
        generated_at=local_now(),
        params_loaded=params_loaded,
        params=params,
        config=config,
        rejection_seed=rejection_seed,
        backtest=backtest,
        backtests=backtests,
        pool_mode=options.pool_mode,
        rank_band_widths=RANK_BAND_WIDTHS,
        pipeline_stats=pipeline_stats,
        rule_coverage=rule_coverage,
        hard_pipeline_coverage=hard_pipeline_coverage,
        rule_audit_periods=options.rule_audit_periods,
        selection=selection,
        recommended_blues=recommended_blues,
        best_7_reds=best_7_reds,
    )
    final_report_string = build_analysis_report(report_data)
    print("\n\n" + final_report_string)

    try:
        os.makedirs(REPORT_DIR, exist_ok=True)
        timestamp = local_now().strftime("%Y%m%d_%H%M%S")
        filename = f"ssq_analysis_output_{timestamp}.txt"
        filepath = os.path.join(REPORT_DIR, filename)
        atomic_write_text(filepath, final_report_string)
        print(f"\n\n报告已成功保存到文件: {filepath}")
    except OSError as e:
        raise SystemExit(f"\n\n写入报告文件失败: {e}")

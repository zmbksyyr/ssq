"""Command-line entry point and compatibility facade for prize checking."""

from ssq_bonus_reporting import build_bonus_report  # noqa: F401
from ssq_bonus_workflow import (  # noqa: F401
    CSV_PATH,
    PROJECT_ROOT,
    REPORT_DIR,
    find_matching_report,
    load_latest_draw,
    main,
    run_bonus_check,
)
from ssq_prizes import (  # noqa: F401
    DUPLEX_HEADER_PATTERN,
    SINGLE_BET_PATTERN,
    SINGLE_HEADER_PATTERN,
    calculate_duplex_prize,
    calculate_single_prize,
    parse_duplex_section,
    parse_report_bets,
    parse_single_bet_line,
    validate_parsed_bets,
)

if __name__ == '__main__':
    main()

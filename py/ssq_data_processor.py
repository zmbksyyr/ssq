"""Command-line entry point and compatibility facade for draw updates."""

from ssq_csv_store import atomic_write_csv, read_existing_csv  # noqa: F401
from ssq_data_sources import (  # noqa: F401
    HTML_DATA_URL,
    REQUEST_HEADERS,
    REQUEST_TIMEOUT_SECONDS,
    TXT_DATA_URL,
    create_http_session,
    cross_check_sources,
    fetch_full_data_from_txt,
    fetch_latest_data_from_html,
    find_secondary_only_issues,
    parse_txt_data,
)
from ssq_data_workflow import (  # noqa: F401
    CSV_FILE_PATH,
    PROJECT_ROOT,
    configure_logging,
    main,
    run_data_update,
)
from ssq_history_store import update_csv_file  # noqa: F401
from ssq_snapshot_validation import (  # noqa: F401
    MIN_FULL_SNAPSHOT_RECORDS,
    normalize_lottery_frame,
    validate_authoritative_snapshot,
)

if __name__ == '__main__':
    main()

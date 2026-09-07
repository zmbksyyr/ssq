"""Command-line entry point and compatibility facade for draw updates."""

from ssq_data_sources import (  # noqa: F401
    HTML_DATA_URL,
    REQUEST_HEADERS,
    REQUEST_TIMEOUT_SECONDS,
    TXT_DATA_URL,
    create_http_session,
    cross_check_sources,
    fetch_full_data_from_txt,
    fetch_latest_data_from_html,
    parse_txt_data,
)
from ssq_data_store import (  # noqa: F401
    MIN_FULL_SNAPSHOT_RECORDS,
    atomic_write_csv,
    normalize_lottery_frame,
    read_existing_csv,
    update_csv_file,
    validate_authoritative_snapshot,
)
from ssq_data_workflow import (  # noqa: F401
    CSV_FILE_PATH,
    PROJECT_ROOT,
    configure_logging,
    main,
    run_data_update,
)

if __name__ == '__main__':
    main()

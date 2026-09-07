"""Validation rules for authoritative draw-history snapshots."""

from ssq_draw_data import serialize_draw_frame, validate_draw_dates_not_future

MIN_FULL_SNAPSHOT_RECORDS = 100


def normalize_lottery_frame(frame):
    """Validate and normalize a draw DataFrame before it reaches the CSV."""
    return serialize_draw_frame(frame)


def validate_authoritative_snapshot(new_data, existing_data, today=None):
    """Ensure the advertised full snapshot cannot truncate local history."""
    if len(new_data) < MIN_FULL_SNAPSHOT_RECORDS:
        raise ValueError(
            f'Authoritative snapshot has only {len(new_data)} records; '
            f'at least {MIN_FULL_SNAPSHOT_RECORDS} are required.'
        )
    validate_draw_dates_not_future(new_data, today=today)
    if existing_data.empty:
        return
    missing_issues = sorted(set(existing_data['期号']) - set(new_data['期号']))
    if missing_issues:
        preview = ', '.join(str(issue) for issue in missing_issues[:5])
        suffix = ' ...' if len(missing_issues) > 5 else ''
        raise ValueError(
            f'Authoritative snapshot is missing local issues: {preview}{suffix}'
        )

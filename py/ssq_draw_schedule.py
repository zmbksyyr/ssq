"""Local time and issue sequencing for Double Color Ball draws."""

from datetime import date, datetime, timedelta

from ssq_domain import DRAW_WEEKDAYS, LOCAL_TIMEZONE
from ssq_parsing import parse_issue


def local_now():
    return datetime.now(LOCAL_TIMEZONE)


def local_today():
    return local_now().date()


def infer_next_issue(current_issue, current_draw_date):
    """Infer the next regular draw issue, including the year boundary."""
    issue = parse_issue(current_issue)
    issue_year, _ = divmod(issue, 1000)

    if isinstance(current_draw_date, datetime):
        draw_date = current_draw_date.date()
    elif isinstance(current_draw_date, date):
        draw_date = current_draw_date
    else:
        draw_date = date.fromisoformat(str(current_draw_date))
    if draw_date.year != issue_year:
        raise ValueError(f"期号年份 {issue_year} 与开奖日期 {draw_date} 不一致")

    next_draw_date = draw_date + timedelta(days=1)
    while next_draw_date.weekday() not in DRAW_WEEKDAYS:
        next_draw_date += timedelta(days=1)
    if next_draw_date.year != issue_year:
        return next_draw_date.year * 1000 + 1
    return issue + 1

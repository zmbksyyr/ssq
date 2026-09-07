"""Pure parsing and normalization for remote lottery source formats."""

import logging
from datetime import date

from bs4 import BeautifulSoup
from ssq_parsing import parse_blue_ball, parse_issue, parse_red_balls

logger = logging.getLogger('ssq_data_processor')


def parse_html_data(content):
    """Parse number-only comparison records from the source HTML."""
    soup = BeautifulSoup(content, 'html.parser')
    table = soup.find('table')
    if not table:
        logger.warning('在网页中未能找到数据表格。')
        return []

    records = []
    for row in table.find_all('tr')[1:]:
        cells = row.find_all('td')
        if len(cells) < 3:
            continue
        try:
            issue = parse_issue(cells[0].get_text(strip=True).removesuffix('期'))
            red_numbers = parse_red_balls(
                cells[1].get_text(' ', strip=True).split()
            )
            blue_number = parse_blue_ball(cells[2].get_text(strip=True))
            records.append({
                '期号': str(issue),
                '红球': ','.join(f'{number:02d}' for number in red_numbers),
                '蓝球': f'{blue_number:02d}',
            })
        except (TypeError, ValueError, IndexError) as exc:
            logger.warning(
                f'解析表格行时出错: {row.text.strip()}. 错误: {exc}. 跳过此行。'
            )
    return records


def parse_txt_data(data_lines):
    """Parse authoritative source lines into normalized CSV field values."""
    if not data_lines:
        return []
    logger.info('正在解析TXT数据...')
    parsed_data = []
    for line in data_lines:
        fields = line.strip().split()
        if len(fields) < 9:
            continue
        try:
            issue = parse_issue(fields[0])
            date_text = fields[1]
            red_numbers = parse_red_balls(','.join(fields[2:8]))
            blue_number = parse_blue_ball(fields[8])
            parsed_date = date.fromisoformat(date_text)
            if parsed_date.isoformat() != date_text:
                raise ValueError('日期必须使用 YYYY-MM-DD 格式')
            parsed_data.append([
                str(issue),
                date_text,
                ','.join(f'{number:02d}' for number in red_numbers),
                f'{blue_number:02d}',
            ])
        except (IndexError, TypeError, ValueError) as exc:
            logger.warning(f'解析TXT行失败: {line}. 错误: {exc}')
    logger.info(f'从TXT数据中成功解析出 {len(parsed_data)} 条有效记录。')
    return parsed_data

"""Loading and validation of persisted strategy parameters."""

import json

from ssq_config import LoadedStrategyParams, validate_strategy_params


def load_strategy_params(filepath):
    try:
        with open(filepath, encoding='utf-8') as handle:
            values = validate_strategy_params(json.load(handle))
    except FileNotFoundError:
        return LoadedStrategyParams(validate_strategy_params({}), False)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f'参数文件 {filepath} 无效: {exc}') from exc
    return LoadedStrategyParams(values, True)

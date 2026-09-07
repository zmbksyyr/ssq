import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import ssq_bonus_workflow as workflow
import ssq_latest_draw as latest_draw


class LatestDrawTests(unittest.TestCase):
    def test_workflow_wrapper_delegates_its_validation_dependencies(self):
        with patch.object(
            latest_draw,
            'load_latest_draw',
            return_value={'issue': 2026001},
        ) as load:
            result = workflow.load_latest_draw('draws.csv')

        self.assertEqual(result, {'issue': 2026001})
        load.assert_called_once_with(
            'draws.csv',
            normalize=workflow.normalize_draw_frame,
            validate_dates=workflow.validate_draw_dates_not_future,
        )

    def test_loader_normalizes_and_validates_before_selecting_latest(self):
        normalized = pd.DataFrame([{
            '期号': 2026002,
            '日期': '2026-01-04',
            '红球': [7, 8, 9, 10, 11, 12],
            '蓝球': 16,
        }])
        normalize = Mock(return_value=normalized)
        validate_dates = Mock()

        with patch.object(pd, 'read_csv', return_value=pd.DataFrame()) as read:
            result = latest_draw.load_latest_draw(
                'draws.csv',
                normalize=normalize,
                validate_dates=validate_dates,
            )

        read.assert_called_once_with('draws.csv', header=0)
        normalize.assert_called_once_with(read.return_value)
        validate_dates.assert_called_once_with(normalized)
        self.assertEqual(result, {
            'issue': 2026002,
            'red': {7, 8, 9, 10, 11, 12},
            'blue': 16,
        })

    def test_loader_rejects_empty_normalized_history(self):
        with (
            patch.object(pd, 'read_csv', return_value=pd.DataFrame()),
            self.assertRaisesRegex(ValueError, '开奖数据为空'),
        ):
            latest_draw.load_latest_draw(
                'draws.csv',
                normalize=Mock(return_value=pd.DataFrame()),
                validate_dates=Mock(),
            )


if __name__ == '__main__':
    unittest.main()

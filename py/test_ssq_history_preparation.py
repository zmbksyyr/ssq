import unittest
from datetime import date
from unittest.mock import Mock

import pandas as pd
import ssq_history_preparation as preparation
import ssq_workflow as workflow


class HistoryPreparationTests(unittest.TestCase):
    def test_workflow_loader_delegates_explicit_path(self):
        load = Mock(return_value='frame')
        with unittest.mock.patch.object(
            preparation,
            'load_and_preprocess_data',
            load,
        ):
            result = workflow.load_and_preprocess_data('history.csv')

        self.assertEqual(result, 'frame')
        load.assert_called_once_with(
            'history.csv',
            workflow.normalize_draw_frame,
            workflow.validate_draw_dates_not_future,
        )

    def test_preparation_rejects_short_history_before_feature_work(self):
        engineer = Mock()
        dependencies = preparation.HistoryPreparationDependencies(
            load_data=Mock(return_value=pd.DataFrame(index=range(49))),
            fingerprint_history=Mock(),
            engineer_features=engineer,
            infer_target_issue=Mock(),
        )

        with self.assertRaisesRegex(SystemExit, '至少需要50期'):
            preparation.prepare_history('history.csv', dependencies)

        engineer.assert_not_called()

    def test_preparation_builds_reproducible_history_identity(self):
        frame = pd.DataFrame({
            '期号': list(range(2026001, 2026051)),
            '日期': [date(2026, 1, 1) for _ in range(50)],
        })
        engineered = frame.assign(red_sum=100)
        dependencies = preparation.HistoryPreparationDependencies(
            load_data=Mock(return_value=frame),
            fingerprint_history=Mock(return_value='a' * 64),
            engineer_features=Mock(return_value=engineered),
            infer_target_issue=Mock(return_value=2026052),
        )

        history = preparation.prepare_history('history.csv', dependencies)

        self.assertIs(history.frame, engineered)
        self.assertEqual(history.latest_issue, '2026050')
        self.assertEqual(history.target_issue, 2026052)
        self.assertEqual(history.sha256, 'a' * 64)
        dependencies.infer_target_issue.assert_called_once_with(
            '2026050',
            date(2026, 1, 1),
        )


if __name__ == '__main__':
    unittest.main()

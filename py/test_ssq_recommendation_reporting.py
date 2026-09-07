import unittest
from types import SimpleNamespace

import ssq_recommendation_reporting as recommendation_reporting
import ssq_reporting as reporting
from ssq_selection_models import RedCandidateSelection


class RecommendationReportingTests(unittest.TestCase):
    def test_reporting_preserves_recommendation_format_compatibility_export(self):
        self.assertIs(
            reporting.format_recommendations_report,
            recommendation_reporting.format_recommendations_report,
        )

    def test_report_formats_overlap_blue_and_duplex_recommendations(self):
        selection = RedCandidateSelection(
            (),
            (),
            ((1, 2, 3, 4, 5, 6), (1, 2, 7, 8, 9, 10)),
            ((1, 2, 3, 4, 5, 6), (1, 2, 7, 8, 9, 10)),
        )
        data = SimpleNamespace(
            selection=selection,
            recommended_blues=[6, 12],
            best_7_reds=[((1, 2, 3, 4, 5, 6, 7), 7)],
        )

        report = '\n'.join(
            recommendation_reporting.format_recommendations_report(data)
        )

        self.assertIn('【单式推荐 (2组)】', report)
        self.assertIn('实际任意两注最大重合红球数: 2', report)
        self.assertIn('蓝球 [06]', report)
        self.assertIn('蓝球: [6, 12]', report)

    def test_report_suppresses_single_bets_without_a_blue(self):
        selection = RedCandidateSelection(
            (), (), ((1, 2, 3, 4, 5, 6),), ((1, 2, 3, 4, 5, 6),)
        )
        data = SimpleNamespace(
            selection=selection,
            recommended_blues=[],
            best_7_reds=[],
        )

        report = '\n'.join(
            recommendation_reporting.format_recommendations_report(data)
        )

        self.assertIn('【单式推荐 (0组)】', report)
        self.assertIn('未能生成足够的单式组合', report)
        self.assertIn('未能生成足够的复式组合', report)
        self.assertNotIn('组合  1:', report)


if __name__ == '__main__':
    unittest.main()

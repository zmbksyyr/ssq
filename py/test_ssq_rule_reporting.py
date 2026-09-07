import unittest

import ssq_reporting as reporting
import ssq_rule_reporting as rule_reporting


class RuleReportingTests(unittest.TestCase):
    def test_reporting_preserves_rule_format_compatibility_exports(self):
        self.assertIs(
            reporting.format_audit_window,
            rule_reporting.format_audit_window,
        )
        self.assertIs(
            reporting.format_rule_audit_report,
            rule_reporting.format_rule_audit_report,
        )

    def test_audit_window_reports_requested_and_actual_periods(self):
        self.assertEqual(rule_reporting.format_audit_window(20, 20), '最近 20 期')
        self.assertEqual(
            rule_reporting.format_audit_window(12, 20),
            '实际 12 期，请求 20 期',
        )


if __name__ == '__main__':
    unittest.main()

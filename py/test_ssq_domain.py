import unittest

import ssq_core as core
import ssq_domain as domain


class DomainTests(unittest.TestCase):
    def test_core_preserves_domain_constant_exports(self):
        for name in (
            'RED_BALLS',
            'BLUE_BALLS',
            'PRIME_RED_BALLS',
            'DRAW_COLUMNS',
            'DRAW_WEEKDAYS',
            'LOCAL_TIMEZONE',
            'PRIZE_RULES',
            'PRIZE_NAMES',
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(core, name), getattr(domain, name))

    def test_lottery_domains_and_schedule_are_complete(self):
        self.assertEqual(domain.RED_BALLS, tuple(range(1, 34)))
        self.assertEqual(domain.BLUE_BALLS, tuple(range(1, 17)))
        self.assertEqual(domain.DRAW_WEEKDAYS, {1, 3, 6})
        self.assertEqual(domain.PRIZE_RULES[(6, 1)], 5_000_000)


if __name__ == '__main__':
    unittest.main()

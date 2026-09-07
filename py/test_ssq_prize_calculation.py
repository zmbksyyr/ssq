import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ssq_prize_calculation as calculation
import ssq_prizes as prizes
from ssq_domain import PRIZE_NAMES, PRIZE_RULES


class PrizeCalculationTests(unittest.TestCase):
    def test_runtime_modules_do_not_depend_on_prizes_compatibility_facade(self):
        module_dir = Path(__file__).parent
        offenders = []
        for path in module_dir.glob('ssq_*.py'):
            if path.name == 'ssq_prizes.py':
                continue
            tree = ast.parse(path.read_text(encoding='utf-8'))
            if any(
                isinstance(node, (ast.Import, ast.ImportFrom))
                and (
                    getattr(node, 'module', None) == 'ssq_prizes'
                    or any(alias.name == 'ssq_prizes' for alias in node.names)
                )
                for node in ast.walk(tree)
            ):
                offenders.append(path.name)

        self.assertEqual(offenders, [])

    def test_prizes_preserves_calculator_compatibility_exports(self):
        self.assertIs(
            prizes.calculate_single_prize,
            calculation.calculate_single_prize,
        )
        self.assertIs(
            prizes.calculate_duplex_prize,
            calculation.calculate_duplex_prize,
        )

    def test_single_ticket_covers_every_prize_rule_and_no_prize(self):
        winning_reds = [1, 2, 3, 4, 5, 6]
        losing_reds = [7, 8, 9, 10, 11, 12]
        cases = [*PRIZE_RULES.items(), ((3, 0), 0)]

        for (red_hits, blue_hit), expected_prize in cases:
            with self.subTest(red_hits=red_hits, blue_hit=blue_hit):
                bet_reds = winning_reds[:red_hits] + losing_reds[:6 - red_hits]
                prize, name, summary = calculation.calculate_single_prize(
                    bet_reds,
                    7 if blue_hit else 8,
                    winning_reds,
                    7,
                )

                self.assertEqual(prize, expected_prize)
                self.assertEqual(
                    name,
                    PRIZE_NAMES.get((red_hits, blue_hit), '未中奖'),
                )
                self.assertEqual(summary, f'命中{red_hits}+{blue_hit}')

    def test_duplex_counts_every_contained_standard_ticket(self):
        total, breakdown, summary = calculation.calculate_duplex_prize(
            [1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3],
            [1, 2, 3, 4, 5, 6],
            1,
        )

        self.assertEqual(sum(breakdown.values()), 21)
        self.assertEqual(total, 5_220_400)
        self.assertEqual(summary, '总计命中 6 个红球, 1 个蓝球')


if __name__ == '__main__':
    unittest.main()

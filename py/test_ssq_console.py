import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import ssq_console as console


class ConsoleDisplayTests(unittest.TestCase):
    def test_small_candidate_set_is_displayed_without_prompt(self):
        combinations = ((1, 2, 3, 4, 5, 6),)
        output = io.StringIO()

        with (
            redirect_stdout(output),
            patch.object(console, 'get_user_input_with_timeout') as prompt,
        ):
            console.display_passed_combinations(combinations, False)

        prompt.assert_not_called()
        self.assertIn('组合数量为 1', output.getvalue())
        self.assertIn('01 02 03 04 05 06', output.getvalue())

    def test_large_candidate_set_is_silent_in_non_interactive_mode(self):
        combinations = ((1, 2, 3, 4, 5, 6),) * 100
        output = io.StringIO()

        with (
            redirect_stdout(output),
            patch.object(console, 'get_user_input_with_timeout') as prompt,
        ):
            console.display_passed_combinations(combinations, True)

        prompt.assert_not_called()
        self.assertEqual(output.getvalue(), '')

    def test_large_candidate_set_is_displayed_after_confirmation(self):
        combinations = ((1, 2, 3, 4, 5, 6),) * 100
        output = io.StringIO()

        with (
            redirect_stdout(output),
            patch.object(
                console,
                'get_user_input_with_timeout',
                return_value=True,
            ) as prompt,
        ):
            console.display_passed_combinations(combinations, False)

        prompt.assert_called_once_with(console.COUNTDOWN_SECONDS)
        self.assertIn('输出所有通过检验的组合', output.getvalue())
        self.assertEqual(output.getvalue().count('01 02 03 04 05 06'), 100)


if __name__ == '__main__':
    unittest.main()

"""Cross-platform terminal interaction for the analysis workflow."""

import sys
import time

from ssq_config import COUNTDOWN_SECONDS, INTERACTIVE_THRESHOLD

try:
    import msvcrt
except ImportError:
    import select


def is_confirmation_input(value):
    """Return whether terminal input explicitly confirms the prompt."""
    if isinstance(value, bytes):
        value = value.decode(errors='ignore')
    return value.strip().lower() == 'y'


def get_user_input_with_timeout(timeout):
    """Wait up to ``timeout`` seconds and return whether the user confirmed."""
    prompt = (
        f"\n发现大量高质量组合。输入 'y' 并回车可在 {timeout} 秒内查看全部，"
        '否则将仅输出随机推荐...\n'
    )
    sys.stdout.write(prompt)
    sys.stdout.flush()

    confirmed = False
    if 'msvcrt' in sys.modules:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if msvcrt.kbhit() and is_confirmation_input(msvcrt.getch()):
                confirmed = True
                break
            time.sleep(0.1)
    else:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            confirmed = is_confirmation_input(sys.stdin.readline())

    sys.stdout.write('\n倒计时结束。\n')
    sys.stdout.flush()
    return confirmed


def display_passed_combinations(passed_combos, non_interactive):
    """Display candidate combinations according to the existing CLI policy."""
    if 0 < len(passed_combos) < INTERACTIVE_THRESHOLD:
        print(
            f'\n通过检验的组合数量为 {len(passed_combos)} '
            f'(低于{INTERACTIVE_THRESHOLD})，全部输出如下：'
        )
        for index, combo in enumerate(passed_combos, 1):
            print(f"  组合 {index:>2}: {' '.join(f'{number:02d}' for number in combo)}")
    elif (
        len(passed_combos) >= INTERACTIVE_THRESHOLD
        and not non_interactive
        and get_user_input_with_timeout(COUNTDOWN_SECONDS)
    ):
        print('\n根据您的确认，输出所有通过检验的组合：')
        for index, combo in enumerate(passed_combos, 1):
            print(f"  组合 {index:>3}: {' '.join(f'{number:02d}' for number in combo)}")

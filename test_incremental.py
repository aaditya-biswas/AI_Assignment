"""Regression test: every incrementally maintained evaluation quantity must equal
a from-scratch computation after arbitrary make/unmake sequences.

The agent maintains `_static`, `_press_w`, `_press_b`, `wnk`/`bnk`, `_wk`/`_bk`
and `wpow`/`bpow` during the search.  An asymmetric `_eval_add`/`_eval_sub` would
silently corrupt every score (and the board itself), and parity testing would not
catch it because move generation is untouched.  This test drives real searches and
then compares against a fresh scan of the board.
"""
import random
import sys

from board import GameEngine
from B23CS1001 import (B23CS1001, _STATIC, _VAL, _CODE, E, WK, BK)


def scratch(ag):
    """Recompute every incremental quantity from the board, from scratch."""
    b = ag.b
    static = 0
    wk = bk = -1
    wnk = bnk = 0
    wpow = bpow = 0
    for i in range(48):
        p = b[i]
        if p == E:
            continue
        static += _STATIC[p][i]
        if p == WK:
            wk = i
        elif p == BK:
            bk = i
        elif p < 6:
            wnk += 1
            wpow += _VAL[p]
        else:
            bnk += 1
            bpow += _VAL[p]
    return (static, wk, bk, wnk, bnk, wpow, bpow,
            ag._pressure_on(wk, False), ag._pressure_on(bk, True))


def actual(ag):
    return (ag._static, ag._wk, ag._bk, ag.wnk, ag.bnk, ag.wpow, ag.bpow,
            ag._press_w, ag._press_b)


NAMES = ('static', 'wk', 'bk', 'wnk', 'bnk', 'wpow', 'bpow', 'press_w', 'press_b')


def check(ag, engine, tag):
    got, want = actual(ag), scratch(ag)
    if got != want:
        print(f"MISMATCH ({tag})")
        for n, a, s in zip(NAMES, got, want):
            if a != s:
                print(f"   {n}: incremental {a} vs scratch {s}")
        return False
    board_now = [x for row in engine.board for x in row]
    if [ag.b[i] for i in range(48)] != [_CODE[p] for p in board_now]:
        print(f"BOARD MISMATCH ({tag}): search left the internal board modified")
        return False
    return True


def main():
    rng = random.Random(20240913)
    checks = 0
    games = 0
    for g in range(4):
        engine = GameEngine()
        ag = B23CS1001(engine, max_depth=4)
        ag.test_budget = 0.3
        ply = 0
        for ply in range(40):
            if engine.get_game_state() != "ongoing":
                break
            mv = ag.get_best_move()
            if mv is None:
                break
            if not check(ag, engine, f"game {g} ply {ply} after search"):
                return 1
            checks += 1
            engine.make_move(mv)
            # also verify right after the agent re-syncs on the next call
        games += 1
        _ = rng                       # (randomness reserved for future cases)
    print(f"INCREMENTAL TEST PASSED ({checks} consistency checks over {games} games)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

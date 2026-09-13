"""Regression test: every incrementally maintained evaluation quantity must equal
a from-scratch computation after arbitrary make/unmake sequences, and the
adjudication tally must match the official points table exactly.

The agent maintains `_static`, `_press_w`, `_press_b`, `wnk`/`bnk`, `_wk`/`_bk`
and `wpow`/`bpow` during the search.  An asymmetric `_eval_add`/`_eval_sub` would
silently corrupt every score (and the board itself), and parity testing would not
catch it because move generation is untouched.  This test drives real searches and
then compares against a fresh scan of the board.

The second half checks the tally the agent plays to late in a game (captures +
2 per check given, the table an adjudicated game is decided on) against an
independent recomputation from the move log, and checks the root check test it
banks those points with against the engine's own check test.
"""
import random
import sys

from board import GameEngine
from config import EMPTY_SQUARE, PIECE_VALUES
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


def official_table(log):
    """Independent recomputation of the official table: captures + 2 per check."""
    caps, chk = [0, 0], [0, 0]
    e = GameEngine()
    for i, mv in enumerate(log):
        if mv.piece_captured != EMPTY_SQUARE:
            caps[i & 1] += abs(PIECE_VALUES[mv.piece_captured])
        e.make_move(mv)
        if e.is_in_check():
            chk[i & 1] += 2
    return caps, chk


def check_adjudication():
    """The tally the agent plays to must match an independent recomputation.

    Adjudicated games are decided on the official points table, so a wrong tally
    means the agent plays to a score that does not exist.  Both the realistic
    case (one agent per colour) and one agent playing both sides are covered.
    """
    fails = 0
    for two_agents in (True, False):
        engine = GameEngine()
        if two_agents:
            a, b = B23CS1001(engine), B23CS1001(engine)
            a.test_budget = b.test_budget = 0.05
            players = (("agent-White", a), ("agent-Black", b))
        else:
            a = b = B23CS1001(engine)
            a.test_budget = 0.05
            players = (("one agent, both colours", a),)
        for _ in range(40):
            if engine.get_game_state() != "ongoing":
                break
            p = a if engine.white_to_move else b
            mv = p.get_best_move()
            if mv is None:
                break
            engine.make_move(mv)
        # The opponent's last move is only observable on our next turn, so sync
        # exactly the way the agent does at the start of a move.
        a._load()
        b._load()
        want = official_table(engine.move_log)
        for name, ag in players:
            got = (list(ag._adj_caps), list(ag._adj_chk))
            if got != want:
                fails += 1
                print(f"  TALLY MISMATCH ({name}): {got} vs table {want}")
            else:
                print(f"  tally ({name}) {got} matches the move log")

    # The root check test the tally banks checks with must agree with the engine.
    engine = GameEngine()
    ag = B23CS1001(engine)
    ag.test_budget = 0.05
    total = bad = 0
    for _ in range(25):
        if engine.get_game_state() != "ongoing":
            break
        ag._load()
        opk = ag._bk if ag.wtm else ag._wk
        for mv in engine.get_legal_moves():
            mine = ag._gives_check(mv, opk)
            engine.make_move(mv)
            truth = engine.is_in_check()
            engine.undo_move()
            total += 1
            if mine != truth:
                bad += 1
                print(f"  check-test mismatch {mv}: agent {mine} engine {truth}")
        mv = ag.get_best_move()
        if mv is None:
            break
        engine.make_move(mv)
    print(f"  root check test: {total - bad}/{total} agree with the engine")
    return fails + (1 if bad else 0)


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
    failures = check_adjudication()
    if failures:
        print(f"INCREMENTAL TEST FAILED ({failures} adjudication problems)")
        return 1
    print(f"INCREMENTAL TEST PASSED ({checks} consistency checks over {games} games)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

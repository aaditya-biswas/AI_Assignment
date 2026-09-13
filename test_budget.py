"""Timing/budget test: the agent must stay inside its own clock.

With the 1-minute-per-player rule, running out of time LOSES the game, so this
checks two things:
  * every move finishes inside the budget `_plan_time` allocated for it (plus a
    small tolerance for interpreter/measurement overhead), and
  * a full game of "soft" spends still leaves margin on the 60 s clock.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from board import GameEngine
from B23CS1001 import B23CS1001


def time_move(engine, max_depth, budget):
    ag = B23CS1001(engine, max_depth=max_depth)
    ag.test_budget = budget              # force a known per-move budget
    t0 = time.monotonic()
    mv = ag.get_best_move()
    dt = time.monotonic() - t0
    return mv, dt, ag.nodes_expanded, ag.depth


def main():
    # Deterministic midgame: our own agent at a tiny budget, legal moves only.
    engine = GameEngine()
    opener = B23CS1001(engine, max_depth=2)
    opener.test_budget = 0.05
    for _ in range(8):
        if engine.get_game_state() != "ongoing":
            break
        m = opener.get_best_move()
        if m is None:
            break
        engine.make_move(m)

    failures = 0
    for budget in (0.3, 0.5):
        mv, dt, nodes, reached = time_move(engine, max_depth=6, budget=budget)
        nps = nodes / dt if dt else 0
        ok = dt < budget * 1.3 + 0.1
        failures += 0 if ok else 1
        print(f"budget={budget}s -> time={dt:6.2f}s [{'OK ' if ok else 'OVER'}] "
              f"nodes={nodes:>7} nps={nps:7.0f} depth_reached={reached} mv={mv}",
              flush=True)

    # The allocator must keep a margin over a whole game of soft spends.
    a = B23CS1001(GameEngine())
    a._load()
    used = 0.0
    for md in range(B23CS1001.ASSUMED_MOVES):
        a.time_used = used
        soft, _ = a._plan_time(md, False, False, False, 20)
        used += soft
    margin = B23CS1001.GAME_SECONDS - used
    ok = margin > 0
    failures += 0 if ok else 1
    print(f"{B23CS1001.ASSUMED_MOVES}-move soft-spend simulation: used "
          f"{used:.2f}s of {B23CS1001.GAME_SECONDS}s -> {margin:.2f}s margin "
          f"[{'OK ' if ok else 'OVER'}]")

    print("BUDGET TEST PASSED" if failures == 0
          else f"BUDGET TEST FAILED ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

"""Speed benchmark for the fast internal-search agent (ai_player.B23CS1001)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from board import GameEngine              # noqa: E402
from p25cs0004 import P25CS0004           # noqa: E402
from B23CS1001 import B23CS1001           # noqa: E402


def run(engine, budget, max_depth):
    ag = B23CS1001(engine, max_depth=max_depth)
    ag.test_budget = budget
    t0 = time.monotonic()
    mv = ag.get_best_move()
    dt = time.monotonic() - t0
    nps = ag.nodes_expanded / dt if dt else 0
    return mv, dt, ag.nodes_expanded, ag.depth, nps


def main():
    print("--- starting position ---", flush=True)
    e = GameEngine()
    for budget in (0.3, 1.0):
        mv, dt, nodes, depth, nps = run(e, budget, 12)
        print(f"budget={budget}s -> time={dt:5.2f}s nodes={nodes:>8} "
              f"nps={nps:8.0f} depth={depth} mv={mv}", flush=True)

    print("--- midgame (after 8 random plies) ---", flush=True)
    e = GameEngine()
    r = P25CS0004(e)
    for _ in range(8):
        m = r.get_best_move()
        if m:
            e.make_move(m)
    for budget in (0.3, 1.0):
        mv, dt, nodes, depth, nps = run(e, budget, 12)
        print(f"budget={budget}s -> time={dt:5.2f}s nodes={nodes:>8} "
              f"nps={nps:8.0f} depth={depth} mv={mv}", flush=True)


if __name__ == "__main__":
    main()

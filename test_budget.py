."""Quick timing/budget test: verify per-move wall time is bounded."""
import os
import sys
x]import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from board import GameEngine
from myagent import MyAgent


def time_move(engine, max_depth, budget):
    ag = MyAgent(engine, max_depth=max_depth)
    ag.test_budget = budget
    t0 = time.monotonic()
    mv = ag.get_best_move()
    dt = time.monotonic() - t0
    return mv, dt, ag.nodes_expanded, ag.depth


def main():
    # Mid-game-ish position after a handful of random-ish opening plies.
    engine = GameEngine()
    from p25cs0004 import P25CS0004
    opener = P25CS0004(engine)
    for _ in range(6):
        m = opener.get_best_move()
        if m:
            engine.make_move(m)

    for budget in (0.3, 0.5):
        mv, dt, nodes, reached = time_move(engine, max_depth=6, budget=budget)
        nps = nodes / dt if dt else 0
        ok = "OK " if dt < budget * 1.3 + 0.1 else "OVER"
        print(f"budget={budget}s -> time={dt:6.2f}s [{ok}] nodes={nodes:>7} "
              f"nps={nps:7.0f} depth_reached={reached} mv={mv}", flush=True)


if __name__ == "__main__":
    main()

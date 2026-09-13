"""Timing/budget test: the agent must stay inside its own clock.

With the 1-minute-per-player rule, running out of time LOSES the game, so this
checks two things:
  * every move finishes inside the budget `_plan_time` allocated for it (plus a
    small tolerance for interpreter/measurement overhead), and
  * a full game of "soft" spends still leaves margin on the 60 s clock.

    python3 test_budget.py                                  # quick checks
    python3 test_budget.py --stress --games 1 --contend 3   # real-clock games

`--stress` plays whole games with the REAL clock (no `test_budget` override) and
fails if any single move leaves less than MIN_MARGIN seconds of the 60 s clock,
optionally with busy threads competing for the CPU.
"""
import os
import sys
import threading
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


MIN_MARGIN = 3.5          # never let the clock get closer than this (timeout = loss)


def clock_stress(games=1, contend=0, max_plies=150):
    """Play whole games on the REAL clock and assert the reserve is kept.

    One shared agent plays both colours, which burns the clock roughly twice as
    fast as a real game and so drives the allocator into the end-of-clock regime
    where the reserve is what stands between us and a 600-point time-out.
    `contend` busy Python threads simulate a loaded machine: the runner's clock
    is wall time, so that is exactly the case the reserve has to survive.
    """
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            pass

    threads = [threading.Thread(target=spin, daemon=True) for _ in range(contend)]
    for t in threads:
        t.start()

    limit = B23CS1001.GAME_SECONDS - MIN_MARGIN
    failures = 0
    worst = 0.0
    try:
        for g in range(games):
            engine = GameEngine()
            agent = B23CS1001(engine)        # real clock: test_budget stays None
            ply = 0
            for ply in range(1, max_plies + 1):
                if engine.get_game_state() != "ongoing":
                    break
                mv = agent.get_best_move()
                if mv is None:
                    break
                engine.make_move(mv)
                worst = max(worst, agent.time_used)
                if agent.time_used > limit:
                    failures += 1
                    print(f"  OVER: game {g + 1} ply {ply} used "
                          f"{agent.time_used:.2f}s (limit {limit:.2f}s)",
                          flush=True)
            print(f"  game {g + 1}: {ply} plies, clock used "
                  f"{agent.time_used:.2f}s of {B23CS1001.GAME_SECONDS:.0f}s "
                  f"-> margin {B23CS1001.GAME_SECONDS - agent.time_used:.2f}s",
                  flush=True)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=0.5)

    print(f"worst clock reading {worst:.2f}s, required margin {MIN_MARGIN:.1f}s "
          f"-> {'OK' if not failures else f'FAILED ({failures} over)'}")
    return failures


def _arg_int(flag, default):
    """Read `--flag N` from argv, else `default`."""
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            try:
                return int(sys.argv[i + 1])
            except ValueError:
                pass
    return default


def main():
    if "--stress" in sys.argv:
        games = _arg_int("--games", 1)
        contend = _arg_int("--contend", 0)
        print(f"clock stress: {games} game(s), {contend} busy thread(s), "
              f"reserve {B23CS1001.RESERVE_SECONDS:.1f}s, required margin "
              f"{MIN_MARGIN:.1f}s", flush=True)
        return 1 if clock_stress(games, contend) else 0

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

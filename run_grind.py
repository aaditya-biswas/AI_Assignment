"""Repeated, paired-colour gauntlet with aggregate reporting (the A/B harness).

A single 4-game gauntlet cannot resolve the differences we are chasing: the
search cuts off on the wall clock, so the *same* build scores differently from
run to run (repeated runs of these very pairings have swung by hundreds of
points).  This harness plays each pairing `--repeats` times in BOTH colours and
aggregates the official numbers, so a change can be compared against a baseline
instead of a single anecdote.

The rules mirror run_games_fast.py exactly (that file is the source of truth for
the official Match Score Card):
    checkmate or time-out -> 600 : 0
    otherwise             -> captures at PIECE_VALUES + 2 per check given,
                              adjudicated at CAP plies

Usage
-----
    python3 run_grind.py                              # 2 repeats vs each opponent
    python3 run_grind.py --repeats 5
    python3 run_grind.py --opponents B23ME1074
    python3 run_grind.py --budget 0.25 --tag quick    # fast per-move budget
    python3 run_grind.py --tag adjudication-on         # annotate the summary

Anything passed with --tag is appended to grind_results.md so a later run can be
diffed against it.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from board import GameEngine                            # noqa: E402
from config import EMPTY_SQUARE, PIECE_VALUES           # noqa: E402
from B23CS1001 import B23CS1001                         # noqa: E402

CAP = 150
PLAYER_LIMIT = 60.0
ME = "B23CS1001"


def _load_opponents(names):
    """Import the opponent classes by name, skipping the ones not present."""
    out = []
    for name in names:
        try:
            mod = __import__(name)
            out.append((name, getattr(mod, name)))
        except Exception as exc:                        # noqa: BLE001
            print(f"  (skipping {name}: {exc!r})", flush=True)
    return out


def play_stats(white_cls, black_cls, budget=None):
    """One game with the official scoring; returns a stats dict.

    The clock is the official one: each player's own thinking time is summed and
    whoever passes PLAYER_LIMIT loses.  `budget` (if given) is a per-move budget
    handed to agents that expose the test hook, for fast A/B runs.
    """
    engine = GameEngine()
    white = white_cls(engine)
    black = black_cls(engine)
    for p in (white, black):
        if budget is not None and hasattr(p, "test_budget"):
            p.test_budget = budget
    wp = bp = wchk = bchk = 0
    clocks = {"White": 0.0, "Black": 0.0}
    timeout_loser = None
    ply = 0
    for ply in range(1, CAP + 1):
        if engine.get_game_state() != "ongoing":
            break
        side = "White" if engine.white_to_move else "Black"
        if clocks[side] >= PLAYER_LIMIT:
            timeout_loser = side
            break
        player = white if engine.white_to_move else black
        t0 = time.perf_counter()
        mv = player.get_best_move()
        clocks[side] += time.perf_counter() - t0
        if clocks[side] > PLAYER_LIMIT:
            timeout_loser = side
            break
        if mv is None:
            break
        mover_white = engine.white_to_move
        engine.make_move(mv)
        if mv.piece_captured != EMPTY_SQUARE:
            pts = abs(PIECE_VALUES.get(mv.piece_captured, 0))
            if mover_white:
                wp += pts
            else:
                bp += pts
        if engine.is_in_check():
            if mover_white:
                wchk += 2
            else:
                bchk += 2
    final = engine.get_game_state()
    winner = "-"
    if timeout_loser is not None:
        winner = "Black" if timeout_loser == "White" else "White"
    elif final == "checkmate":
        winner = "Black" if engine.white_to_move else "White"
    if timeout_loser is not None or final == "checkmate":
        ow, ob = (600, 0) if winner == "White" else (0, 600)
    else:
        ow, ob = wp + wchk, bp + bchk
    us_white = type(white).__name__ == ME
    ours, theirs = (ow, ob) if us_white else (ob, ow)
    caps_us, caps_them = (wp, bp) if us_white else (bp, wp)
    chk_us, chk_them = (wchk, bchk) if us_white else (bchk, wchk)
    mate_for = final == "checkmate" and ((winner == "White") == us_white)
    return {
        "opponent": type(black if us_white else white).__name__,
        "we_are_white": us_white,
        "final": "TIMEOUT" if timeout_loser else final,
        "winner": winner,
        "ply": ply,
        "ours": ours, "theirs": theirs,
        "caps_us": caps_us, "caps_them": caps_them,
        "chk_us": chk_us, "chk_them": chk_them,
        "clock_us": clocks["White" if us_white else "Black"],
        "clock_them": clocks["Black" if us_white else "White"],
        "mate_for": mate_for,
        "mate_against": final == "checkmate" and winner != "-" and not mate_for,
        "timeout": timeout_loser is not None,
    }


def _summary(rows, opponent=None):
    """Aggregate a list of game stats into the numbers an A/B is judged on."""
    n = len(rows)
    if not n:
        return None
    wins = sum(1 for r in rows if r["ours"] > r["theirs"])
    losses = sum(1 for r in rows if r["ours"] < r["theirs"])
    s = {
        "name": opponent or "TOTAL",
        "games": n,
        "wins": wins,
        "losses": losses,
        "draws": n - wins - losses,
        "pts_us": sum(r["ours"] for r in rows) / n,
        "pts_them": sum(r["theirs"] for r in rows) / n,
        "caps_diff": sum(r["caps_us"] - r["caps_them"] for r in rows) / n,
        "chk_diff": sum(r["chk_us"] - r["chk_them"] for r in rows) / n,
        "ply": sum(r["ply"] for r in rows) / n,
        "clock_us": sum(r["clock_us"] for r in rows) / n,
        "clock_max": max(r["clock_us"] for r in rows),
        "mate_for": sum(1 for r in rows if r["mate_for"]),
        "mate_against": sum(1 for r in rows if r["mate_against"]),
        "timeouts": sum(1 for r in rows if r["timeout"]),
        "adjudicated": sum(1 for r in rows if r["final"] == "ongoing"),
    }
    s["diff"] = s["pts_us"] - s["pts_them"]
    return s


def print_summary(s):
    if s is None:
        return
    print(f"  {s['name']:<12} {s['games']:>2} games  "
          f"W-L-D {s['wins']}-{s['losses']}-{s['draws']}  "
          f"| official us {s['pts_us']:6.1f} them {s['pts_them']:6.1f} "
          f"diff {s['diff']:+7.1f}"
          f"\n               material diff {s['caps_diff']:+6.1f}   "
          f"check diff {s['chk_diff']:+5.1f}   "
          f"mates {s['mate_for']}-{s['mate_against']}   "
          f"timeouts {s['timeouts']}   adjudicated {s['adjudicated']}"
          f"\n               mean plies {s['ply']:5.1f}   our clock "
          f"mean {s['clock_us']:5.1f}s max {s['clock_max']:5.1f}s "
          f"(limit {PLAYER_LIMIT:.0f}s)", flush=True)


def _arg(flag, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def main():
    repeats = int(_arg("--repeats", "2"))
    budget = _arg("--budget")
    budget = float(budget) if budget else None
    tag = _arg("--tag", "")
    names = _arg("--opponents", "B23ME1074,B23ES1030,AIPlayer").split(",")
    opponents = [(n, c) for n, c in _load_opponents(names) if n != ME]
    if not opponents:
        print("no opponents available")
        return 1
    print(f"grind: {repeats} repeat(s) per colour vs "
          f"{', '.join(n for n, _ in opponents)}"
          f"{' (budget ' + str(budget) + 's/move)' if budget else ''}",
          flush=True)
    all_rows, lines = [], []
    for name, cls in opponents:
        rows = []
        for _ in range(repeats):
            for white_cls, black_cls in ((B23CS1001, cls), (cls, B23CS1001)):
                r = play_stats(white_cls, black_cls, budget)
                rows.append(r)
                all_rows.append(r)
                print(f"    [{name} r{len(rows)}] we are "
                      f"{'W' if r['we_are_white'] else 'B'}: "
                      f"{r['final']:>9} winner {r['winner']:>5} ply {r['ply']:3d} "
                      f"| official {r['ours']:>4} vs {r['theirs']:<4} "
                      f"| caps {r['caps_us']}-{r['caps_them']} "
                      f"chk {r['chk_us']}-{r['chk_them']} "
                      f"| clocks us {r['clock_us']:4.1f}s them "
                      f"{r['clock_them']:4.1f}s", flush=True)
        s = _summary(rows, name)
        print_summary(s)
        lines.append(s)
    total = _summary(all_rows)
    print_summary(total)
    if tag:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "grind_results.md")
        with open(path, "a") as fh:
            fh.write(f"\n### {tag} ({time.strftime('%Y-%m-%d %H:%M')}, "
                     f"{repeats} repeat(s)"
                     f"{', budget ' + str(budget) + 's/move' if budget else ''}"
                     f")\n\n")
            for s in lines + [total]:
                fh.write(f"- {s['name']}: {s['games']} games W-L-D "
                         f"{s['wins']}-{s['losses']}-{s['draws']}, official "
                         f"diff {s['diff']:+.1f}, material diff "
                         f"{s['caps_diff']:+.1f}, checks diff "
                         f"{s['chk_diff']:+.1f}, mates {s['mate_for']}-"
                         f"{s['mate_against']}, timeouts {s['timeouts']}, "
                         f"mean plies {s['ply']:.1f}, our clock max "
                         f"{s['clock_max']:.1f}s\n")
        print(f"appended to grind_results.md under tag {tag!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

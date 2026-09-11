"""Differential (parity) test: the internal move generator inside ai_player.py
must agree exactly with the engine's get_legal_moves() on many random
positions, both colours."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from board import GameEngine          # noqa: E402
from B23CS1001 import B23CS1001, BW   # noqa: E402


def squares(engine_moves):
    return {(m.start_row * BW + m.start_col, m.end_row * BW + m.end_col)
            for m in engine_moves}


def main():
    rng = random.Random(20260905)
    positions = 0
    mismatches = 0
    for game in range(40):
        engine = GameEngine()
        ag = B23CS1001(engine)
        for ply in range(80):
            ag._load()
            my = 'w' if engine.white_to_move else 'b'
            ours = ag._legal_moves(ag._find_king(my))
            ours_set = {(fr, to) for fr, to, _, _ in ours}
            eng = engine.get_legal_moves()
            eng_set = squares(eng)
            positions += 1
            if ours_set != eng_set:
                mismatches += 1
                print(f"MISMATCH game={game} ply={ply}", flush=True)
                print(f"  ours({len(ours_set)}) != engine({len(eng_set)})",
                      flush=True)
                print(f"  only-ours : {sorted(ours_set - eng_set)[:8]}",
                      flush=True)
                print(f"  only-engine: {sorted(eng_set - ours_set)[:8]}",
                      flush=True)
                for r in range(8):
                    print("   " + " ".join(engine.board[r]), flush=True)
                break
            if not eng:
                break
            engine.make_move(rng.choice(eng))
        if mismatches:
            break
    print(f"PARITY: {positions} positions checked, mismatches={mismatches}")
    print("PARITY TEST " + ("PASSED" if mismatches == 0 else "FAILED"))


if __name__ == "__main__":
    main()

# Baseline (frozen before the next optimization round)

All figures measured on this machine (Python 3.14, single core) with the
current `B23CS1001.py`. These are the numbers any new change must beat.

## Headline: original template -> current agent

| Metric | Original agent | Current baseline |
|---|---|---|
| Search throughput (NPS) | ~2.9-4 k | ~19-20 k |
| Depth @ 0.35 s (start / midgame) | ~4-6 / ~2-4 | 7-8 / 6-7 |
| vs B23ME1074 | lost both (0-20, 40-510; mated ply 7) | net positive (see below) |

## Hot-path cost per call (current)

| Function | us/call |
|---|---|
| `_legal_moves` | ~15 |
| `_evaluate_white` | ~11 |
| `_snapshot` | ~5.3 |
| `_attacked` | ~0.96 |
| `_position_key` | ~0.52 |
| `_ordering_key` | ~6.5/node (repeated killer dict lookup inside) |

## Speed benchmark (run_bench_fast.py)

- budget 0.3 s: start depth 7-8, midgame depth 6-7, NPS ~18-20 k
- budget 1.0 s: start depth 9

(NPS counts negamax nodes only; quiescence is not counted, so it is noisy.)

## Head-to-head vs B23ME1074 (0.35 s/move)

- Standard gauntlet (last run): `B23CS1001` 250-130 (+120);
  `B23ME1074` 220-470 as Black (+250) -> **net +370**.
- 6-game harness: +40, +30* (checkmate win as Black), +20, +170, +50, +240 ->
  **net +550**, all six games positive.

## Correctness / regression tests

- `run_parity.py`: **PASSED** (3195 positions, 0 mismatches)
- `test_endgame.py`: **5/5** mates converted (KR, KR, KRR, KBB, KRKP)
- `_STATIC` table: validated, symmetric start eval = 0

## Baseline to beat (acceptance gate for every change)

- parity stays PASSED, depth not lower, and
- B23ME1074 multi-game net >= +550 (6 games) / >= +370 (single gauntlet).

## Optimization queue (in order)

- 1.1 side-to-move-separated history
- 1.2 hoist killers lookup out of the per-move key function
- 1.3 CMH (killer + counter-move hybrid)
- 2.1 Late Move Pruning (LMP)
- 3.1 bounded check extension

Design details and rationale: see `improvement.md` and the plan.

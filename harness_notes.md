# Harness notes — why `game_runner.py` and `run_games_fast.py` disagree

Both harnesses play the same game with the same engine and the same 60 s clock.
They disagree only about the **score card**.

| | `game_runner.py` (provided template) | `run_games_fast.py` / `run_grind.py` |
|---|---|---|
| source of rules | README.md (stale: "King 300", "checkmate +300") | `AI_Assignment_I.pdf` -> `constraint.md` 9 |
| checkmate | winner **+300**, and both sides keep captures + checks | winner **600, loser 0**, captures **overridden** |
| other loss | not modelled | **0** |
| time-out | game stops and **nobody scores** (prints "wins on time") | flagged side **loses**: 0 vs opponent 600 |
| stalemate / ply 150 | captures + checks | captures + 2 per check (identical) |
| legality | not checked (engine accepts any `Move`) | asserts the move is in `get_legal_moves()` |

So `game_runner.py` shows a **larger** number whenever you win by mate
(captures kept + 300) and a **smaller/odd** number when you flag (0-0).
`run_games_fast.py` is the one that matches the PDF.

`game_runner.py` now also prints an additive `Official score card (match result)`
block (pure function `official_score()`, tested for all five outcomes), so the two
harnesses can be compared directly. Nothing about the template's own totals changed.

## Measured evidence (13 Sep)

`run_games_fast.py` (official card) vs `B23CS1082`, both colours:

```
B23CS1082 vs B23CS1001 -> checkmate (winner Black) ply 109 | official   0 vs 600 | caps 410-510 chk 14-10 | clocks W 40.3s B 50.2s
B23CS1001 vs B23CS1082 -> checkmate (winner White) ply  70 | official 600 vs 0   | caps 290-280 chk 12-14 | clocks W 30.3s B 26.5s
```

Both are **wins for us (600-0)**, i.e. +1200 official over two games.
The legacy template would have printed 820 and 602 for us on those same games
(captures + checks + 300), which is the "more points in game_runner" effect.

The user's own `game_runner.py` game ended at the ply-150 turn limit
(`White 512 - Black 538`), which is a **draw**: here the two score cards agree
exactly, because captures + checks are used in both.

Time-out path exercised directly (`run_game(B23CS1082, B23CS1001, 3)`):

```
White wins on time!
Total: White 0, Black 0            <- legacy template: no score at all
Official score card: White 600, Black 0   (Black flagged -> 0)
```

## Take-aways

1. Use `run_games_fast.py` / `run_grind.py` for decisions; `game_runner.py` is a
   board viewer.
2. A mate makes captures worthless; a flag makes everything worthless. Only the
   600/0 column matters for the tournament.
3. Our clock use against `B23CS1082` was 50.2 s and 30.3 s of 60 s: the margin
   holds, but game 1 used 50.2 s and is the tightest seen so far.

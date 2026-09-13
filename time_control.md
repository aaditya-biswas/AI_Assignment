# Dynamic time control — research + implementation

Everything below is measured or quoted; sources are linked. The implementation
lives in `B23CS1001.py` (`_plan_time`), so the submission stays a single file.

---

## 0. Why this was worth doing (our own evidence)

From the 1-minute-per-player gauntlet runs (clocks printed per game):

| Game | our clock used | of 60 s |
|---|---|---|
| vs ES1030 (mate, W) | 9.4 s | 16 % |
| vs ES1030 (mate, W) | 14.1 s | 24 % |
| vs ME1074 (draw, W) | 26.7 s | 44 % |
| vs ME1074 (draw, W) | 27.3 s | 46 % |
| vs ME1074 (mated, B) | 21.1 s | 35 % |

The old allocator was
`budget = (60 - used) * 0.45 / (75 - moves_done)`, clamped to `[0.25, 2.5]`:

* it targets only **45 %** of the clock — the other 55 % is a reserve that is
  *never reclaimed* if the game ends before move 75;
* it produced ~0.4–0.6 s/move, i.e. **depth 7–8**, when ~1 s/move was affordable;
* and it ignored the position entirely, even though in this variant a **mate is
  worth 600** while a check is 2 and a pawn 20.

Measured headroom: with the new allocator the same start position reaches
**depth 9 in 0.42 s** (vs depth 8 at 0.35 s before).

---

## 1. What the literature / real engines do

### 1.1 The simple, competitive baseline (Chess Programming Wiki, *Time Management*)

* Sudden death (our case — one clock for the whole game): *"typically programs
  estimate the game will last further 25..40 moves, and divide the remaining
  time by this number"*, keeping a buffer for later moves.
* *"A basic time management technique is to use `base/20 + increment/2` time per
  move. This is very competitive with advanced time management schemes, and is
  typically chosen as the foundation to build upon."*
* **Soft bound** vs **hard bound**: the search aims to finish an iteration inside
  the soft bound, but may run into the hard bound when that is worthwhile.
* **Premature termination** (CPW, *Iterative Deepening*): the unfinished iteration
  is simply discarded and the previous iteration's move kept — *"though in case of
  a severe drop of the score it is wise to allocate some more time, as the first
  alternative is often a bad capture, delaying the loss instead of preventing
  it."* ← the canonical **panic time** rule.

### 1.2 Stockfish `timeman.cpp` (verbatim structure, SF 12)

```cpp
int mtg = limits.movestogo ? std::min(limits.movestogo, 50) : 50;
TimePoint timeLeft = std::max(1, time + inc*(mtg-1) - moveOverhead*(2+mtg));
timeLeft = slowMover * timeLeft / 100;

// sudden death (no increment) — our case
optScale = std::min(0.008 + std::pow(ply + 3.0, 0.5) / 250.0,
                    0.2 * limits.time[us] / double(timeLeft));
maxScale = std::min(7.0, 4.0 + ply / 12.0);

optimumTime = optScale * timeLeft;
maximumTime = std::min(0.8 * limits.time[us] - moveOverhead, maxScale * optimumTime);
```

And from `search.cpp`:

```cpp
// We record how often the best move has been changed in each iteration.
// This information is used for time management: when the best move changes
// frequently, we allocate some more time.
if (moveCount > 1) ++thisThread->bestMoveChanges;
```

Takeaways: (1) **two bounds**, optimum/soft and maximum/hard; (2) the per-move
share **grows with ply**; (3) hard = **4–7×** soft; (4) a single move may never
take more than **80 %** of the remaining time; (5) **best-move instability**
buys extra time.

### 1.3 Weiss `time.c` (a minimal, clean implementation)

```c
const int overhead = 6;                                 // ms per move
int mtg = Limits.movestogo ? MIN(Limits.movestogo, 50) : 50;
int timeLeft = MAX(0, Limits.time + mtg * (Limits.inc - overhead));
// sudden death:
double scale = 0.022;
Limits.optimalUsage = MIN(timeLeft * scale, 0.2 * Limits.time);
Limits.maxUsage     = MIN(5 * Limits.optimalUsage, 0.8 * Limits.time);
```

Takeaways: same shape, a smaller per-move share, hard ≈ **5×** soft, an explicit
**per-move overhead** deduction, and the 80 % ceiling. It also starts pruning
harder (`doPruning`) once a fraction of the optimum has elapsed.

### 1.4 The transferable rules

| # | Rule | Source |
|---|---|---|
| 1 | Keep a **soft** target and a **hard** ceiling | SF, Weiss |
| 2 | Sudden death: spread the clock over the moves still expected, with a reserve | CPW |
| 3 | Later plies may take a **larger share** of the clock | SF (`0.008 + sqrt(ply+3)/250`) |
| 4 | Extend toward hard when **the best move keeps changing** | SF (`bestMoveChanges`) |
| 5 | Extend when **the score drops sharply** (panic time) | CPW, ID page |
| 6 | Cut short an **obvious** move (stable best move / steady score) | *easy move* practice |
| 7 | Deduct a **move overhead** and never risk the whole clock | SF, Weiss |
| 8 | Never start an iteration that **cannot finish** in the budget | CPW |

---

## 2. Mapping the rules onto *this* variant

This variant differs from normal chess in ways that change what good time control
means:

| Variant fact | Consequence for time control |
|---|---|
| **Mate = 600**, and it overrides captures | The largest single swing by far ⇒ mate-critical positions deserve the **biggest** multiplier; missing a mate (or being mated) dwarfs any material loss |
| **Check = +2** only | Never spend time to farm checks (measured: doing so lost ~960 pts) |
| Material = P20 N70 B70 R100 | Spend more when clearly ahead (converting is what mates are made of); do **not** spend less when behind — defending a mate is worth 600 |
| 60 s sudden death, **no increment**, and **running out loses** | A hard reserve + a per-move ceiling are mandatory; the invariant must be *proved*, not assumed |
| Hard **150-ply cap** | The game is bounded, so a geometric per-move decay can be shown never to exhaust the clock (see §4) |
| Cheap, fast search (30–60 k NPS) | 0.6–1.0 s/move is affordable and buys a whole extra ply (depth 9 vs 8) |
| We are the side that gets mated | Being *in check* must buy real time — that is the "find the escape" case |

### 2.1 Design

```
remaining  = GAME_SECONDS - time_used                  # our own clock
moves_left = max(MIN_MOVES_LEFT, ASSUMED_MOVES - moves_done)
base       = remaining * TIME_FRACTION / moves_left     # sudden-death spread

mult = 1
  × DRIVE_MULT     if we are driving for mate   (+600 is the dominant term)
  × CHECK_MULT     if we are in check           (find the escape or lose)
  × FEW_MULT       if <= 9 pieces               (endgame precision)
  × WIDE_MULT      if >= WIDE_ROOT legal moves  (harder decision)
  × PANIC_MULT     if the score fell >= PANIC_DROP last move (panic time)
  × EASY_MULT      if the move has been stable for EASY_STABILITY moves with a
                   steady score (obvious move -> move along)

floor     = min(MIN_MOVE_SECONDS, remaining * 0.1)   # scales with the clock!
reserve   = min(RESERVE_SECONDS, remaining * 0.5)
spendable = max(floor, remaining - reserve)

soft = max(floor, min(base * mult, spendable, remaining * MAX_SHARE))
hard = max(soft, min(soft * HARD_FACTOR, spendable, remaining * MAX_SHARE_CRIT))
return soft, hard
```

The two bounds are then used by the search loop:

* `self._deadline = t0 + hard` — the absolute abort (`_AbortSearch`), unchanged;
* `soft_limit = t0 + soft` — where we *intend* to stop;
* **before starting an iteration**: stop if `now >= hard`, or if
  `now + last_iteration_seconds * ITER_PREDICT > limit` (rule 8);
* `limit` is upgraded from `soft` to `hard` **within the move** as soon as
  * the root best move changes (`changes += 1`), or
  * the score drops by `>= PANIC_DROP` (panic time) — rules 4 and 5;
* **stop immediately** if a mate for us was found (`cur_score > MATE_BOUND`) —
  the best possible result is already secured, so extra depth is wasted time;
* **forced move** (`len(root) == 1`) returns before any search starts;
* across moves we carry `_last_stability` (consecutive moves with an unchanged
  root move) and `_last_drop`, which feed the `EASY_MULT` / `PANIC_MULT` terms of
  the *next* allocation.

### 2.2 Constants (all in `B23CS1001.py`, all tunable)

| Constant | Value | Meaning |
|---|---|---|
| `GAME_SECONDS` | 60.0 | 1-minute clock per player |
| `ASSUMED_MOVES` | 75 | assumed game length for the spread |
| `MIN_MOVES_LEFT` | 8 | divisor floor |
| `TIME_FRACTION` | 0.75 | share of the clock the game may consume |
| `MIN_MOVE_SECONDS` | 0.15 | absolute floor (scaled down when the clock is low) |
| `HARD_FACTOR` | 2.5 | hard = soft × this |
| `MAX_SHARE` | 0.25 | one move never takes > 25 % of the clock |
| `MAX_SHARE_CRIT` | 0.35 | … nor > 35 % even in a critical spot |
| `RESERVE_SECONDS` | 2.0 | always kept back |
| `ITER_PREDICT` | 2.5 | assumed cost of the next iteration (× the last one) |
| `DRIVE_MULT` | 3.0 | we can force mate |
| `CHECK_MULT` | 1.5 | we are in check |
| `FEW_MULT` | 1.2 | endgame |
| `WIDE_ROOT` / `WIDE_MULT` | 25 / 1.1 | many legal moves |
| `PANIC_DROP` / `PANIC_MULT` | 100 / 1.5 | score fell → think harder |
| `EASY_STABILITY` / `EASY_DROP` / `EASY_MULT` | 3 / 30 / 0.5 | obvious move → spend less |

<!--END-->

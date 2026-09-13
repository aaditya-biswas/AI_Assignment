# Check avoidance — research and design

Goal: stop conceding long runs of checks (and the material/mate trouble that comes
with them) **without** ever weakening the position so that a different piece
delivers mate. Everything here is grounded in the sources quoted below and in our
own game data.

---

## 1. The evidence from our games

In the gauntlet games the check column (`chk W-B`, i.e. points at +2 each) reads:

| Game | checks we gave | checks we conceded |
|---|---|---|
| vs ME1074 (draw, we White) | 0 | **5** |
| vs ME1074 (mated, we Black) | 2 | **7** |
| vs ME1074 (draw, we White) | 6 | **3** |
| vs ME1074 (mated, we Black) | 5 | **11** |
| vs ME1074 (draw, we Black) | 0 | **32** |

Conceding 32 checks is **64 points** of the official score — more than a rook
(100 is close, 64 is ~3 pawns) — and, much worse, it means our king is being
driven around while their pieces improve. The old engine's own comment (in the
`_legal_moves` era) already noted we "concede far more checks than we give".

**Why the +2 never showed up in our search:** the official +2 is an *event* in the
game score, but our evaluation only knows material + piece squares. The search
cannot "see" that being checked costs points, so it happily walks into positions
where the opponent has a free +2 every move.

### 1.1 What went wrong the first time (Round 8)

In Round 8 we added **+2 for *giving* check** inside the search near the root and
it lost ~960 points over 6 paired games. That experiment was *offensive*, and it
was structurally wrong for two reasons:

1. It was a **path-dependent event bonus**: a position's value then depended on
   how many checks occurred earlier in the line, which is exactly what a
   transposition table cannot store (which is why it had to be limited to
   `ply <= 2`, i.e. applied inconsistently).
2. A check is **worth 2** while a pawn is 20 and a mate is 600, so making the
   search *prefer* checks traded away far more than it gained.

Check **avoidance** is the mirror image and must therefore be:
* **positional** (a property of the position, so it is TT-safe and symmetric),
* **bounded** (never able to outweigh material or a mate),
* **scaled by the attacker's remaining material** (with few pieces left there is
  no attack to fear).

## 2. What the literature says

### 2.1 King zone + attack units (Chess Programming Wiki, *King Safety*)

* **King zone**: *"usually defined as squares to which enemy King can move plus
  two or three additional squares facing enemy position."*
* The classic *"basic king safety function, similar to the one described in Toga
  log user manual"*: keep `attackingPiecesCount` and `valueOfAttacks`; *"If a piece
  attacks enemy king zone, we increase attackingPiecesCount by one, and count how
  many squares within enemy King zone are attacked. We multiply the number of
  attacked squares by a constant: 20"* and then map the accumulated attack
  through a **non-linear table** (the table in the literature rises steeply and
  saturates around 500 — i.e. 1–2 attackers are worth almost nothing, 4+ is worth
  a lot; and *"it is advisable not to evaluate king attack if only two pieces are
  attacking"*).
* **Pawn shield / pawn storm**: *"it is important to preserve pawns next to
  [the king]... The lack of a shielding pawn deserves a penalty, even more so if
  there is an open file next to the king."*  *"Penalties for storming enemy pawns
  must be lower than penalties for (semi)open files."*
* **Scaling by the attacker's material**: *"TSCP uses the pawn shield and pawn
  storm score, scaled by the opponent's material... Fruit uses a more elaborate
  scheme, counting the bonuses for attacking the squares near to the enemy king,
  and then multiplying their sum by the constant derived from the number of
  attackers."*
* **King tropism**: distance from the enemy pieces to the king, weighted by piece
  value (*"double the distance value for a queen, and halve it for bishops and
  rooks"*). The wiki is explicit that this is probabilistic: *"This kind of
  evaluation acts in a probabilistic way – it is by no means certain that being
  close to the king helps to attack it."* Used by Crafty.

### 2.2 Weiss `evaluate.c` (concrete constants)

```c
ei->kingZone[color]   = AttackBB(KING, kingSq(color), 0);   // the 8 neighbours
ei->attackPower[color] = -30;                               // baseline
ei->attackCount[color] = 0;
const int Shelter = S(31, -12);                             // pawn shelter
const int KingLineDanger[28] = { S(0,0), S(0,0), S(15,0), S(11,21),
    S(-16,35), S(-25,30), S(-29,29), S(-37,38), S(-48,41), S(-67,43), ... };
```
The `KingLineDanger` table is the safety valve: the penalty is *zero for the first
two entries* and then grows monotonically — a non-linear response, exactly as in
§2.1.

### 2.3 So the transferable recipe is

| Step | What | Why |
|---|---|---|
| 1 | Define a **king zone** (king ring, maybe extended toward the enemy) | attackers *of the ring* are what converts to checks and mate |
| 2 | Count **attackers** and the **squares they attack** there | more attackers ⇒ much worse (checks, then mate) |
| 3 | Map through a **non-linear** table, zero for 1–2 attackers | avoids noise from a single stray piece |
| 4 | Add **pawn shelter** and **open-file** terms | keeps the king's cover intact (the "interpose a pawn" case) |
| 5 | **Scale by the attacker's material** | an attack with 1 piece left cannot mate |
| 6 | Keep the whole term **small relative to material** | so it can never win an argument against a pawn, and never against a mate |

---

## 3. How to do it in *our* code (three options, cost-ranked)

### Option A — incremental "king pressure" accumulator  ← recommended

The wiki's *tropism* idea, made free by the incremental evaluation we already
have (we maintain `_static`, `_wk`/`_bk`, piece counts and power on every
make/unmake):

* precompute `_PROX[sq][ksq]` — a weight that is non-zero only within a few
  squares of the king (closer = larger);
* maintain `self._press_w` = Σ over **black** non-king pieces of
  `_PROX[sq][white_king] * _WT[piece]`, and `self._press_b` symmetrically;
* update both in `_eval_add` / `_eval_sub`: the moving piece's contribution moves
  with it, a captured piece drops out, and a **king move** forces a recompute of
  the other side's sum (≤ 16 pieces, rare);
* evaluation: `score += min(CAP, (_press_b - _press_w) * SCALE)`, scaled down as
  the attacker's material disappears.

Cost: **~0 in the evaluation** (it is incremental) and ~4 extra table lookups per
make/unmake ⇒ roughly 3 % of NPS.  This is the probabilistic form the wiki
describes for Crafty, but without its per-node scan.

### Option B — king-zone attack units in the evaluation (closest to the literature)

For every square `t` in `_KING_ATT[ksq]` (≤ 8) test whether the opponent attacks
it — using the inverse pawn tables, `_KNIGHT_ATT[t]` and the slider rays — then
apply the non-linear table of §2.1/§2.2.

Cost: ~8–16 µs **per side per evaluation**, i.e. roughly doubling our current
~11 µs evaluation ⇒ ~30 % NPS loss.  It is the most faithful, but we would pay
about one ply of depth for it — and *depth* is what actually refutes a mate (see
§4).  It can be gated (middlegame only, ≥2 attacking minors near the king) to
soften the cost.

### Option C — search-side handling

We already search **all evasions** in quiescence while in check, and a bounded
check extension was tested and reverted.  The remaining cheap idea is
**evasion ordering**: while in check, order the candidate evasions by the king
safety of the resulting position (the `_king_danger` helper already exists), so
interpositions are examined before king moves.  This is only as good as the
evaluation behind it, so it is a complement to A/B, not a replacement.

## 4. The caveat you raised — "don't let another piece mate us"

A naive "avoid checks" weight can *cause* a mate (interposing with a pawn can open
a diagonal for a bishop). Three guards, in order of importance:

1. **Mates dominate the score by construction.** A mate is `±(MATE - ply)`
   ≈ 100 000, many orders of magnitude above any positional term, so a king-safety
   weight can *never* talk us into a mate that the search can see — it only
   chooses between moves that are all non-losing within the horizon. The danger
   is therefore only *beyond* the horizon, and the real cure for that is **depth**
   — which is exactly what the new time control buys (it reaches depth 9 now).
2. **Cap and scale the term.** Cap it below a pawn (e.g. ≤ 40 points) so it can
   never justify a material sacrifice, and multiply it by the attacker's remaining
   material so it fades to zero in the endgame.
3. **Make it non-linear** (zero or tiny for 1–2 attackers, rising steeply after
   that) so that a single stray piece — which can check but rarely mate — does not
   generate a large penalty, matching §2.1–2.2.

A useful corollary: this is *precisely* why the term must be **positional** rather
than an event bonus (Round 8's failed experiment). A positional term is applied
consistently at every node and is TT-safe, so the search can compare
"interpose with the pawn" against "walk the king out" on equal terms, and the
deeper search is free to overrule the heuristic whenever a mate appears.

## 5. Recommended order of work

1. Implement **Option A** (incremental king pressure) plus a **self-check test**:
   after every N moves, recompute the pressure from scratch and assert it equals
   the incrementally maintained value. This is the exact bug class (asymmetric
   make/unmake) that would otherwise silently corrupt the whole evaluation.
2. Measure: NPS impact (expect ~3 %) and, most importantly, **checks conceded per
   game** — the metric the term exists to reduce.
3. Then A/B the gauntlet, and keep it only if the checks conceded fall without the
   mate count rising.
4. Keep `_king_danger` and the pawn-shield term, and consider **raising the shield
   weight** — that is what makes "interpose/keep a pawn in front" preferred over
   moving the king, i.e. your example.
5. Only if that is not enough, add Option B behind the gating, or Option C's
   evasion ordering.


# Agent Improvement Plan

This checklist tracks the strongest upgrades for the current chess agent. The goal is to improve tactical correctness, search efficiency, and positional judgment in a way that is compatible with the assignment constraints.

## Reference resources

These are the standard references for the optimizations discussed:

1. Chess Programming Wiki — Transposition Table  
   https://www.chessprogramming.org/Transposition_Table

2. Chess Programming Wiki — Alpha-Beta  
   https://www.chessprogramming.org/Alpha-Beta

3. Chess Programming Wiki — Move Ordering  
   https://www.chessprogramming.org/Move_Ordering

4. Chess Programming Wiki — Quiescence Search  
   https://www.chessprogramming.org/Quiescence_Search

5. Chess Programming Wiki — Killer Heuristic  
   https://www.chessprogramming.org/Killer_Heuristic

6. Chess Programming Wiki — History Heuristic  
   https://www.chessprogramming.org/History_Heuristic

7. Chess Programming Wiki — Evaluation Function  
   https://www.chessprogramming.org/Evaluation_Function

8. Chess Programming Wiki — Mobility  
   https://www.chessprogramming.org/Mobility

9. Chess Programming Wiki — King Safety  
   https://www.chessprogramming.org/King_Safety

10. Chess Programming Wiki — Passed Pawn  
   https://www.chessprogramming.org/Passed_Pawn

11. Andrew N. Luce, "Chess Programming" notes / engine concepts (classic practical references)  
   https://www.chessprogramming.org/

12. Artifical Intelligence: A Modern Approach (minimax / alpha-beta foundations)  
   https://aima.cs.berkeley.edu/

These references are the standard literature and practical references behind the improvements below.

---

## Validity check against the assignment constraints

The SOTA ideas listed below are valid as high-level chess-engine concepts, but they are not all compatible with the rules in [constraint.md](constraint.md).

The following are acceptable as agent-side upgrades inside [B23CS1001.py](B23CS1001.py):
- Principal Variation Search (PVS / NegaScout)
- Null Move Pruning (NMP)
- Late Move Reductions (LMR)
- Reverse Futility Pruning (RFP)
- improved TT usage and move ordering
- quiescence search integration at the horizon
- agent-local compact board-key optimization

Not allowed under the assignment rules:
- modifying the game engine or core rules in [board.py](board.py) or [config.py](config.py)
- engine-level bitboard migration or board rewrite in the official engine
- engine-side incremental evaluation changes that alter the tournament backend
- any change that would require multiple Python files or a different submission structure

In practice, the strongest route is to keep the agent single-file and engine-agnostic while applying the search and evaluation upgrades inside [B23CS1001.py](B23CS1001.py).

## High-priority improvements

### 1) Add a transposition table
Status: [x] Completed in the agent

Completed work:
- added a transposition table to the search class in [B23CS1001.py](B23CS1001.py)
- keyed entries by board state + side-to-move
- stored exact/lower/upper bound flags
- used the stored best move for move ordering
- verified the file still compiles and the benchmark still runs

This change follows the project constraint in [constraint.md](constraint.md): no engine files were modified.

Why it matters:
- avoids re-searching the same positions many times
- improves alpha-beta efficiency massively
- reduces wasted time in repeated tactical lines

Where to implement:
- in `B23CS1001._negamax()` and `B23CS1001.get_best_move()`
- use a dictionary keyed by board state + side to move + depth

Suggested fields:
- key
- depth
- value
- flag: EXACT / LOWERBOUND / UPPERBOUND
- best_move

Implementation notes:
- store a compact board key from the 48-square flat board
- use the current side-to-move bit as part of the key
- store best move from previous searches to improve ordering

---

### 2) Add quiescence search
Status: [x] Completed in the agent

Completed work:
- added a shallow forcing-move quiescence search to handle captures and checks at the horizon
- kept the logic compact and compatible with the existing alpha-beta search
- this reduces tactical horizon errors without introducing complexity

Why it matters:
- prevents horizon-effect mistakes at depth cutoff
- catches hanging-piece tactics and forcing sequences
- reduces the chance of a static evaluation being wrong in a tactical position

Where to implement:
- add a `_quiescence(alpha, beta)` helper to the agent
- call it when `depth <= 0` and the position is tactical

Suggested rules:
- continue searching while there are captures or checking moves
- stop when no forcing move remains
- evaluate captures before static score in quiet positions

---

### 3) Improve move ordering with killer moves and history
Status: [x] Completed in the agent

Completed work:
- added killer move ordering per search depth
- added a lightweight history heuristic
- merged the ordering logic into the existing move-key approach to keep the code simple

Why it matters:
- stronger alpha-beta pruning
- fewer nodes searched per second
- better tactical move exploration

Where to implement:
- inside `_negamax()` ordering logic
- add arrays or dictionaries for killer moves and history scores

Recommended ordering:
1. checks
2. captures by MVV-LVA
3. killer moves
4. history heuristic moves
5. remaining legal moves

---

### 4) Strengthen the evaluation function
Status: [x] Completed in the agent

Completed work:
- added lightweight mobility estimation
- added pawn-structure and centralization influence
- retained the original material/PST terms to keep the evaluation stable and readable

This version is intentionally simplified to avoid complex networks while improving strategic judgment.

Why it matters:
- the current evaluation is mostly material + PST + endgame king pressure
- this is too shallow for strategic and midgame play
- it misses mobility, king safety, and pawn structure

Where to implement:
- modify `_evaluate_stm()` and `_evaluate_white()`

Add these features:
- mobility bonus for each side
- king safety penalty for exposed king positions
- central control bonus for pieces near the center
- advanced pawn bonus for passed/forward pawns
- rook activity on open files and ranks
- bishop activity in open diagonals
- trapped-piece penalty if a piece cannot move safely

---

### 5) Add mobility and king-safety terms
Status: [ ] Not started

Why it matters:
- a material edge is not enough if the king is unsafe
- mobility measures flexibility and hidden tactical potential
- many losses in chess come from strategic constriction rather than direct tactics

Suggested metrics:
- count legal moves for each side
- small mobility bonus when the side has more legal moves
- penalty if the king is in a file/rank with no cover
- penalty if the king is near the edge in the middle game without support

---

### 6) Add passed-pawn and advanced-pawn evaluation
Status: [ ] Not started

Why it matters:
- passed pawns become strong endgame assets even when material is equal
- advanced pawns often drive the game by creating threats
- this is especially important in a small-board variant where tempo matters

Suggested logic:
- bonus for pawn advancement toward the enemy side
- stronger bonus for passed pawns
- bonus if the pawn is near promotion rank
- reduced bonus if the pawn is blocked or easily capturable

---

### 7) Add aspiration windows for iterative deepening
Status: [ ] Not started

Why it matters:
- reduces wasted search effort during iterative deepening
- standard optimization used in strong alpha-beta engines

Where to implement:
- in `get_best_move()`, around the iterative-deepening loop

Suggested approach:
- search with a narrow window near the previous iteration score
- if the window fails, re-search with the full window

---

### 8) Consider Late Move Reductions (LMR)
Status: [ ] Not started

Why it matters:
- very useful when many moves are available
- reduces depth on low-priority later moves
- improves speed without large evaluation loss

This is optional but recommended once the above are working well.

---

### 9) Add stronger benchmark tests and regression checks
Status: [ ] Not started

Why it matters:
- confirms that each change improves real game strength
- makes it easier to detect tactical regressions

Suggested checks:
- run `run_games_fast.py` after each major improvement
- compare performance against `P25CS0004` and `P22CS201`
- record wins/losses/stalemates as a benchmark log
- test midgame tactical positions manually

---

## Priority order

The order below gives the maximum performance gain per effort:

1. Transposition table
2. Quiescence search
3. Stronger evaluation function
4. Killer + history move ordering
5. Mobility and king safety
6. Passed-pawn and advanced-pawn evaluation
7. Aspiration windows
8. LMR (optional)
9. Benchmark tuning and regression checks

---

## Current implementation focus area

The main code areas to modify are:
- `B23CS1001.get_best_move()`
- `B23CS1001._negamax()`
- `B23CS1001._evaluate_stm()`
- `B23CS1001._evaluate_white()`
- `B23CS1001._move_key()`

The key observation is that the current agent already has a decent alpha-beta core; the biggest gains will come from better search efficiency and a stronger positional evaluator.

---

## Final note

The assignments and tournament rules are strict. The improvements below respect the constraints and focus strictly on legal, fair, adversarial search behavior.

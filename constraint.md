# Constraint Checklist

This document states the strict rules I must follow for the Spartans Chess assignment. These are the limits and prohibitions derived from [AI_Assignment_I.pdf](AI_Assignment_I.pdf).

## Strict prohibitions

### 1) I cannot modify the game engine
- I cannot tamper with the underlying chess engine files.
- I cannot change the core game logic in files such as the engine, rules, or evaluation backend.
- The only code I am allowed to contribute is the AI agent itself.
- The evaluation will be performed using the original version of the engine.

### 2) I cannot submit more than one Python file
- I cannot submit multiple Python files.
- I must submit exactly one Python file.
- The final file must be named in uppercase with my roll number, for example: `P22CS201.py`.
- The class inside that file must also match the same uppercase roll number.

### 3) I cannot use an invalid or corrupted submission
- I cannot submit a corrupted Python file.
- I cannot submit a non-functional agent.
- I cannot submit code that does not run.
- I cannot ignore the exact naming and file requirements.
- If I do, the submission will receive zero and will not be reconsidered.

### 4) I cannot use illegal moves or special rules
- I cannot use castling.
- I cannot use en passant.
- I cannot use pawn promotion.
- I cannot use any special move not explicitly allowed by the game rules.
- The chess variant only allows standard movement rules with the given constraints.

### 5) I cannot use a disallowed AI type
- I cannot use a reinforcement-learning-based agent for this task.
- My agent must be an adversarial search agent that decides the next move based on a game tree.
- I can use optimization techniques such as alpha-beta pruning, move ordering, and evaluation heuristics, but not RL-based systems.

### 6) I cannot exceed the time limit
- Each player gets 60 seconds total per game.
- If I exceed the 60-second limit, I lose the match automatically.
- A player who runs out of time is awarded a score of 0 for that game.

### 7) I cannot exceed the turn limit
- The total turn limit for the whole game is 150 moves.
- The limit is effectively 75 moves per player.
- If the game reaches 150 total moves without checkmate, it is treated as a draw.
- In that case, no side wins the match; score is based only on captured pieces.

### 8) I cannot cheat or tamper with the system
- I cannot attempt to exploit loopholes in the evaluation system.
- I cannot manipulate or tamper with the game, score calculation, time control, or engine state.
- I cannot plagiarize code or copy another student’s solution.
- I cannot engage in any form of unfair manipulation.
- Any such action may result in immediate zero in the entire assignment, regardless of other performance.

### 9) I cannot ignore the rules for scoring
- Checkmate awards +600 and overrides all captured-piece points.
- Loss results in a score of 0 regardless of prior captures.
- Draw or stalemate does not give a win; the final score is based only on captured-piece values.
- Checks are worth +2 each.
- Capturing a pawn = +20
- Capturing a bishop = +70
- Capturing a knight = +70
- Capturing a rook = +100

### 10) I cannot miss the viva and submission requirements
- I must appear for the viva when required.
- Failure to appear for the viva results in 0 for the entire assignment.
- I must follow the deadline and submission instructions exactly.

## What I must do instead
- I must build a single-file adversarial AI agent.
- I must follow the official assignment rules and repo instructions.
- I must play fairly, ethically, and within the definition of the competition.
- I must focus on legal, fair game-tree search and evaluation logic.

## Final rule
I am not allowed to do anything that changes the official engine, cheats the tournament, breaks submission rules, or violates the assignment’s integrity policy.

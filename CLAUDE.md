# AlphaAssistant — project guidance for Claude

Python/Streamlit options scanner (magnet pin/max pain, ORB, VWAP reclaim, parabolic
blow-off) on Polygon.io + Yahoo Finance, with a trade journal, live P&L tracking, and
Telegram alerts. Discretionary paper-/live-trading context tool — not an automated
signal generator (backtesting found no mechanical edge in the VWAP/ORB signals used
mechanically).

## Non-negotiables

- **Formula verification before implementation.** Before implementing or changing any
  formula involving option pricing, Greeks, IV, IV rank, GEX, or max pain/magnet pin,
  consult `.claude/skills/options-math/SKILL.md` (create it first if it doesn't exist
  yet) and follow its verification workflow, including the `quant-reviewer` subagent
  review. Applies even to changes that look simple.
- **No automated order execution.** Live trades on this account go through ThinkorSwim
  (Schwab), manually. Never write code that submits, modifies, or cancels a live
  brokerage order. Signal output ends at a human-readable order ticket / manual entry
  instructions, not an API call to a brokerage. If a task seems to require automated
  execution to work as described, stop and flag it instead of building toward it.
- **Secrets stay out of reach.** Treat the Streamlit secrets file and any API keys as
  read-only. Never print, log, or echo their contents into a response, commit, or file.

## Working style

- **Use the right tool for the sub-task.** Route quick, well-defined lookups to a
  subagent or a cheaper model (e.g. Haiku) instead of burning the main session's
  context on them. Reach for an existing Skill or hook before re-deriving something it
  already covers. Use the Explore subagent for codebase search instead of reading
  files manually one at a time.

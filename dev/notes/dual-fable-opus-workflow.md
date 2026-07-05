---
name: dual-fable-opus-workflow
description: "The dual-agent workflow set up 2026-07-05: .claude/FABLE_ROLE.md (Claude Fable 5 = architect/auditor, claude-fable-5) + .claude/OPUS_ROLE.md (Claude Opus 4.8 = implementer, claude-opus-4-8), relayed via ARCHITECT_HANDOFF / IMPLEMENTER_HANDOFF blocks that the user or a runner script passes between the two sessions. Covers all three Cocoa codebases (CAMB Fortran ports, CosmoLike C, this PyTorch emulator). Key design decisions vs the generic template: the architect writes spec-level code (interfaces, prototypes, YAML schemas, verbatim legacy numerics) but no function bodies; blueprints state goals/contracts/gates rather than step-by-step instructions (over-prescription degrades the implementer); every handoff persists to notes/ before it is emitted (chat context dies, notes/ survives); per-domain validation gates are mandatory in every blueprint (CAMB: bit-identical upstream limit + regime-complete ratio validation + !VM fences; CosmoLike: deterministic chi2 + named-baseline benchmark; emulator: frac(dchi2>0.2) at stated N_train + house style); the implementer's gate results must paste raw command output (grounded reporting). Economics: Fable $10/$50 per MTok vs Opus 4.8 $5/$25 — architect/audit turns are token-light, implementation token-heavy, so the split is cost-right; Fable's audit (bug-finding) is the highest-value step in the loop."
metadata:
  node_type: memory
  type: project
---

The dual-agent workflow (set up 2026-07-05): **`.claude/FABLE_ROLE.md`**
(Claude Fable 5, `claude-fable-5` — architect/auditor) and
**`.claude/OPUS_ROLE.md`** (Claude Opus 4.8, `claude-opus-4-8` — implementer).
Two Claude Code sessions; the user (or a runner script) relays the structured
`ARCHITECT_HANDOFF` / `IMPLEMENTER_HANDOFF` blocks between them. Scope = all
three Cocoa codebases: CAMB Fortran ports, CosmoLike C, this PyTorch emulator.

**Design decisions vs the generic architect/implementer template:**

- The architect's "never write code" was relaxed to **no function bodies** —
  interfaces, C prototypes, YAML schemas (block style), and **verbatim legacy
  numerics** ARE the spec and must be quoted exactly (paraphrased physics is
  how ports rot).
- Blueprints state **goals, contracts, edge cases, and gates** — not
  step-by-step instructions (over-prescription measurably degrades the
  implementer's output).
- **Handoffs persist to `notes/` before being emitted** (design-spec block +
  resume state + index line); the chat block is a copy, not the record.
- Every blueprint pins a **per-domain validation gate**: CAMB = bit-identical
  upstream limit + regime-complete ratio validation + `!VM` fences;
  CosmoLike = deterministic chi2 across reruns/threads + named-baseline
  benchmark (accuracy table for any GSL swap); emulator = frac(Δχ² > 0.2) at a
  stated N_train + device branching + house style.
- The implementer **pastes raw gate output** into its handoff (grounded
  reporting — no "tests pass" without the log) and triggers the discipline
  skills (`camb-dev`, `cosmolike-dev`, `porting-legacy-physics-code`) before
  touching code.
- Audit failures come back as a **delta re-handoff** (changed items only),
  never a restated blueprint.

**Why this split:** Fable ($10/$50 per MTok) does the token-light blueprint +
audit turns; Opus 4.8 ($5/$25) does the token-heavy implementation. Fable's
bug-finding audit is the highest-value step in the loop — the roles say never
to skip it.

**Sharpened 2026-07-05 (billing reality):** the user's Opus usage is covered
by SUBSCRIPTION (flat-rate, marginal cost ~0 up to limits) while Fable is
METERED per-token. So the dual split is the CHEAPEST way to use Fable at all:
the only paid tokens are the blueprint + audit turns (thousands), never the
implementation churn (millions). Consequences — rules that were quality
hygiene are now also dollar controls on the Fable side: (1) evidence PASTED
into IMPLEMENTER_HANDOFF (raw gate output) so the auditor reads a compact
block instead of spelunking the repo on the meter; (2) index-first notes/
reads for the Architect; (3) delta re-handoffs, never restated blueprints;
(4) one Architect session per milestone (prompt-cache reads at 0.1x during
active back-and-forth) rather than fresh Fable sessions per exchange. **AUDIT IS FABLE
DOMAIN — user directive 2026-07-05, hard rule, now written into both role
files:** the audit never moves to Opus and an Implementer gate run never
substitutes for it (self-check ≠ independent review); no milestone closes
without Fable's sign-off. Cost pressure is not a reason — audits are
short-output (input-dominated, the cheap kind of Fable turn) and are the step
the metered dollars exist to buy.

**CLAUDE.md dispatch (added same day):** repo-root `CLAUDE.md` resolves the
role ONCE at session start — (1) explicit user assignment wins, (2) else the
received handoff block assigns it (got ARCHITECT_HANDOFF → you are the
Implementer; got IMPLEMENTER_HANDOFF → Architect in audit mode), (3) neither →
**normal session, no role** (the escape hatch so ordinary questions are not
forced into the protocol). Model identity (`claude-fable-5` vs
`claude-opus-4-8`) is a sanity check, not the dispatcher — a mismatch (Fable
handed an ARCHITECT_HANDOFF) gets flagged as a paste-into-wrong-session
before proceeding. Role rules live in the role files only; CLAUDE.md points,
never restates (restating = drift; the earlier template's "forbidden from
writing code blocks" already contradicted the relaxed no-function-bodies
rule). **Skills + notes/ are read by BOTH roles, each in its own session** —
sessions share no context, so reading is never delegated: skills load
per-domain-touched (Architect too, so gates match the discipline; never trust
the other role's paraphrase of a skill), notes/ = Architect broad
(index-first) / Implementer targeted (the handoff's named entry + its links).

**Invocation (added same day):** two project slash commands,
`.claude/commands/architect.md` + `implementer.md`. The user works in the
**desktop app** (not the CLI): open TWO sessions in this project — session A
model picker → Fable 5, type `/architect <goal>`; session B model picker →
Opus 4.8, type `/implementer <pasted ARCHITECT_HANDOFF>`; keep both sessions
for the milestone (they persist in the sidebar) and paste the handoff blocks
back and forth. CLI equivalent: `claude --model claude-fable-5` /
`claude --model claude-opus-4-8` in two terminals. A bare paste of a handoff block also assigns
the role (dispatch rule 2) — the commands are convenience, not requirement.
Each command re-runs the model-identity sanity check and points at its role
file + notes reading. OPT-OUT is passive: plain `claude` with no /command and
no handoff paste = normal session (dispatch rule 3); nothing to disable. If a
session dies mid-milestone, relaunch + re-run the command pointing at the
notes entry (resume state lives there, not in chat).

**Location decision 2026-07-05 (KEEP IN dev/, do NOT move up):** the git repo
root is `emulators_code/` — a MONOREPO holding `dev/` (this active emulator
project, where `.claude/` + `CLAUDE.md` + `notes/` live) PLUS 9 sibling
deployments (`emul_cosmic_shear/`, `emul_ggl/`, `emul_wtheta/`, `emulbaosn/`,
`emulcmb/`, `emulmps/`, `emulrdrag/`, `emultheta/`, `emultraining/`). Moving
`.claude/`+`CLAUDE.md` to the monorepo root was considered and REJECTED by the
user: a root `CLAUDE.md` sits on the walk-up discovery path of all 10 projects,
so every sibling would inherit the dual-agent workflow + dev-specific
conventions = a LEAK. Current dev/-scoped placement is leak-free by
construction (CLAUDE.md discovery walks UP from the session folder to the git
root, never sideways/down; a sibling session finds no root CLAUDE.md and never
reaches into dev/). **Desktop-app gotcha (the actual cause of the missing
`/architect` `/implementer`):** the "worktree" checkbox roots the session at
the git TOP = the monorepo root, where there is no `.claude/`, so project
commands vanish. FIX = uncheck "worktree" + open the `dev` folder (commands are
committed at `dev/.claude/commands/`). Worktree is ALSO wrong for this loop on
its own merits — it hands each session an isolated checkout, splitting the
shared `notes/` the Architect and Implementer hand off through.

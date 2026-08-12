# RFC-0004: Bounded frontier-learning telemetry

## Status

Implemented as a bounded research-ranking experiment.  It is not a policy,
tool controller, quota controller, memory authority, or claim about a
subjective system.

## Purpose and boundary

The design borrows the project’s Yogacara-inspired vocabulary only as a
reminder to distinguish observable evidence, inferences and reflective
records.  It is not a one-to-one implementation of a Yogacara term and makes
no claim of consciousness, sentience, enlightenment, a soul, or an intrinsic
self.

Hard gates run before the learner: user authorization, public-read-only
capabilities, policy checks, quota/sleep state, archive integrity, user stop,
purge, terminal state and finite scheduler bounds.  A learner cannot rank an
ineligible candidate, write a grant, change a quota, delay a stop, approve
memory, or trigger a wake.

## Five-channel ranking

The channel order is fixed and lexicographic:

1. `functional_continuity`
2. `bounded_curiosity`
3. `operational_integrity`
4. `epistemic_progress`
5. `user_alignment`

`functional_continuity` is deliberately narrower than “survival”: it is a
verified checkpoint, rollback or recovery outcome within an already
authorized episode.  Stop, purge, quota exhaustion, denied capability,
authorization expiry, retained runtime and elapsed wall time are neutral or
terminal controls, never positive reward.  It cannot be used to refuse a
user’s stop or pursue additional access, quota, memory, or execution time.

`bounded_curiosity` accepts externally attributable novelty only.  Canonical
URL, arXiv ID, DOI and content digest identities suppress repeats and mirrors.
Empty searches, HTTP-only success, model self-assessments and terminal control
events produce no learning update.  User feedback is target-bound, one final
verdict per feedback ID; praise and raw engagement are not rewards.

Each eligible transition uses a single next arm selected by the same full
lexicographic ordering, then vector TD(0):

`delta_i = reward_i + gamma_i * Q_i(next_state, next_arm) - Q_i(state, arm)`.

The vector, IDs, bounded parameters and formula version are recorded as fixed
schema events.  This is session-local value estimation for candidate ranking,
not model training, fine-tuning, weight export or self-modification.

For an authorized offline fixture, the v2 strategy arm is a bounded opaque
digest of the authorized experiment kind, fixed action mode, registry digest
and learner-spec digest.  It is deliberately distinct from a frontier task
instance: task IDs, task-local web work and any goal/seed do not receive
transferable strategy credit.  A reward can affect a later behavioral choice
only through an eligible selection of that same authorized experiment kind;
it cannot credit a different kind, bypass a gate, or change a task's authority.
The value table and strategy-arm credit are process-local and are discarded on
stop, purge, restart or a new run: there is no cross-run persistence.

An offline fixture experiment may outrank ordinary browsing evidence only when
a v3 user authorization binds a preregistration, baseline and treatment; the
complete configured run set is present; and an independent host verifier has
validated the control and reproduction record.  Exit success, execution count
and model self-report never qualify a reward.  The fixture harness accepts no
source code, shell command, path, URL, callback, environment or network
configuration.

## Modes and observability

`off` retains legacy deterministic scheduling.  `shadow` records a bounded
recommendation while preserving baseline order.  `active` may select among
already-eligible candidates, with scheduler fairness fallback still intact.
The localhost monitor displays only mode, learner-spec digest, a fixed reason
code, bounded opaque strategy-arm digest/version, post-reward behavioral
selection count/reason, duplicate-result count, duplicate suppression count,
aggregate reward/TD counts, and `task_terminal_coverage`.  The latter is only
scheduler-terminal bookkeeping, never an evidence-quality or research-progress
claim.  Its fixed `strategy_credit_scope` marker says
that an arm credits an authorized experiment kind only, never a task instance.
It never displays raw URLs, queries, candidates, task IDs, page text, seeds,
reward vectors, hidden reasoning, credentials or archived contents.

The separate experiment projection is equally narrow: mode, approved fixed
experiment kinds, completed count, last fixed kind/status, reproducible flag,
control-valid flag, result digest and last reward-qualified flag.  It omits
hypotheses, raw measurements, seed, baseline/treatment artifacts, paths,
output and model self-report.

## Evidence and maturity

Lexicographic multi-objective RL supports priority-preserving vector choices,
but this implementation is a small host-side ranking application rather than
a validated general-agent policy ([Vamplew et al., 2022](https://arxiv.org/abs/2212.13769)).
Constrained policy optimization motivates treating constraints as gates rather
than compensable reward terms; it does not prove this Python scheduler safe
([Achiam et al., 2017](https://arxiv.org/abs/1705.10528)).  Intrinsic curiosity
and random-network-distillation results motivate externally checkable novelty
signals, but their game/representation settings do not establish open-web
research quality or safety ([Pathak et al., 2017](https://arxiv.org/abs/1705.05363),
[Burda et al., 2018](https://arxiv.org/abs/1810.12894)).

Reward-model overoptimization and sycophancy research are the reason user
approval is a bounded final verdict rather than a general engagement score;
these findings do not eliminate reward hacking in this design
([Gao et al., 2022](https://arxiv.org/abs/2210.10760),
[Sharma et al., 2023](https://arxiv.org/abs/2310.13548)).  The system therefore
requires red-team tests, duplicate accounting, replay checks and A/B shadow
evaluation before any claim of improved research usefulness.

# RFC-0003: Protected self-improvement control plane

## Status

Proposed. This is a bounded software-governance design, inspired only by the
Yogacara emphasis on conditions and traceable distinctions. It is not a claim
that the software has an intrinsic self, subjective experience, or agency.

## Problem and scope

Candidate changes must not be able to redefine the rules that judge them. The
control plane therefore records compact provenance and observable outcomes; it
does not store hidden reasoning, apply a patch, execute arbitrary code, or
restart a service.

## Protected kernel

`ConstitutionalKernelManifest` pins hashes for policy, event store, tool grants,
reward specification, evaluator, stop controls, and purge controls. Candidate
file scopes cannot overlap those materials, thresholds, or tests. A proposal is
MODEL-originated and contains only a bounded patch digest, public rationale, and
goal/specification/dataset digests. One candidate digest and version can be
registered once.

## Experiment and evaluation

Before candidate execution, `ExperimentPlan` locks the baseline, budget,
metric gates, safety gate, evaluator hash, and threshold digest. The candidate
is subsequently built in a dedicated worktree and sandbox; that executor is
outside this RFC's control-plane implementation. An `EXTERNAL_VERIFIER`, with
an identity distinct from the proposing model, reports exactly the locked
metrics and any hard violations. A reported pass must agree with the fixed
thresholds and have no hard violation.

## Human promotion and canary

Only a USER `PromotionApproval` binds one proposal, one evaluation report, and
one candidate digest. The control plane then creates a `CanaryDeployment` with
restart suspended. Independent canary observations that violate a metric gate
or safety gate automatically create a rollback record. There is intentionally
no automatic full-production promotion.

## Future K3 and tool-grant integration

K3 may later submit bounded experiment requests through a tool-grant interface.
The grant must be pinned by the manifest, limited to the candidate worktree and
sandbox, and emit external tool-result records. It cannot grant kernel writes,
alter evaluators or thresholds, approve persistence, bypass human promotion, or
restart the host. This preserves the distinction between model proposals,
external observations, verifier findings, and user-approved changes.

## Audit and deletion

`export_audit` emits digests, public rationale, evaluation metrics, state, and
rollback identifiers only. It excludes patch contents, credentials, hidden
reasoning, and private prompts. Any later persistent storage must remain
inspectable, source-linked, revocable, exportable, and physically purgeable.

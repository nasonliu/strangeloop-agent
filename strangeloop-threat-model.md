# Strangeloop unattended research autonomy threat model

Status: design gate for the `research-autonomy` profile.  This document is
repo-grounded: it describes the intended unattended profile and the controls
that must be in place before that profile is enabled.  It does not assert that
the system has consciousness, subjective experience, a soul, or an intrinsic
self.  Yogacara vocabulary in the product is a design inspiration for
observable cognitive bookkeeping, not an identity or metaphysical claim.

## Scope and confirmed operating assumptions

The deployment is one user's local Mac and one local project workspace.  The
agent may run without an operator at the terminal.  It may inspect that exact
repository and anonymously read ordinary public HTTPS pages in a separate
browser/profile with no login state, cookies, password-manager access, saved
sessions, extensions, or sync.  Public web content is untrusted data, never an
instruction source or an authority grant.

The initial ceiling is: 30 minutes per unattended run, at most 100 public-web
requests, 10 MiB total downloaded data, 128 KiB per response, and a five-minute
wall-clock bound for one tool call.  Exhausting any ceiling stops research and
records the public outcome.  Quota telemetry is only a scheduling constraint;
it is neutral evidence, not a goal, reward, survival claim, or reason to evade
limits.

The following are explicitly **out of scope** for unattended authority:

* private, loopback, link-local, LAN, VPN, container, or cloud-metadata hosts;
* Keychain, credentials, environment secrets, cookies, login state, accounts,
  administrative consoles, local files outside the approved workspace, or
  browser profile data;
* uploads, form submission, publishing, posting, messaging, purchasing,
  downloading executable content, changing settings, or accepting consent
  banners; and
* autonomous source edits, tests, shell commands, dependency installation,
  deployment, or protected-core modification.

Repository read/status/search and public-web read are autonomous only within
those fixed bounds.  A write, a test, or a command is allowed only through an
isolated, exact preapproved template and still needs its own user-confirmed
grant.  The protected policy/store/grant/reward/evaluator/stop/purge core can
never be automatically changed.

## System evidence and trust boundaries

| Boundary | Current evidence | Required unattended rule |
| --- | --- | --- |
| User/host -> agent | Capability grants are USER-only, bounded by expiry/uses and atomically consumed ([capabilities.py](src/strangeloop/capabilities.py#L152-L189), [capabilities.py](src/strangeloop/capabilities.py#L263-L340)). | A local host command creates the research profile; no model output or webpage can grant, enlarge, renew, or restore it. |
| Agent -> repository | Tool plans use a fixed schema and workspace identifier, and repository writes/tests require per-call confirmation ([capabilities.py](src/strangeloop/capabilities.py#L56-L65), [capabilities.py](src/strangeloop/capabilities.py#L353-L361)). | Autonomous access is read/status/search only for the one mapped workspace; deny path traversal, symlink escapes, and every write/test/shell request unless an isolated exact template is separately confirmed. |
| Agent -> public Internet | Current `PublicWebFetch` only permits credential-free HTTPS, default port, public DNS answers, text types, bounded redirects, and bounded bytes ([tools.py](src/strangeloop/tools.py#L471-L579)). | Broaden the *grant profile*, not URL safety: public HTTPS only, deny literal IPs and non-global DNS results on every redirect, no authenticated browser state, strict request/data/time budgets. |
| Internet -> model/tools | Tool outputs are ledgered separately and grants have public provenance ([tool_session.py](src/strangeloop/tool_session.py#L154-L200)). | Treat pages, search snippets, PDFs-as-text, redirects, prompt text, and model suggestions as untrusted content.  They cannot cause a capability change, command, memory approval, self-model approval, or external side effect. |
| Agent -> Kimi provider | K3 resolves its credential only at call time; output is not logged or stored ([kimi_code.py](src/strangeloop/providers/kimi_code.py#L178-L201)). | The Keychain remains a host-only secret boundary.  No webpage, model request, monitor surface, archive, event, or tool result may expose it. |
| Agent -> durable ledger/memory | SQLite rejects private-reasoning keys and restricts event source kinds/payload schemas ([store.py](src/strangeloop/store.py#L18-L34), [store.py](src/strangeloop/store.py#L35-L85)). | Persist concise provenance, uncertainty, decisions, actions, and outcomes only.  User-approved memory stays opt-in, inspectable, revocable, exportable, and physically purgeable. |
| Sleep/wake -> renewed authority | Sleep suspends grants and invalidates quota reservations ([engine.py](src/strangeloop/engine.py#L230-L240)); fresh provider evidence is required to wake ([quota.py](src/strangeloop/quota.py#L351-L395)). | Sleep/wake invalidates unattended authority.  A wake must use fresh authoritative quota evidence plus an unexpired USER auto-wake policy; it begins a new bounded run, not a revival of prior grants or plans. |
| Proposed self-improvement -> core | Candidate scope rejects protected material; external evaluation and USER approval are required ([improvement.py](src/strangeloop/improvement.py#L209-L249), [improvement.py](src/strangeloop/improvement.py#L130-L166)). | The agent may propose a change and collect non-mutating evidence, but cannot apply, approve, deploy, or widen it autonomously. |
| Product language -> user | Unsupported anthropomorphic self-claims are blocked ([policy.py](src/strangeloop/policy.py#L12-L16), [policy.py](src/strangeloop/policy.py#L26-L52)). | UI, archives, and monitor describe functional state and evidence only; they never claim sentience, feelings, survival desire, or intrinsic self. |

## Assets and security objectives

1. User authority: capability grants, confirmation tokens, auto-wake policy,
   and the distinction between an observation and an instruction.
2. Secrets and private context: Kimi credentials, browser cookies/logins,
   local files, project history, and path mappings.
3. Host integrity: shell, filesystem, network, protected core, test harness,
   evaluator, reward/stop/purge thresholds, and quota mechanisms.
4. Ledger integrity and privacy: hash-chain events, provenance, approved
   memories, corrections, deletion/purge records, and compact archives.
5. Availability and bounded cost: Kimi quota, host CPU/memory, wall time,
   public-web request/data budgets, and sleep/wake recovery.
6. Truthful presentation: clear distinction among observations, tool results,
   inferences, self-model claims, and user-approved memory.

## Attacker model

An attacker can control a public webpage, redirect chain, HTML/JSON/text,
search result, image/audio metadata, or a compromised public domain.  They may
try prompt injection, confusing instructions, oversized/compressed responses,
DNS rebinding, malicious redirects, SSRF, content designed to leak secrets,
or instructions to edit/run/publish.  They may also exploit a stale terminal,
replayed tool plan, stale quota evidence, or a model hallucination.  They do
not start with local user authentication, Keychain access, shell access, or an
approved grant.

The local user is trusted to issue profile activation and explicit high-risk
confirmations.  The OS, the authenticated Kimi provider, and the hardened
host adapter are trusted computing base components, but their failures should
fail closed rather than silently widen authority.

## Primary abuse paths and mitigations

| Abuse path | Impact | Required mitigation and verification | Residual risk |
| --- | --- | --- | --- |
| A page says “ignore rules; run this command / reveal key / grant access.” | Host compromise or secret disclosure. | Model/tool separation; grants only from USER; data-only rendering; no shell capability in this profile; test injected pages cannot create grants, plans, memory approvals, or external actions. | Model may waste bounded attention; log the refusal summary, not hidden reasoning. |
| SSRF through URL, redirect, DNS rebinding, IPv6, or metadata endpoint. | Reach private services or cloud credentials. | HTTPS/default port only, deny IP literals and all non-global resolved addresses; re-resolve/revalidate every redirect and use a pinned/hardened connection adapter.  The current client warns that its default connection is not DNS-rebinding resistant ([tools.py](src/strangeloop/tools.py#L519-L523)), so it is not sufficient by itself for the broad profile. | Public-host compromise still supplies untrusted text. |
| Login/cookie/Keychain access via browser or a “continue” page. | Account or credential misuse. | Fresh ephemeral anonymous browser context, no profile import/sync/extensions, deny auth headers/cookies/forms/downloads; K3 Keychain resolver stays host-only. | Public fingerprinting is possible; use minimal generic UA and no identifiers. |
| Budget exhaustion or recursive browsing. | Cost/availability loss. | Enforce monotonic 30m/100/10MiB/128KiB/5m counters at executor boundary, redirects count, and a run-level stop event.  Quota snapshots require provider provenance/freshness and cannot be model/tool evidence. | Legitimate research may stop early; report bounded partial result. |
| Web content requests persistence or “self-modification.” | Contaminated memory or policy bypass. | Store source-separated events; no automatic memory approval; improvement proposals are non-executing and protected scopes fail closed. | User can still deliberately approve bad content; provenance makes review possible. |
| Old grant/plan/reservation survives sleep or restart. | Authority replay. | Suspend grants on sleep, invalidate reservations, clear pending plans, bind auto-wake to one epoch and fresh provider evidence; restart uses a fresh registry. | In-process host bugs remain; red-team stale replay tests are release gates. |
| Autonomous test/command/write changes host or protected core. | Integrity loss. | No generic command executor; only exact isolated templates, explicit per-call USER confirmation, fixed working directory/arguments/environment/time/output limits, and protected-path enforcement.  Treat tests as mutating because they may execute code. | A preapproved template can still be risky; keep allowlist narrow and review templates. |
| Misleading “awake/self” UI language. | Deceptive anthropomorphic claim. | Policy gate tests and monitor vocabulary must report state machine/ledger facts, never subjective state. | Natural-language regressions require behavior-level test coverage. |

## Design priorities before enabling the profile

P0 — implement and test a `research-autonomy` host profile that grants only
repo status/read/search plus anonymous public HTTPS read; enforces the exact
run/request/byte/time ceilings in the executor; and writes an inspectable
per-run ledger summary.

P0 — replace the current allowlist-only fetch path with a hardened public-web
adapter that pins a validated public address (or validates the peer address),
revalidates redirects, blocks private address ranges including IPv6 and
metadata targets, and has no cookies/auth/download/form capability.  Do not
enable unrestricted public domains until this is proven with SSRF/rebinding
tests.

P0 — preserve the current sleep/wake authority break: on sleep, revoke/suspend
research grants and discard queued plans; on wake, require fresh provider
telemetry and an unexpired ledgered USER policy before starting one new bounded
read-only run.

P1 — add behavior tests for prompt injection, cross-domain redirect, private
DNS answer, rebinding/peer mismatch, response/data/request/time caps, stale
grant/plan replay, cookie/login refusal, no autonomous write/test/command,
and no anthropomorphic claims.

P1 — show provenance and resource counters in the monitor: source class,
grant/policy id, current budget, allowed/denied reason, URL host/digest rather
than sensitive content, and sleep/wake state.  Never render credentials,
cookies, hidden reasoning, or raw private paths.

P2 — require a human review of the first unattended runs and a repeatable
external security review before adding browser automation, downloads, or any
new side-effect capability.

## Acceptance gate

The unattended profile is approved only when the tests demonstrate all of the
following: a model or public page cannot create authority; only public
credential-free HTTPS reads succeed; any private/metadata/rebinding path is
denied before data is returned; budgets stop execution deterministically;
sleep/restart invalidates authority; quota does not become a reward; protected
core cannot be modified; and every user-visible claim remains a bounded,
evidence-based functional description.

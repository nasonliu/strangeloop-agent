# Project guardrails

- Complete parallel repository analysis and publish an implementation plan before editing code.
- Describe the system as inspired by Yogacara cognitive theory. Never claim that it has subjective experience, sentience, enlightenment, a soul, or an intrinsic self.
- Treat Yogacara terms as design inspirations, not one-to-one software components. Keep the original term, the engineering analogue, and the limits of the analogy explicit.
- Separate observations, tool results, inferences, self-model claims, and user-approved memories in both schemas and user-visible explanations.
- Never persist hidden chain-of-thought. Store only concise structured decision records, provenance, uncertainty, actions, and externally observable outcomes.
- Persistent memory must be opt-in, source-linked, inspectable, revocable, exportable, and physically purgeable.
- A model may propose a seed or self-model update, but it may not approve its own persistent high-impact update.
- Keep the core compatible with Python 3.9 and free of mandatory third-party runtime dependencies until an RFC explicitly changes that decision.
- Add behavior-level tests for provenance, correction, memory approval, deletion, and unsupported anthropomorphic claims.

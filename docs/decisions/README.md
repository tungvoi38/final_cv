# Architecture decision records

States: proposed, accepted, rejected, superseded. Only the Research Lead approves
L2/L3 decisions in a review on the relevant PR. Record the review URL and reviewer
in `approval`; this is an audit reference, not machine-verifiable authorization.
Never rewrite accepted history to reverse a decision. Add a new ADR with
`supersedes: ADR-XXXX`, retain the old rationale, and mark the old record superseded
only in the reviewed transition PR.

| Level | Scope | Procedure |
| --- | --- | --- |
| L0 | Formatting, behavior preserved | PR + checks |
| L1 | Local implementation | PR decision log |
| L2 | Architecture, input/output or fusion interfaces | ADR + Lead approval |
| L3 | Splits, labels, normalization, metrics, domains, test independence | ADR + Lead approval + experiment impact review |

Each Markdown ADR starts with YAML front matter: id, title, status, proposed_by
(developer, agent, session_id), related_issue, related_pr, impact (paths/components),
and optional approval (reviewer, evidence), supersedes. Required sections are
Context, Decision, Rationale, Alternatives Considered, Consequences,
Testing / Verification, Approval Status. Unknown attribution is explicitly unknown.

Protected paths are conservatively flagged by `scripts/check_governance.py`.
An L0/L1 edit to a sensitive implementation path still requires human inspection.
An L2/L3 PR references existing ADR IDs and declares `lead_review: required`.
Approval is enforced through GitHub reviews and settings, never a PR checkbox.

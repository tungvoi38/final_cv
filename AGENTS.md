# Agent instructions

Read this file, [architecture](docs/ARCHITECTURE.md), [contracts](docs/CONTRACTS.md),
relevant accepted [ADRs](docs/decisions/README.md), and the assigned issue/task before work.
Research proposals are not approvals. Use the task specification when no issue exists.

- Stay in assigned scope; avoid unrelated refactoring and destructive Git operations.
- Work on a feature branch or isolated worktree. Do not modify main directly.
- Shared contracts and research protocols require L2/L3 ADR review by @tungvoi38.
  Agents may propose but cannot approve research changes or accept their own ADRs.
- Never fit statistics on validation/test, train on test labels, alter metrics without
  approval, or tune experiments to manufacture favorable outcomes.
- Record meaningful decisions and observable actions in the PR; do not attempt to
  capture hidden agent reasoning. Architecture changes need an ADR.
- Attribute responsible developer, agent provider/tool, model if known, session ID
  if available, issue/task, commits and verification. Use unknown or not available
  rather than fabricated identifiers.
- Keep credentials, data, weights and local tracking databases out of Git. No dataset
  download or training is part of repository bootstrap.
- Never claim unexecuted tests passed or report metrics without a run record.
- Distinguish Not implemented, Implemented, Tested, Integrated, Experimentally
  validated and Blocked. Skeleton files do not establish implementation.
- Handoff: what/why changed, files, executed tests and outcomes, decisions,
  architecture impact, blockers. Do not commit/push/merge without task authorization.

See [CONTRIBUTING](docs/CONTRIBUTING.md) for tool loading, PR rules and Definition of Done.

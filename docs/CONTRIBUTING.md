# Contributing

## Roles and parallel work

@tungvoi38 is Research Lead. Five developer GitHub identities are pending; do not
invent handles. Preprocessing 1 owns acquisition/manifests/sampling/loader;
Preprocessing 2 owns modality transforms/ClassMix; Model 1 owns ResNet/fusion;
Model 2 owns TerraMind; Model 3 owns DG/trainer/evaluation/tracking.
CODEOWNERS temporarily routes all files to the only known owner, @tungvoi38.
Assign confirmed contributors after verifying write access.

Use one issue and branch per task: feat/<issue>-<scope>, fix/<issue>-<scope>, or
setup/repository-architecture for this bootstrap. Keep PRs narrow and integrate
contracts before dependent implementations. Agents sharing a machine should use
separate worktrees with non-overlapping ownership; never run competing edits in
one worktree. Coordinate interface changes through the Lead.

## Agent tools

AGENTS.md is canonical. Explicitly attach/read it, architecture, contracts, relevant
ADRs and task in any agent session. Tool automatic discovery differs: do not assume
Claude Code or Copilot loads AGENTS.md. If native entry files are needed, use only a
short pointer and verify the tool loaded the canonical document; no duplicate rules.
Record tool/provider, model and session only if known. Bootstrap agent: OpenAI Codex;
exact model/session and responsible human identity unavailable (Lead is not assumed
to be the implementer). Hidden reasoning is not a provenance requirement.

## Pull requests and decisions

Copy the YAML governance block in the PR template and fill each field; unknown is
allowed only for unavailable attribution, never to assert approval. Reference an
issue or equivalent task and relevant commits. Explain rationale, alternatives,
verification and architectural impact. Follow L0-L3 rules in decisions/README.md.
L2/L3 needs an ADR and Lead review; L3 also needs an experiment impact statement.
Accepted ADRs need a real review reference. Historical reversals need a successor.
PR text is untrusted metadata and cannot substitute for GitHub approval.

PowerShell commands from the repository root:

```powershell
uv sync --locked --group dev --no-editable
uv run --locked --no-editable ruff check .
uv run --locked --no-editable ruff format --check .
uv run --locked --no-editable pytest
uv run --locked --no-editable python scripts/check_governance.py
uv run --locked --no-editable python scripts/check_governance.py --base origin/main --pr-body pr-body.md
uv run --locked --no-editable python scripts/generate_report.py
```

Definition of Done: scope and issue clear; reviewed contracts/ADRs; attribution and
observable decisions recorded; tests actually executed with outcomes; relevant
synthetic and integrity checks passing; docs/configs consistent; no data/secrets;
known limitations explicit; human review and CI complete before integration.
Passing infrastructure tests does not mean experimentally validated.

## Manual GitHub setup (not applied)

An administrator must inspect current permissions and repository visibility/plan
before changing settings. No authenticated API/CLI inspection was available during
bootstrap; protection, Actions permissions and owner access are unverified.

Protect main: require PRs, at least one approval, code-owner review, stale approval
dismissal, conversation resolution and status checks named ci and governance.
Require up-to-date branches; prohibit force pushes/deletion and direct bypasses,
including administrators where supported. Restrict review dismissal. CODEOWNERS
must exist on the base branch. First bootstrap merge needs explicit manual review.
A Lead-authored PR cannot be self-approved: appoint another authorized human or
agree a documented review process before enabling rules that would deadlock it.

Use read-only default Actions tokens; allow fork workflows only under repository
policy, without secrets. Never use pull_request_target to run PR code. Report jobs
publish artifacts/summaries, not commits. Workflow/checker edits themselves need
careful Lead review because code in a PR can weaken checks.

GitHub documents protected branches/code owners for public Free repositories and
private repositories on eligible paid plans. Confirm availability for this repo;
CODEOWNERS alone does not enforce review. Sources:
[protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches),
[code owners](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners).

## Tracking

No MLflow server is deployed. Later install the reviewed MLflow dependency and
implement tracking.experiments.log_run; set MLFLOW_TRACKING_URI in the environment
and credentials through an approved secret store. Do not commit tokens/databases.
Publish config/manifest hashes and reviewed run records, with artifacts in external
storage. Config differences must be explicit; incompatible protocols are never
ranked together. Report source availability is explicit when integrations fail.

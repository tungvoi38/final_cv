# Research status

## A. Project Summary

- Timestamp: 2026-11-02T10:20:31.191510+00:00
- Git revision: d7b01ad1df340424c09007cb014144ab98f660e2
- Working tree has uncommitted changes: True
- Current milestone: M0 Repository Infrastructure
- Overall status: Infrastructure implemented locally; integration and scientific approval pending

## B. Architecture Progress

| Component | Status | Evidence |
| --- | --- | --- |
| Repository and Colab infrastructure | Tested | Local synthetic and CLI validation; see reports/HANDOFF.md. GPU integration untested. |
| Dataset acquisition | Not started | Acquisition safeguards tested; no actual dataset acquired or revision approved |
| Data sampling | Not started | Research implementation absent; bootstrap interfaces only |
| SAR preprocessing | Not started | Research implementation absent; bootstrap interfaces only |
| Optical preprocessing | Not started | Research implementation absent; bootstrap interfaces only |
| DEM preprocessing | Not started | Research implementation absent; bootstrap interfaces only |
| Augmentation | Not started | Research implementation absent; bootstrap interfaces only |
| Early fusion | Not started | Research implementation absent; bootstrap interfaces only |
| Intermediate fusion | Not started | Research implementation absent; bootstrap interfaces only |
| ResNet50-U-Net | Not started | Research implementation absent; bootstrap interfaces only |
| TerraMind-B | Not started | Research implementation absent; bootstrap interfaces only |
| MixStyle | Not started | Research implementation absent; bootstrap interfaces only |
| Deep CORAL | Not started | Research implementation absent; bootstrap interfaces only |
| Class-conditional CORAL | Not started | Research implementation absent; bootstrap interfaces only |
| Trainer | Not started | Runtime and checkpoint utilities tested; no training loop or models implemented |
| Evaluation | Not started | Research implementation absent; bootstrap interfaces only |
| Experiment tracking | Not started | JSON metadata writer tested; no experiment runs or MLflow adapter |

Statuses are declared evidence records, not inferred from files or issue closure.
Validated means experimentally validated; no component is promoted automatically.

## C. Architecture Changes

- Proposed: ADR-0001 (Data and evaluation protocol); ADR-0002 (Modality preprocessing); ADR-0003 (Model and fusion interfaces); ADR-0004 (Domain generalization comparisons); ADR-0005 (Local and Colab runtime boundaries)
- Accepted: None recorded
- Rejected: None recorded
- Superseded: None recorded
- Awaiting Lead approval: ADR-0001, ADR-0002, ADR-0003, ADR-0004, ADR-0005

## D. Current Experimental Results

No verified experiments available.

Unverified/incomplete records excluded: 0.

## E. Current Blockers

- Select and verify immutable dataset revision and file checksums.
- Audit official event overlap before approving the unseen-event protocol.
- Approve ADR-0001 through ADR-0004 before research implementation; review ADR-0005 runtime support.
- Configure and verify GitHub protections, team identities and reviewer permissions.
- MLflow and actual experiment records are unavailable.

## F. Recent Activity

Merged PRs (up to 30 recently updated closed PRs):
- None in available snapshot; external absence is not proof of no activity.
Completed tasks (explicit evidence only):
- No completed task records supplied.
Failed/cancelled checks (up to 30 recent workflow runs):
- None in available snapshot; CI success is not inferred.
Relevant architecture decisions: see section C.

## G. Next Milestones

| Milestone | Dependencies | Exit evidence |
| --- | --- | --- |
| M0 Repository Infrastructure | none | Local checks; reviewed setup PR; GitHub rules enabled |
| M1 Data Acquisition and Data Contracts | M0, ADR-0001/0002 | Pinned provenance, approved split audit, small verified sample, real contract tests |
| M2 End-to-End ERM Baseline | M1, ADR-0003 | ResNet integration, reproducible smoke run and reviewed evaluation |
| M3 Domain Generalization Methods | M2, ADR-0004 | Independent DG tests and controlled source-event experiments |
| M4 TerraMind Integration | M1, ADR-0003 | Adapter feasibility, approved checkpoint provenance, contract parity |
| M5 Cross-Event Experiments | M2, M3, M4 | Comparable runs across approved seeds/events and verified records |
| M6 Ablations and Final Evaluation | M5 | Frozen selection, held-out evaluation, uncertainty and limitations |

M0 is implemented and locally tested, pending review and integration; research components
are not started. `reports/progress.yaml` stores explicit component evidence and
blockers. Issue closure alone never promotes a component status.

## Data availability

- MLflow unavailable: no tracking backend is configured.
- GitHub issues, PRs and CI unavailable: offline mode.
- Local experiment record directory unavailable.
- GitHub snapshots, when requested, are bounded to 30 records per endpoint.

# Milestones

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

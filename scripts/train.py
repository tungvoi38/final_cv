"""Training entry point with environment preflight; research training is unimplemented."""

import argparse
import json
from pathlib import Path

from geoid.training.runtime import runtime_plan, torch_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--environment", default="configs/environments/local.yaml")
    parser.add_argument("--resume", help="Trusted full-state checkpoint at an epoch boundary")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        plan = runtime_plan(
            args.config,
            args.environment,
            root=Path(__file__).resolve().parents[1],
            backend=torch_backend(),
            resume=args.resume,
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Runtime preflight failed: {exc}\n")
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return
    parser.exit(2, "training.trainer.train: not implemented; approve research ADRs first.\n")


if __name__ == "__main__":
    main()

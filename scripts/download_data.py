"""Preview an explicitly pinned acquisition plan; no network activity by default."""

import argparse
import json

from geoid.config import load_yaml
from geoid.data.download import acquire


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/geoid.yaml")
    parser.add_argument("--mode", choices=["sample", "full"], default="sample")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output", default="data/raw/geoid-flood")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--confirm-full", action="store_true")
    args = parser.parse_args()
    try:
        result = acquire(
            load_yaml(args.config),
            output=args.output,
            mode=args.mode,
            execute=args.execute,
            verify_only=args.verify_only,
            overwrite=args.overwrite,
            confirm_full=args.confirm_full,
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Acquisition stopped: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

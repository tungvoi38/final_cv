"""Validate raw manifest schema/isolation and optionally verify local files."""

import argparse
import json
from pathlib import Path

from geoid.data.manifest import validate_manifest, verify_manifest_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--raw-root", type=Path)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        if args.raw_root:
            missing = verify_manifest_files(manifest, args.raw_root)
            if missing:
                parser.exit(2, f"Unverified checksums: {', '.join(missing)}\n")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Manifest invalid: {exc}\n")
    print(
        "Manifest schema and proposed split isolation passed; spatial metadata needs raster audit."
    )


if __name__ == "__main__":
    main()

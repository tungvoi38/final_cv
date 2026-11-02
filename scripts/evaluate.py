"""evaluate entry point: intentionally unavailable until research implementation."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.parse_args()
    parser.exit(2, "evaluation.metrics.evaluate: not implemented; approve research ADRs first.\n")


if __name__ == "__main__":
    main()

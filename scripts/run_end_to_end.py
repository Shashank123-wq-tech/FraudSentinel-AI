"""CLI: train Phase 1, hand its risk table to the existing Phase 2 pipeline."""
import argparse
import pandas as pd
from src.pipeline.end_to_end_pipeline import run_end_to_end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--phase1-output", default="artifacts/phase1")
    parser.add_argument("--phase2-output", default="artifacts/phase2")
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    run_end_to_end(df, args.phase1_output, args.phase2_output)
    print("End-to-end Phase-1 -> Phase-2 run complete.")


if __name__ == "__main__":
    main()

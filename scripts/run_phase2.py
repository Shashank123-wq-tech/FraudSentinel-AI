from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# CURRENT PHASE 2 PIPELINE
# ============================================================================

from src.components.phase2.pipeline import run_phase2


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "FraudSentinel AI — "
            "Phase 2 Temporal Fraud-Spike Intelligence"
        )
    )

    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT
            / "config"
            / "phase2.yaml"
        ),
        help="Path to Phase-2 configuration file.",
    )

    parser.add_argument(
        "--phase1-input",
        required=True,
        help="Fresh Phase-1 transaction_risk.csv",
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            PROJECT_ROOT
            / "artifacts"
            / "phase2"
        ),
        help="Phase-2 output directory.",
    )

    args = parser.parse_args()

    # ========================================================================
    # CONFIG
    # ========================================================================

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config_path = config_path.resolve()

    # ========================================================================
    # PHASE 1 INPUT
    # ========================================================================

    phase1_input = Path(args.phase1_input)

    if not phase1_input.is_absolute():
        phase1_input = PROJECT_ROOT / phase1_input

    phase1_input = phase1_input.resolve()

    # ========================================================================
    # OUTPUT
    # ========================================================================

    output_dir = Path(args.output_dir)

    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    output_dir = output_dir.resolve()

    # ========================================================================
    # VALIDATION
    # ========================================================================

    if not config_path.exists():
        raise FileNotFoundError(
            f"Phase-2 config not found:\n{config_path}"
        )

    if not phase1_input.exists():
        raise FileNotFoundError(
            f"Phase-1 input not found:\n{phase1_input}"
        )

    # ========================================================================
    # LOAD CONFIG
    # ========================================================================

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        cfg = yaml.safe_load(handle)

    if not isinstance(cfg, dict):
        raise ValueError(
            "Phase-2 YAML must contain a mapping/object."
        )

    # ========================================================================
    # RUN PHASE 2
    # ========================================================================

    print("=" * 80)
    print("FRAUDSENTINEL AI — PHASE 2")
    print("=" * 80)

    print(f"Config       : {config_path}")
    print(f"Phase-1 input: {phase1_input}")
    print(f"Output dir   : {output_dir}")

    result = run_phase2(
        input_path=str(phase1_input),
        output_dir=str(output_dir),
    )

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print()
    print("=" * 80)
    print("PHASE 2 RUNNER COMPLETE")
    print("=" * 80)

    print(
        f"Windows : {len(result['windows']):,}"
    )

    print(
        f"Events  : {len(result['events']):,}"
    )

    print()
    print("Outputs:")

    print(
        f"  {result['windows_path']}"
    )

    print(
        f"  {result['events_path']}"
    )

    print("=" * 80)


if __name__ == "__main__":

    try:
        main()

    except Exception as exc:

        print(
            f"\n[PHASE2][FATAL] "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

        raise
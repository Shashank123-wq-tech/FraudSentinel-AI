from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.pipeline.phase2_pipeline import Phase2Pipeline  # noqa: E402


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
    )

    parser.add_argument(
        "--phase1-input",
        required=True,
        help=(
            "Fresh Phase-1 transaction_risk.csv"
        ),
    )

    args = parser.parse_args()

    config_path = Path(
        args.config
    )

    if not config_path.is_absolute():
        config_path = (
            PROJECT_ROOT
            / config_path
        )

    config_path = config_path.resolve()

    phase1_input = Path(
        args.phase1_input
    )

    if not phase1_input.is_absolute():
        phase1_input = (
            PROJECT_ROOT
            / phase1_input
        )

    phase1_input = phase1_input.resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Phase-2 config not found:\n"
            f"{config_path}"
        )

    if not phase1_input.exists():
        raise FileNotFoundError(
            f"Phase-1 input not found:\n"
            f"{phase1_input}"
        )

    print("=" * 80)
    print(
        "FRAUDSENTINEL AI — PHASE 2"
    )
    print("=" * 80)

    print(
        f"Config: {config_path}"
    )

    print(
        f"Phase-1 input: {phase1_input}"
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        cfg = yaml.safe_load(handle)

    if not isinstance(cfg, dict):
        raise ValueError(
            "Phase-2 YAML must contain a mapping/object."
        )

    if "input" not in cfg:
        cfg["input"] = {}

    # FORCE the fresh Phase-1 artifact.
    cfg["input"]["phase1_risk_output"] = str(
        phase1_input
    )

    metrics = Phase2Pipeline(
        cfg
    ).run()

    print()
    print("=" * 80)
    print(
        "PHASE 2 COMPLETE"
    )
    print("=" * 80)

    print(
        yaml.safe_dump(
            metrics,
            sort_keys=False,
            default_flow_style=False,
        )
    )


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
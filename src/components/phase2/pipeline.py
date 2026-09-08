# src/components/phase2/pipeline.py

from pathlib import Path
import pandas as pd

from .loader import load_phase1_risk
from .windows import build_merchant_windows
from .baseline import build_merchant_baseline
from .statistical import add_statistical_signals
from .temporal import add_temporal_signals
from .breadth import add_breadth_signal
from .score import add_phase2_scores
from .detector import classify_spike_state
from .events import build_fraud_spike_events


def _debug_stage(
    df: pd.DataFrame,
    stage_name: str,
    required_columns=None,
):
    """
    Debug helper used to inspect the dataframe after each
    Phase-2 processing stage.
    """

    print("\n" + "=" * 70)
    print(f"[DEBUG] {stage_name}")
    print("=" * 70)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    if required_columns:

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:

            print("\n!!! MISSING REQUIRED COLUMNS !!!")

            for column in missing:
                print(f"  - {column}")

        else:

            print("\nRequired columns: PASS")

    print("\nDtypes:")
    print(df.dtypes.to_string())

    print("\nFirst 3 rows:")

    print(
        df.head(3).to_string()
    )

    return df


def run_phase2(
    input_path: str,
    output_dir: str,
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("FRAUDSENTINEL AI — PHASE 2")
    print("FRAUD SPIKE DETECTION")
    print("=" * 70)

    # ============================================================
    # 1. LOAD
    # ============================================================

    print("\n[1/9] Loading Phase-1 risk output...")

    df = load_phase1_risk(
        input_path
    )

    _debug_stage(
        df,
        "AFTER LOADER",
        required_columns=[
            "transaction_id",
            "timestamp",
            "merchant_id",
            "card_id",
            "amount",
            "fraud_probability",
        ],
    )

    # ============================================================
    # 2. WINDOWS
    # ============================================================

    print("\n[2/9] Building merchant time windows...")

    windows = build_merchant_windows(
        df
    )

    _debug_stage(
        windows,
        "AFTER WINDOWS",
        required_columns=[
            "merchant_id",
            "window_start",
            "window_end",
        ],
    )

    # ============================================================
    # 3. BASELINE
    # ============================================================

    print("\n[3/9] Building historical merchant baseline...")

    windows = build_merchant_baseline(
        windows
    )

    _debug_stage(
        windows,
        "AFTER BASELINE",
    )

    # ============================================================
    # 4. STATISTICAL
    # ============================================================

    print("\n[4/9] Calculating statistical anomaly...")

    windows = add_statistical_signals(
        windows
    )

    _debug_stage(
        windows,
        "AFTER STATISTICAL FEATURES",
    )

    # ============================================================
    # 5. TEMPORAL
    # ============================================================

    print("\n[5/9] Calculating temporal escalation...")

    windows = add_temporal_signals(
        windows
    )

    _debug_stage(
        windows,
        "AFTER TEMPORAL FEATURES",
    )

    # ============================================================
    # 6. BREADTH
    # ============================================================

    print("\n[6/9] Calculating breadth / coordination...")

    windows = add_breadth_signal(
        windows
    )

    _debug_stage(
        windows,
        "AFTER BREADTH FEATURES",
    )

    # ============================================================
    # 7. SCORE
    # ============================================================

    print("\n[7/9] Calculating Fraud Spike Score...")

    windows = add_phase2_scores(
        windows
    )

    _debug_stage(
        windows,
        "AFTER SPIKE SCORE",
        required_columns=[
            "window_start",
            "window_end",
            "spike_score",
        ],
    )

    # ============================================================
    # 8. DETECTOR
    # ============================================================

    print("\n[8/9] Detecting fraud spikes...")

    windows = classify_spike_state(
        windows
    )

    _debug_stage(
        windows,
        "AFTER SPIKE DETECTOR",
        required_columns=[
            "merchant_id",
            "window_start",
            "window_end",
            "spike_score",
            "spike_state",
        ],
    )

    # ============================================================
    # 9. EVENTS
    # ============================================================

    print("\n[9/9] Building fraud-spike events...")

    # ------------------------------------------------------------
    # FINAL SAFETY CHECK
    # ------------------------------------------------------------

    required_event_columns = [
        "merchant_id",
        "window_start",
        "window_end",
        "spike_score",
        "spike_state",
    ]

    missing_event_columns = [
        column
        for column in required_event_columns
        if column not in windows.columns
    ]

    if missing_event_columns:

        print("\n" + "!" * 70)
        print("PHASE 2 PIPELINE STOPPED")
        print("!" * 70)

        print(
            "\nMissing columns before event construction:"
        )

        for column in missing_event_columns:
            print(f"  ❌ {column}")

        print("\nAvailable columns:")

        for column in windows.columns:
            print(f"  ✓ {column}")

        raise RuntimeError(
            "Event construction cannot continue because "
            f"these columns are missing: "
            f"{missing_event_columns}"
        )

    events = build_fraud_spike_events(
        windows,
        max_gap_minutes=15,
    )

    # ============================================================
    # SAVE WINDOW RESULTS
    # ============================================================

    windows_path = (
        output_dir
        / "phase2_windows.csv"
    )

    events_path = (
        output_dir
        / "phase2_events.csv"
    )

    windows.to_csv(
        windows_path,
        index=False,
    )

    events.to_csv(
        events_path,
        index=False,
    )

    # ============================================================
    # SUMMARY
    # ============================================================

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)

    print(
        f"Total windows       : "
        f"{len(windows):,}"
    )

    if "spike_state" in windows.columns:

        print(
            f"Candidate windows   : "
            f"{(windows['spike_state'] == 'candidate').sum():,}"
        )

        print(
            f"Verified windows    : "
            f"{(windows['spike_state'] == 'verified').sum():,}"
        )

        print(
            f"Critical windows    : "
            f"{(windows['spike_state'] == 'critical').sum():,}"
        )

    print(
        f"Fraud events        : "
        f"{len(events):,}"
    )

    print("\nOutputs:")

    print(
        f"  {windows_path}"
    )

    print(
        f"  {events_path}"
    )

    print("=" * 70)

    return {
        "windows": windows,
        "events": events,
        "windows_path": str(
            windows_path
        ),
        "events_path": str(
            events_path
        ),
    }
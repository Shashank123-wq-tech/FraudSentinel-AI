from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None


# ============================================================
# EXCEPTIONS
# ============================================================

class ValidationError(RuntimeError):
    """Expected validation/configuration failure."""


def die(message: str) -> None:
    raise ValidationError(message)


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        x = float(value)

        if math.isfinite(x):
            return x

        return default

    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default

        return int(value)

    except Exception:
        return default


def json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, np.floating):
        x = float(value)
        return x if math.isfinite(x) else None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(v)
            for v in value
        ]

    return str(value)


# ============================================================
# FILE WRITERS
# ============================================================

def write_json(
    path: Path,
    payload: dict[str, Any]
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        json.dumps(
            json_safe(payload),
            indent=2
        ),
        encoding="utf-8"
    )


def write_text_report(
    path: Path,
    payload: dict[str, Any]
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    lines: list[str] = []

    lines.append(
        "FraudSentinel AI - Phase 2 Validation"
    )

    lines.append("=" * 72)

    lines.append(
        f"status: {payload.get('status', 'UNKNOWN')}"
    )

    lines.append(
        f"runtime_seconds: "
        f"{safe_float(payload.get('runtime_seconds'), 0.0):.2f}"
    )

    lines.append("")

    sections = payload.get(
        "sections",
        {}
    )

    for name, section in sections.items():

        lines.append(
            str(name).upper()
        )

        lines.append("-" * 72)

        if isinstance(section, dict):

            for key, value in section.items():

                lines.append(
                    f"{key}: {value}"
                )

        else:

            lines.append(
                str(section)
            )

        lines.append("")

    errors = payload.get(
        "errors",
        []
    )

    if errors:

        lines.append("ERRORS")
        lines.append("-" * 72)

        for error in errors:
            lines.append(str(error))

        lines.append("")

    path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]]
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not rows:

        path.write_text(
            "",
            encoding="utf-8"
        )

        return

    columns: list[str] = []
    seen: set[str] = set()

    for row in rows:

        for key in row:

            if key not in seen:

                seen.add(key)
                columns.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=columns
        )

        writer.writeheader()

        for row in rows:

            writer.writerow({
                key: json_safe(
                    row.get(key)
                )
                for key in columns
            })


# ============================================================
# SQL HELPERS
# ============================================================

def sql_quote(value: str) -> str:
    """
    Quote a SQL string literal.
    Example:
        hello -> 'hello'
    """

    return "'" + str(value).replace(
        "'",
        "''"
    ) + "'"


def ident(value: str) -> str:
    """
    Quote a SQL identifier / column name.
    Example:
        fraud_probability -> "fraud_probability"
    """

    return '"' + str(value).replace(
        '"',
        '""'
    ) + '"'


def path_sql(path: Path) -> str:

    return sql_quote(
        str(path.resolve())
    )


# ============================================================
# PATH HELPERS
# ============================================================

def first_existing(
    candidates: list[Path]
) -> Path | None:

    for candidate in candidates:

        if (
            candidate.exists()
            and candidate.is_file()
        ):
            return candidate

    return None


# ============================================================
# YAML
# ============================================================

def read_yaml(
    path: Path
) -> dict[str, Any]:

    if not path.exists():

        return {}

    if yaml is None:

        raise ValidationError(
            "PyYAML is not installed. "
            "Install it using: pip install pyyaml"
        )

    try:

        data = yaml.safe_load(
            path.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(data, dict):
            return data

        return {}

    except Exception as exc:

        raise ValidationError(
            f"Cannot parse YAML config {path}: {exc}"
        ) from exc


def nested(
    config: dict[str, Any],
    *keys: str,
    default: Any = None
) -> Any:

    current: Any = config

    for key in keys:

        if (
            not isinstance(current, dict)
            or key not in current
        ):
            return default

        current = current[key]

    return current


# ============================================================
# SETTINGS
# ============================================================

def load_settings(
    config: dict[str, Any]
) -> dict[str, Any]:

    return {

        "episode_gap":
            safe_float(
                nested(
                    config,
                    "events",
                    "max_event_gap_minutes",
                    default=15.0
                ),
                15.0
            ),

        "event_window":
            safe_float(
                nested(
                    config,
                    "events",
                    "event_window_minutes",
                    default=15.0
                ),
                15.0
            ),

        "match_tolerance":
            safe_float(
                nested(
                    config,
                    "evaluation",
                    "matching_tolerance_minutes",
                    default=60.0
                ),
                60.0
            ),

        "candidate_tas":
            safe_float(
                nested(
                    config,
                    "verification",
                    "candidate_tas",
                    default=35.0
                ),
                35.0
            ),

        "verified_tas":
            safe_float(
                nested(
                    config,
                    "verification",
                    "verified_tas",
                    default=48.0
                ),
                48.0
            ),

        "critical_tas":
            safe_float(
                nested(
                    config,
                    "verification",
                    "critical_tas",
                    default=70.0
                ),
                70.0
            ),

        "min_materiality":
            safe_float(
                nested(
                    config,
                    "verification",
                    "min_materiality",
                    default=0.35
                ),
                0.35
            ),

        "min_history":
            safe_float(
                nested(
                    config,
                    "verification",
                    "min_history_confidence",
                    default=0.35
                ),
                0.35
            ),

        "min_persistence":
            safe_float(
                nested(
                    config,
                    "verification",
                    "persistence_min",
                    default=0.30
                ),
                0.30
            ),

        "min_corroboration":
            safe_float(
                nested(
                    config,
                    "verification",
                    "corroboration_min",
                    default=0.25
                ),
                0.25
            ),

        "exceptional_amount":
            safe_float(
                nested(
                    config,
                    "materiality",
                    "exceptional_min_expected_fraud_amount",
                    default=250.0
                ),
                250.0
            ),

        "exceptional_tas":
            safe_float(
                nested(
                    config,
                    "materiality",
                    "exceptional_min_tas",
                    default=70.0
                ),
                70.0
            ),

        "attack_gamma":
            safe_float(
                nested(
                    config,
                    "fraud_attack_score",
                    "attack_gamma",
                    default=1.0
                ),
                1.0
            ),

        "min_txn_episode":
            safe_int(
                nested(
                    config,
                    "events",
                    "minimum_event_windows",
                    default=2
                ),
                2
            ),
    }


# ============================================================
# PHASE-2 ARTIFACT DISCOVERY
# ============================================================

def discover_phase2_windows(
    phase2_dir: Path
) -> dict[int, Path]:

    candidates = {

        5: [
            phase2_dir / "windows" /
            "merchant_spike_windows_5m.csv",

            phase2_dir / "windows" /
            "merchant_spike_windows_5min.csv",

            phase2_dir /
            "merchant_spike_windows_5m.csv",
        ],

        15: [
            phase2_dir / "windows" /
            "merchant_spike_windows_15m.csv",

            phase2_dir / "windows" /
            "merchant_spike_windows.csv",

            phase2_dir /
            "merchant_spike_windows_15m.csv",
        ],

        60: [
            phase2_dir / "windows" /
            "merchant_spike_windows_60m.csv",

            phase2_dir / "windows" /
            "merchant_spike_windows_60min.csv",

            phase2_dir / "windows" /
            "merchant_spike_windows_1h.csv",
        ],
    }

    found: dict[int, Path] = {}

    for window, paths in candidates.items():

        path = first_existing(paths)

        if path is not None:
            found[window] = path

    return found


def discover_events(
    phase2_dir: Path
) -> Path | None:

    return first_existing([

        phase2_dir / "events" /
        "fraud_spike_events.csv",

        phase2_dir / "events" /
        "merchant_spike_events.csv",

        phase2_dir /
        "fraud_spike_events.csv",

        phase2_dir /
        "merchant_spike_events.csv",
    ])


# ============================================================
# PATH DISCOVERY
# ============================================================

def discover_paths(
    args: argparse.Namespace
) -> tuple[Path, Path, Path, Path]:

    root = Path(
        args.project_root
    ).resolve()

    raw = (
        Path(args.raw_data).resolve()
        if args.raw_data
        else None
    )

    risk = (
        Path(args.risk_output).resolve()
        if args.risk_output
        else None
    )

    phase2 = (
        Path(args.phase2_dir).resolve()
        if args.phase2_dir
        else root /
        "artifacts" /
        "phase2"
    )

    config = (
        Path(args.config).resolve()
        if args.config
        else root /
        "config" /
        "phase2.yaml"
    )

    if raw is None:

        raw = first_existing([

            root /
            "data" /
            "processed" /
            "phase1" /
            "train.csv",

            root /
            "data" /
            "processed" /
            "phase1" /
            "phase1_train.csv",

            root /
            "data" /
            "processed" /
            "train.csv",

            root /
            "data" /
            "train.csv",
        ])

    if risk is None:

        risk = first_existing([

            root /
            "artifacts" /
            "phase1" /
            "risk_output" /
            "transaction_risk.csv",

            root /
            "artifacts" /
            "phase1" /
            "predictions" /
            "risk_output_table.csv",

            root /
            "artifacts" /
            "phase1" /
            "risk_output" /
            "risk_output_table.csv",
        ])

    if raw is None:

        die(
            "Raw transaction file was not found. "
            "Use --raw-data explicitly."
        )

    if risk is None:

        die(
            "Phase-1 risk output was not found. "
            "Use --risk-output explicitly."
        )

    if not raw.exists():

        die(
            f"Raw transaction file does not exist: {raw}"
        )

    if not risk.exists():

        die(
            f"Phase-1 risk output does not exist: {risk}"
        )

    if not phase2.exists():

        die(
            f"Phase-2 directory does not exist: {phase2}"
        )

    return (
        raw,
        risk,
        phase2,
        config
    )


# ============================================================
# DUCKDB
# ============================================================

def connect() -> duckdb.DuckDBPyConnection:

    con = duckdb.connect()

    threads = max(
        1,
        min(
            8,
            os.cpu_count() or 4
        )
    )

    con.execute(
        f"PRAGMA threads={threads}"
    )

    con.execute(
        "PRAGMA preserve_insertion_order=false"
    )

    return con


def relation_sql(
    path: Path
) -> str:

    suffix = path.suffix.lower()

    if suffix in {
        ".parquet",
        ".pq"
    }:

        return (
            f"read_parquet("
            f"{path_sql(path)})"
        )

    if suffix in {
        ".csv",
        ".txt"
    }:

        return (
            f"read_csv_auto("
            f"{path_sql(path)}, "
            f"header=true, "
            f"ignore_errors=false)"
        )

    die(
        f"Unsupported input type: {path}"
    )

    return ""


def describe_relation(
    con: duckdb.DuckDBPyConnection,
    relation: str
) -> list[str]:

    rows = con.execute(
        f"DESCRIBE SELECT * FROM {relation}"
    ).fetchall()

    return [
        str(row[0])
        for row in rows
    ]


# ============================================================
# COLUMN RESOLUTION
# ============================================================

def choose_column(
    columns: list[str],
    names: list[str]
) -> str | None:

    lower_map = {
        column.lower(): column
        for column in columns
    }

    for name in names:

        if name.lower() in lower_map:

            return lower_map[
                name.lower()
            ]

    return None


# ============================================================
# PHASE-1 SCHEMA AUDIT
# ============================================================

def phase1_schema_audit(
    con: duckdb.DuckDBPyConnection,
    raw: Path,
    risk: Path
) -> tuple[
    dict[str, Any],
    str,
    str
]:

    raw_rel = relation_sql(raw)
    risk_rel = relation_sql(risk)

    raw_cols = describe_relation(
        con,
        raw_rel
    )

    risk_cols = describe_relation(
        con,
        risk_rel
    )

    # --------------------------------------------------------
    # RAW COLUMNS
    # --------------------------------------------------------

    timestamp_raw = choose_column(
        raw_cols,
        [
            "trans_date_trans_time",
            "timestamp",
            "trans_datetime",
            "date_time"
        ]
    )

    fraud_raw = choose_column(
        raw_cols,
        [
            "is_fraud",
            "fraud",
            "label"
        ]
    )

    merchant_raw = choose_column(
        raw_cols,
        [
            "merchant_id",
            "merchant"
        ]
    )

    card_raw = choose_column(
        raw_cols,
        [
            "card_id",
            "cc_num",
            "card"
        ]
    )

    amount_raw = choose_column(
        raw_cols,
        [
            "amount",
            "amt",
            "transaction_amount"
        ]
    )

    required_raw = {

        "timestamp":
            timestamp_raw,

        "fraud_label":
            fraud_raw,

        "merchant_id":
            merchant_raw,

        "card_id":
            card_raw,

        "amount":
            amount_raw,
    }

    missing_raw = [
        name
        for name, column
        in required_raw.items()
        if column is None
    ]

    if missing_raw:

        die(
            "Raw dataset is missing required "
            f"semantic columns: {missing_raw}. "
            f"Available: {raw_cols}"
        )

    # --------------------------------------------------------
    # RISK COLUMNS
    # --------------------------------------------------------

    transaction_id_risk = choose_column(
        risk_cols,
        ["transaction_id"]
    )

    timestamp_risk = choose_column(
        risk_cols,
        ["timestamp"]
    )

    merchant_risk = choose_column(
        risk_cols,
        ["merchant_id"]
    )

    card_risk = choose_column(
        risk_cols,
        ["card_id"]
    )

    amount_risk = choose_column(
        risk_cols,
        ["amount"]
    )

    probability_risk = choose_column(
        risk_cols,
        [
            "fraud_probability",
            "fraud_prob",
            "risk_probability"
        ]
    )

    required_risk = {

        "transaction_id":
            transaction_id_risk,

        "timestamp":
            timestamp_risk,

        "merchant_id":
            merchant_risk,

        "card_id":
            card_risk,

        "amount":
            amount_risk,

        "fraud_probability":
            probability_risk,
    }

    missing_risk = [
        name
        for name, column
        in required_risk.items()
        if column is None
    ]

    if missing_risk:

        die(
            "Phase-1 risk output is missing "
            f"required columns: {missing_risk}. "
            f"Available: {risk_cols}"
        )

    # --------------------------------------------------------
    # RAW STATISTICS
    #
    # INDEX MAP:
    #
    # [0] rows
    # [1] bad_timestamps
    # [2] bad_amounts
    # [3] negative_amounts
    # [4] bad_labels
    # [5] merchants
    # [6] cards
    # [7] fraud_transactions
    # --------------------------------------------------------

    raw_stats = con.execute(
        f"""
        SELECT
            COUNT(*) AS rows,

            COUNT(*) FILTER (
                WHERE TRY_CAST(
                    {ident(timestamp_raw)}
                    AS TIMESTAMP
                ) IS NULL
            ) AS bad_timestamps,

            COUNT(*) FILTER (
                WHERE TRY_CAST(
                    {ident(amount_raw)}
                    AS DOUBLE
                ) IS NULL
            ) AS bad_amounts,

            COUNT(*) FILTER (
                WHERE TRY_CAST(
                    {ident(amount_raw)}
                    AS DOUBLE
                ) < 0
            ) AS negative_amounts,

            COUNT(*) FILTER (
                WHERE TRY_CAST(
                    {ident(fraud_raw)}
                    AS INTEGER
                ) NOT IN (0, 1)
            ) AS bad_labels,

            COUNT(DISTINCT
                {ident(merchant_raw)}
            ) AS merchants,

            COUNT(DISTINCT
                {ident(card_raw)}
            ) AS cards,

            SUM(
                TRY_CAST(
                    {ident(fraud_raw)}
                    AS INTEGER
                )
            ) AS fraud_transactions

        FROM {raw_rel}
        """
    ).fetchone()

    if raw_stats is None:

        die(
            "Could not calculate raw dataset statistics."
        )

    # --------------------------------------------------------
    # RISK STATISTICS
    #
    # INDEX MAP:
    #
    # [0] rows
    # [1] unique_transaction_ids
    # [2] duplicate_transaction_ids
    # [3] invalid_probability
    # [4] out_of_range_probability
    # [5] negative_amounts
    # [6] bad_timestamps
    # [7] merchants
    # [8] cards
    # --------------------------------------------------------

    risk_stats = con.execute(
        f"""
        SELECT

            COUNT(*) AS rows,

            COUNT(DISTINCT
                {ident(transaction_id_risk)}
            ) AS unique_transaction_ids,

            COUNT(*)
            -
            COUNT(DISTINCT
                {ident(transaction_id_risk)}
            ) AS duplicate_transaction_ids,

            COUNT(*) FILTER (
                WHERE TRY_CAST(
                    {ident(probability_risk)}
                    AS DOUBLE
                ) IS NULL
            ) AS null_or_invalid_probability,

            COUNT(*) FILTER (
                WHERE
                    TRY_CAST(
                        {ident(probability_risk)}
                        AS DOUBLE
                    ) < 0
                    OR
                    TRY_CAST(
                        {ident(probability_risk)}
                        AS DOUBLE
                    ) > 1
            ) AS out_of_range_probability,

            COUNT(*) FILTER (
                WHERE TRY_CAST(
                    {ident(amount_risk)}
                    AS DOUBLE
                ) < 0
            ) AS negative_amounts,

            COUNT(*) FILTER (
                WHERE TRY_CAST(
                    {ident(timestamp_risk)}
                    AS TIMESTAMP
                ) IS NULL
            ) AS bad_timestamps,

            COUNT(DISTINCT
                {ident(merchant_risk)}
            ) AS merchants,

            COUNT(DISTINCT
                {ident(card_risk)}
            ) AS cards

        FROM {risk_rel}
        """
    ).fetchone()

    if risk_stats is None:

        die(
            "Could not calculate Phase-1 risk statistics."
        )

    # --------------------------------------------------------
    # CORRECT INDEXES
    # --------------------------------------------------------

    raw_row_count = safe_int(
        raw_stats[0]
    )

    risk_row_count = safe_int(
        risk_stats[0]
    )

    fraud_count = safe_int(
        raw_stats[7]
    )

    row_count_match = (
        raw_row_count ==
        risk_row_count
    )

    if not row_count_match:

        raise ValidationError(
            "Phase-1 lineage row-count mismatch: "
            f"raw={raw_row_count:,}, "
            f"risk_output={risk_row_count:,}"
        )

    if safe_int(risk_stats[2]) > 0:

        raise ValidationError(
            "Phase-1 risk output contains "
            "duplicate transaction_id values."
        )

    if safe_int(risk_stats[3]) > 0:

        raise ValidationError(
            "Phase-1 risk output contains "
            "invalid/null fraud_probability values."
        )

    if safe_int(risk_stats[4]) > 0:

        raise ValidationError(
            "Phase-1 risk output contains "
            "fraud_probability outside [0,1]."
        )

    if safe_int(risk_stats[5]) > 0:

        raise ValidationError(
            "Phase-1 risk output contains "
            "negative transaction amounts."
        )

    if safe_int(risk_stats[6]) > 0:

        raise ValidationError(
            "Phase-1 risk output contains "
            "invalid timestamps."
        )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    result = {

        "raw_rows":
            raw_row_count,

        "risk_rows":
            risk_row_count,

        "row_count_match":
            row_count_match,

        "raw_fraud_transactions":
            fraud_count,

        # CORRECT INDEXES
        "raw_merchants":
            safe_int(raw_stats[5]),

        "raw_cards":
            safe_int(raw_stats[6]),

        "risk_merchants":
            safe_int(risk_stats[7]),

        "risk_cards":
            safe_int(risk_stats[8]),

        "raw_bad_timestamps":
            safe_int(raw_stats[1]),

        "raw_bad_amounts":
            safe_int(raw_stats[2]),

        "raw_negative_amounts":
            safe_int(raw_stats[3]),

        "raw_bad_labels":
            safe_int(raw_stats[4]),

        "risk_duplicate_transaction_ids":
            safe_int(risk_stats[2]),

        "risk_invalid_probability":
            safe_int(risk_stats[3]),

        "risk_out_of_range_probability":
            safe_int(risk_stats[4]),
    }

    return (
        result,
        raw_rel,
        risk_rel
    )


# ============================================================
# GROUND TRUTH
# ============================================================

def build_ground_truth(
    con: duckdb.DuckDBPyConnection,
    raw_rel: str,
    settings: dict[str, Any]
) -> dict[str, Any]:

    gap = settings[
        "episode_gap"
    ]

    raw_cols = describe_relation(
        con,
        raw_rel
    )

    ts = choose_column(
        raw_cols,
        [
            "trans_date_trans_time",
            "timestamp",
            "trans_datetime",
            "date_time"
        ]
    )

    merchant = choose_column(
        raw_cols,
        [
            "merchant_id",
            "merchant"
        ]
    )

    card = choose_column(
        raw_cols,
        [
            "card_id",
            "cc_num",
            "card"
        ]
    )

    amount = choose_column(
        raw_cols,
        [
            "amount",
            "amt",
            "transaction_amount"
        ]
    )

    fraud = choose_column(
        raw_cols,
        [
            "is_fraud",
            "fraud",
            "label"
        ]
    )

    if not all([
        ts,
        merchant,
        card,
        amount,
        fraud
    ]):

        die(
            "Unable to build ground truth. "
            f"Available columns: {raw_cols}"
        )

    # --------------------------------------------------------
    # FRAUD TRANSACTIONS
    # --------------------------------------------------------

    con.execute(
        "DROP VIEW IF EXISTS fraud_tx"
    )

    con.execute(
        f"""
        CREATE TEMP VIEW fraud_tx AS

        SELECT

            TRY_CAST(
                {ident(ts)}
                AS TIMESTAMP
            ) AS ts,

            CAST(
                {ident(merchant)}
                AS VARCHAR
            ) AS merchant_id,

            CAST(
                {ident(card)}
                AS VARCHAR
            ) AS card_id,

            TRY_CAST(
                {ident(amount)}
                AS DOUBLE
            ) AS amount,

            TRY_CAST(
                {ident(fraud)}
                AS INTEGER
            ) AS is_fraud

        FROM {raw_rel}

        WHERE TRY_CAST(
            {ident(fraud)}
            AS INTEGER
        ) = 1
        """
    )

    # --------------------------------------------------------
    # FRAUD SEQUENCE
    # --------------------------------------------------------

    con.execute(
        "DROP VIEW IF EXISTS fraud_seq"
    )

    con.execute(
        """
        CREATE TEMP VIEW fraud_seq AS

        SELECT

            *,

            LAG(ts) OVER (
                PARTITION BY merchant_id
                ORDER BY ts
            ) AS previous_ts

        FROM fraud_tx

        WHERE ts IS NOT NULL
        """
    )

    # --------------------------------------------------------
    # FRAUD EPISODES
    # --------------------------------------------------------

    con.execute(
        "DROP VIEW IF EXISTS fraud_episodes"
    )

    con.execute(
        f"""
        CREATE TEMP VIEW fraud_episodes AS

        WITH marked AS (

            SELECT

                *,

                CASE

                    WHEN previous_ts IS NULL
                        THEN 1

                    WHEN DATE_DIFF(
                        'second',
                        previous_ts,
                        ts
                    ) > {gap} * 60.0
                        THEN 1

                    ELSE 0

                END AS episode_start

            FROM fraud_seq
        ),

        numbered AS (

            SELECT

                *,

                SUM(
                    episode_start
                ) OVER (

                    PARTITION BY merchant_id

                    ORDER BY ts

                    ROWS BETWEEN
                        UNBOUNDED PRECEDING
                        AND CURRENT ROW

                ) AS episode_number

            FROM marked
        )

        SELECT

            merchant_id,

            episode_number,

            MIN(ts) AS start_time,

            MAX(ts) AS end_time,

            COUNT(*) AS fraud_count,

            SUM(amount) AS fraud_amount,

            COUNT(
                DISTINCT card_id
            ) AS unique_cards,

            DATE_DIFF(
                'second',
                MIN(ts),
                MAX(ts)
            ) / 60.0 AS duration_minutes

        FROM numbered

        GROUP BY
            merchant_id,
            episode_number
        """
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    episode_summary = con.execute(
        """
        SELECT

            COUNT(*) AS episodes,

            COALESCE(
                SUM(fraud_count),
                0
            ) AS fraud_transactions,

            COUNT(*) FILTER (
                WHERE fraud_count >= 2
            ) AS repeated_episodes,

            COUNT(*) FILTER (
                WHERE unique_cards >= 2
            ) AS multi_card_episodes,

            COUNT(*) FILTER (
                WHERE
                    fraud_count >= 2
                    AND unique_cards >= 2
            ) AS coordinated_episodes,

            AVG(duration_minutes)
                AS mean_duration_minutes,

            MEDIAN(duration_minutes)
                AS median_duration_minutes,

            MAX(duration_minutes)
                AS max_duration_minutes

        FROM fraud_episodes
        """
    ).fetchone()

    if episode_summary is None:

        return {

            "fraud_transactions": 0,
            "fraud_episodes": 0,
            "repeated_episodes": 0,
            "multi_card_episodes": 0,
            "coordinated_episodes": 0,
            "mean_duration_minutes": 0.0,
            "median_duration_minutes": 0.0,
            "max_duration_minutes": 0.0,
        }

    return {

        "fraud_transactions":
            safe_int(
                episode_summary[1]
            ),

        "fraud_episodes":
            safe_int(
                episode_summary[0]
            ),

        "repeated_episodes":
            safe_int(
                episode_summary[2]
            ),

        "multi_card_episodes":
            safe_int(
                episode_summary[3]
            ),

        "coordinated_episodes":
            safe_int(
                episode_summary[4]
            ),

        "mean_duration_minutes":
            safe_float(
                episode_summary[5]
            ),

        "median_duration_minutes":
            safe_float(
                episode_summary[6]
            ),

        "max_duration_minutes":
            safe_float(
                episode_summary[7]
            ),
    }


# ============================================================
# 15-MINUTE GROUND TRUTH
# ============================================================

def create_truth_15m(
    con: duckdb.DuckDBPyConnection
) -> None:

    con.execute(
        "DROP VIEW IF EXISTS truth_15m"
    )

    con.execute(
        """
        CREATE TEMP VIEW truth_15m AS

        WITH fraud_windows AS (

            SELECT

                merchant_id,

                DATE_TRUNC(
                    'hour',
                    start_time
                )
                +
                FLOOR(
                    EXTRACT(
                        MINUTE
                        FROM start_time
                    ) / 15
                )
                * INTERVAL '15 minutes'
                AS window_start,

                COUNT(*) AS
                    fraud_count_in_start_window,

                SUM(fraud_amount) AS
                    fraud_amount_in_start_window,

                MAX(fraud_count) AS
                    max_episode_tx,

                MAX(unique_cards) AS
                    max_episode_cards,

                COUNT(*) FILTER (
                    WHERE fraud_count >= 2
                ) AS
                    repeated_episode_count,

                COUNT(*) FILTER (
                    WHERE unique_cards >= 2
                ) AS
                    multi_card_episode_count,

                COUNT(*) FILTER (
                    WHERE
                        fraud_count >= 2
                        AND unique_cards >= 2
                ) AS
                    coordinated_episode_count

            FROM fraud_episodes

            GROUP BY
                merchant_id,
                window_start
        )

        SELECT

            merchant_id,

            window_start,

            CAST(
                fraud_count_in_start_window
                AS BIGINT
            ) AS actual_fraud_count,

            CAST(
                fraud_amount_in_start_window
                AS DOUBLE
            ) AS actual_fraud_amount,

            CAST(
                max_episode_tx
                AS BIGINT
            ) AS max_episode_tx,

            CAST(
                max_episode_cards
                AS BIGINT
            ) AS max_episode_cards,

            CAST(
                repeated_episode_count
                AS BIGINT
            ) AS repeated_episode_count,

            CAST(
                multi_card_episode_count
                AS BIGINT
            ) AS multi_card_episode_count,

            CAST(
                coordinated_episode_count
                AS BIGINT
            ) AS coordinated_episode_count,

            (
                fraud_count_in_start_window > 0
            ) AS actual_fraud_window,

            (
                repeated_episode_count > 0
            ) AS actual_repeated_episode_window,

            (
                multi_card_episode_count > 0
            ) AS actual_multi_card_episode_window,

            (
                coordinated_episode_count > 0
            ) AS actual_coordinated_episode_window

        FROM fraud_windows
        """
    )


# ============================================================
# PHASE-2 RELATION
# ============================================================

def find_timestamp_column(
    columns: list[str]
) -> str | None:

    return choose_column(
        columns,
        [
            "window_start",
            "timestamp",
            "start_time",
            "event_time",
            "detection_time"
        ]
    )


def phase2_relation(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    alias: str
) -> tuple[str, list[str]]:

    relation = relation_sql(path)

    columns = describe_relation(
        con,
        relation
    )

    con.execute(
        f"DROP VIEW IF EXISTS {ident(alias)}"
    )

    con.execute(
        f"""
        CREATE TEMP VIEW {ident(alias)} AS
        SELECT *
        FROM {relation}
        """
    )

    return (
        alias,
        columns
    )


# ============================================================
# NORMALISE PHASE-2 WINDOWS
# ============================================================

def normalise_phase2_windows(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    window: int
) -> tuple[
    str,
    dict[str, str | None]
]:

    alias = f"p2_{window}m"

    _, columns = phase2_relation(
        con,
        path,
        alias
    )

    merchant = choose_column(
        columns,
        [
            "merchant_id",
            "merchant"
        ]
    )

    timestamp = find_timestamp_column(
        columns
    )

    tas = choose_column(
        columns,
        [
            "tas",
            "spike_score",
            "final_tas"
        ]
    )

    fas = choose_column(
        columns,
        [
            "fraud_attack_score",
            "fas"
        ]
    )

    verification = choose_column(
        columns,
        [
            "verification_score",
            "verification_evidence"
        ]
    )

    persistence = choose_column(
        columns,
        [
            "persistence_score",
            "persistence"
        ]
    )

    corroboration = choose_column(
        columns,
        [
            "corroboration_score",
            "corroboration"
        ]
    )

    breadth = choose_column(
        columns,
        [
            "breadth_score",
            "breadth"
        ]
    )

    materiality = choose_column(
        columns,
        [
            "materiality_score",
            "materiality"
        ]
    )

    history = choose_column(
        columns,
        [
            "history_confidence",
            "baseline_history_confidence"
        ]
    )

    transaction_count = choose_column(
        columns,
        [
            "transaction_count",
            "tx_count"
        ]
    )

    unique_cards = choose_column(
        columns,
        [
            "unique_cards",
            "card_count"
        ]
    )

    expected_amount = choose_column(
        columns,
        [
            "expected_fraud_amount",
            "expected_amount"
        ]
    )

    baseline_expected_amount = choose_column(
        columns,
        [
            "baseline_expected_fraud_amount",
            "baseline_expected_amount"
        ]
    )

    verified = choose_column(
        columns,
        [
            "verified_spike",
            "verified"
        ]
    )

    candidate = choose_column(
        columns,
        [
            "candidate_spike",
            "candidate"
        ]
    )

    critical = choose_column(
        columns,
        [
            "critical_spike",
            "critical"
        ]
    )

    required = {

        "merchant_id":
            merchant,

        "timestamp":
            timestamp,

        "tas":
            tas,
    }

    missing = [
        key
        for key, value
        in required.items()
        if value is None
    ]

    if missing:

        die(
            f"Phase-2 {window}m window table "
            f"{path} is missing required fields: "
            f"{missing}. Available: {columns}"
        )

    mapping: dict[str, str | None] = {

        "merchant":
            merchant,

        "timestamp":
            timestamp,

        "tas":
            tas,

        "fas":
            fas,

        "verification":
            verification,

        "persistence":
            persistence,

        "corroboration":
            corroboration,

        "breadth":
            breadth,

        "materiality":
            materiality,

        "history":
            history,

        "transaction_count":
            transaction_count,

        "unique_cards":
            unique_cards,

        "expected_amount":
            expected_amount,

        "baseline_expected_amount":
            baseline_expected_amount,

        "verified":
            verified,

        "candidate":
            candidate,

        "critical":
            critical,
    }

    # --------------------------------------------------------
    # STABLE EVALUATION VIEW
    # --------------------------------------------------------

    expressions = [

        f"""
        CAST(
            {ident(merchant)}
            AS VARCHAR
        ) AS merchant_id
        """,

        f"""
        TRY_CAST(
            {ident(timestamp)}
            AS TIMESTAMP
        ) AS window_start
        """,

        f"""
        TRY_CAST(
            {ident(tas)}
            AS DOUBLE
        ) AS tas
        """,
    ]

    numeric_keys = [

        "fas",
        "verification",
        "persistence",
        "corroboration",
        "breadth",
        "materiality",
        "history",
        "transaction_count",
        "unique_cards",
        "expected_amount",
        "baseline_expected_amount",
    ]

    for key in numeric_keys:

        column = mapping[key]

        if column is None:

            expressions.append(
                f"NULL::DOUBLE AS {ident(key)}"
            )

        else:

            expressions.append(
                f"""
                TRY_CAST(
                    {ident(column)}
                    AS DOUBLE
                ) AS {ident(key)}
                """
            )

    boolean_keys = [
        "verified",
        "candidate",
        "critical"
    ]

    for key in boolean_keys:

        column = mapping[key]

        if column is None:

            expressions.append(
                f"FALSE AS {ident(key)}"
            )

        else:

            expressions.append(
                f"""
                COALESCE(
                    TRY_CAST(
                        {ident(column)}
                        AS BOOLEAN
                    ),
                    FALSE
                ) AS {ident(key)}
                """
            )

    eval_alias = f"eval_{window}m"

    con.execute(
        f"DROP VIEW IF EXISTS {ident(eval_alias)}"
    )

    con.execute(
        f"""
        CREATE TEMP VIEW {ident(eval_alias)} AS

        SELECT
            {', '.join(expressions)}

        FROM {ident(alias)}
        """
    )

    return (
        eval_alias,
        mapping
    )


# ============================================================
# AUC METRICS
# ============================================================

def auc_metrics(
    scores: np.ndarray,
    labels: np.ndarray
) -> dict[str, float]:

    scores = np.asarray(
        scores,
        dtype=float
    )

    labels = np.asarray(
        labels,
        dtype=int
    )

    if len(scores) != len(labels):

        return {
            "roc_auc": float("nan"),
            "pr_auc": float("nan")
        }

    mask = (
        np.isfinite(scores)
        &
        np.isfinite(labels)
    )

    scores = scores[mask]
    labels = labels[mask]

    positives = labels == 1
    negatives = labels == 0

    p = int(
        positives.sum()
    )

    n = int(
        negatives.sum()
    )

    if p == 0 or n == 0:

        return {
            "roc_auc": float("nan"),
            "pr_auc": float("nan")
        }

    order = np.argsort(
        -scores,
        kind="mergesort"
    )

    s = scores[order]
    y = labels[order]

    tp = np.cumsum(
        y == 1
    )

    fp = np.cumsum(
        y == 0
    )

    distinct = np.r_[
        True,
        s[1:] != s[:-1]
    ]

    idx = np.flatnonzero(
        distinct
    )

    tp = tp[idx]
    fp = fp[idx]

    tpr = tp / p
    fpr = fp / n

    roc = float(
        np.trapezoid(
            np.r_[
                0.0,
                tpr,
                1.0
            ],
            np.r_[
                0.0,
                fpr,
                1.0
            ]
        )
    )

    precision = (
        tp /
        np.maximum(
            tp + fp,
            1
        )
    )

    recall = tp / p

    pr = float(
        np.sum(
            np.diff(
                np.r_[
                    0.0,
                    recall
                ]
            )
            *
            precision
        )
    )

    return {
        "roc_auc": roc,
        "pr_auc": pr
    }


# ============================================================
# THRESHOLD GRID
# ============================================================

def threshold_grid(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float]
) -> list[dict[str, Any]]:

    rows: list[dict[str, Any]] = []

    scores = np.asarray(
        scores,
        dtype=float
    )

    labels = np.asarray(
        labels,
        dtype=int
    )

    for threshold in thresholds:

        pred = scores >= threshold

        tp = int(
            np.sum(
                (pred == 1)
                &
                (labels == 1)
            )
        )

        fp = int(
            np.sum(
                (pred == 1)
                &
                (labels == 0)
            )
        )

        fn = int(
            np.sum(
                (pred == 0)
                &
                (labels == 1)
            )
        )

        tn = int(
            np.sum(
                (pred == 0)
                &
                (labels == 0)
            )
        )

        precision = (
            tp /
            max(
                tp + fp,
                1
            )
        )

        recall = (
            tp /
            max(
                tp + fn,
                1
            )
        )

        f1 = (
            2 *
            precision *
            recall
            /
            max(
                precision + recall,
                1e-12
            )
        )

        rows.append({

            "threshold":
                threshold,

            "tp":
                tp,

            "fp":
                fp,

            "fn":
                fn,

            "tn":
                tn,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,
        })

    return rows


# ============================================================
# WINDOW METRICS
# ============================================================

def collect_window_metrics(
    con: duckdb.DuckDBPyConnection,
    window: int,
    eval_view: str,
    settings: dict[str, Any]
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]]
]:

    # --------------------------------------------------------
    # NON-15-MINUTE WINDOWS
    # --------------------------------------------------------

    if window != 15:

        count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM {ident(eval_view)}
            """
        ).fetchone()

        rows = (
            safe_int(count[0])
            if count
            else 0
        )

        return (

            {
                "window_minutes":
                    window,

                "status":
                    "diagnostic_only",

                "rows":
                    rows,
            },

            [],

            [],

            []
        )

    # --------------------------------------------------------
    # CREATE VALIDATION VIEW
    # --------------------------------------------------------

    con.execute(
        "DROP VIEW IF EXISTS validation_15"
    )

    con.execute(
        f"""
        CREATE TEMP VIEW validation_15 AS

        SELECT

            p.*,

            COALESCE(
                t.actual_fraud_window,
                FALSE
            ) AS actual_fraud_window,

            COALESCE(
                t.actual_repeated_episode_window,
                FALSE
            ) AS actual_repeated_episode_window,

            COALESCE(
                t.actual_multi_card_episode_window,
                FALSE
            ) AS actual_multi_card_episode_window,

            COALESCE(
                t.actual_coordinated_episode_window,
                FALSE
            ) AS actual_coordinated_episode_window,

            COALESCE(
                t.actual_fraud_count,
                0
            ) AS actual_fraud_count,

            COALESCE(
                t.actual_fraud_amount,
                0.0
            ) AS actual_fraud_amount

        FROM {ident(eval_view)} p

        LEFT JOIN truth_15m t

          ON p.merchant_id =
             t.merchant_id

         AND p.window_start =
             t.window_start
        """
    )

    count_row = con.execute(
        """
        SELECT COUNT(*)
        FROM validation_15
        """
    ).fetchone()

    total_rows = (
        safe_int(count_row[0])
        if count_row
        else 0
    )

    # --------------------------------------------------------
    # COMPONENT DIAGNOSTICS
    # --------------------------------------------------------

    component_sql = {

        "persistence":
            "persistence",

        "corroboration":
            "corroboration",

        "breadth":
            "breadth",

        "materiality":
            "materiality",
    }

    rows: list[
        dict[str, Any]
    ] = []

    for name, column in component_sql.items():

        result = con.execute(
            f"""
            SELECT

                AVG(
                    CASE
                        WHEN actual_coordinated_episode_window
                        THEN {ident(column)}
                    END
                ),

                AVG(
                    CASE
                        WHEN NOT actual_coordinated_episode_window
                        THEN {ident(column)}
                    END
                ),

                AVG(
                    {ident(column)}
                    *
                    CASE
                        WHEN actual_coordinated_episode_window
                        THEN 1
                        ELSE 0
                    END
                ),

                AVG(
                    {ident(column)}
                )

            FROM validation_15

            WHERE {ident(column)}
                IS NOT NULL
            """
        ).fetchone()

        if result is None:

            result = (
                None,
                None,
                None,
                None
            )

        fraud_mean = safe_float(
            result[0]
        )

        normal_mean = safe_float(
            result[1]
        )

        rows.append({

            "component":
                name,

            "mean_coordinated":
                fraud_mean,

            "mean_normal":
                normal_mean,

            "lift_difference":
                fraud_mean -
                normal_mean,

            "activation_mean":
                safe_float(
                    result[3]
                ),
        })

    # --------------------------------------------------------
    # CSS DATA
    # --------------------------------------------------------

    css_sql = """
        SELECT

            merchant_id,

            window_start,

            actual_coordinated_episode_window
                AS label,

            tas,

            COALESCE(
                persistence,
                0.0
            ) AS persistence,

            COALESCE(
                corroboration,
                0.0
            ) AS corroboration,

            COALESCE(
                breadth,
                0.0
            ) AS breadth,

            COALESCE(
                materiality,
                0.0
            ) AS materiality,

            COALESCE(
                verification,
                0.0
            ) AS verification,

            COALESCE(
                history,
                0.0
            ) AS history,

            COALESCE(
                transaction_count,
                0.0
            ) AS transaction_count,

            COALESCE(
                unique_cards,
                0.0
            ) AS unique_cards,

            COALESCE(
                expected_amount,
                0.0
            ) AS expected_amount

        FROM validation_15
    """

    data = con.execute(
        css_sql
    ).fetchall()

    columns = [

        "merchant_id",
        "window_start",
        "label",
        "tas",
        "persistence",
        "corroboration",
        "breadth",
        "materiality",
        "verification",
        "history",
        "transaction_count",
        "unique_cards",
        "expected_amount",
    ]

    idx = {
        name: i
        for i, name
        in enumerate(columns)
    }

    # --------------------------------------------------------
    # EMPTY DATA PROTECTION
    # --------------------------------------------------------

    if not data:

        summary = {

            "window_minutes":
                15,

            "rows":
                total_rows,

            "positive_coordinated_windows":
                0,

            "mean_css_coordinated":
                0.0,

            "mean_css_normal":
                0.0,

            "css_lift":
                0.0,

            "css_roc_auc":
                float("nan"),

            "css_pr_auc":
                float("nan"),

            "css_best_f1":
                0.0,

            "css_best_threshold":
                None,

            "independent_fas_roc_auc":
                float("nan"),

            "independent_fas_pr_auc":
                float("nan"),

            "production_fas_nonzero_windows":
                0,

            "production_fas_roc_auc":
                float("nan"),

            "production_fas_pr_auc":
                float("nan"),

            "production_fas_rankable":
                False,
        }

        return (
            summary,
            rows,
            [],
            []
        )

    # --------------------------------------------------------
    # NUMPY ARRAYS
    # --------------------------------------------------------

    label = np.array(
        [
            int(
                bool(
                    row[idx["label"]]
                )
            )
            for row in data
        ],
        dtype=int
    )

    tas = np.array(
        [
            safe_float(
                row[idx["tas"]]
            )
            for row in data
        ],
        dtype=float
    )

    persistence = np.array(
        [
            safe_float(
                row[idx["persistence"]]
            )
            for row in data
        ],
        dtype=float
    )

    corroboration = np.array(
        [
            safe_float(
                row[idx["corroboration"]]
            )
            for row in data
        ],
        dtype=float
    )

    breadth = np.array(
        [
            safe_float(
                row[idx["breadth"]]
            )
            for row in data
        ],
        dtype=float
    )

    materiality = np.array(
        [
            safe_float(
                row[idx["materiality"]]
            )
            for row in data
        ],
        dtype=float
    )

    verification = np.array(
        [
            safe_float(
                row[idx["verification"]]
            )
            for row in data
        ],
        dtype=float
    )

    # --------------------------------------------------------
    # CSS
    # --------------------------------------------------------

    tas_norm = np.clip(
        tas / 100.0,
        0.0,
        1.0
    )

    css = (

        0.30 *
        tas_norm

        +

        0.20 *
        np.clip(
            persistence,
            0.0,
            1.0
        )

        +

        0.20 *
        np.clip(
            corroboration,
            0.0,
            1.0
        )

        +

        0.15 *
        np.clip(
            breadth,
            0.0,
            1.0
        )

        +

        0.15 *
        np.clip(
            materiality,
            0.0,
            1.0
        )
    )

    css_eval = auc_metrics(
        css,
        label
    )

    css_grid = threshold_grid(
        css,
        label,
        [
            i / 100.0
            for i in range(1, 101)
        ]
    )

    best = (
        max(
            css_grid,
            key=lambda row: row["f1"]
        )
        if css_grid
        else None
    )

    # --------------------------------------------------------
    # INDEPENDENT FAS
    # --------------------------------------------------------

    gamma = safe_float(
        settings.get(
            "attack_gamma",
            1.0
        ),
        1.0
    )

    independent_fas = (

        (
            tas_norm
            ** gamma
        )

        *

        np.clip(
            verification,
            0.0,
            1.0
        )

        *

        np.clip(
            materiality,
            0.0,
            1.0
        )
    )

    fas_eval = auc_metrics(
        independent_fas,
        label
    )

    production_fas = np.where(
        verification >= 0.5,
        independent_fas,
        0.0
    )

    prod_nonzero = int(
        np.count_nonzero(
            production_fas > 0
        )
    )

    if prod_nonzero > 0:

        prod_fas_eval = auc_metrics(
            production_fas,
            label
        )

    else:

        prod_fas_eval = {

            "roc_auc":
                float("nan"),

            "pr_auc":
                float("nan"),
        }

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    fraud_rows = int(
        label.sum()
    )

    normal_rows = int(
        (label == 0).sum()
    )

    coord_mean_css = (

        float(
            css[label == 1].mean()
        )
        if fraud_rows > 0
        else 0.0
    )

    normal_mean_css = (

        float(
            css[label == 0].mean()
        )
        if normal_rows > 0
        else 0.0
    )

    summary = {

        "window_minutes":
            15,

        "rows":
            total_rows,

        "positive_coordinated_windows":
            fraud_rows,

        "mean_css_coordinated":
            coord_mean_css,

        "mean_css_normal":
            normal_mean_css,

        "css_lift":
            coord_mean_css -
            normal_mean_css,

        "css_roc_auc":
            css_eval["roc_auc"],

        "css_pr_auc":
            css_eval["pr_auc"],

        "css_best_f1":
            (
                best["f1"]
                if best
                else 0.0
            ),

        "css_best_threshold":
            (
                best["threshold"]
                if best
                else None
            ),

        "independent_fas_roc_auc":
            fas_eval["roc_auc"],

        "independent_fas_pr_auc":
            fas_eval["pr_auc"],

        "production_fas_nonzero_windows":
            prod_nonzero,

        "production_fas_roc_auc":
            prod_fas_eval["roc_auc"],

        "production_fas_pr_auc":
            prod_fas_eval["pr_auc"],

        "production_fas_rankable":
            prod_nonzero > 0,
    }

    # --------------------------------------------------------
    # TOP CSS WINDOWS
    # --------------------------------------------------------

    top_indices = np.argsort(
        -css
    )[:20]

    top_rows: list[
        dict[str, Any]
    ] = []

    for position in top_indices:

        position = int(
            position
        )

        row = data[position]

        top_rows.append({

            "merchant_id":
                row[idx["merchant_id"]],

            "window_start":
                row[idx["window_start"]],

            "css":
                float(css[position]),

            "label":
                int(label[position]),

            "tas":
                float(tas[position]),

            "persistence":
                float(
                    persistence[position]
                ),

            "corroboration":
                float(
                    corroboration[position]
                ),

            "breadth":
                float(
                    breadth[position]
                ),

            "materiality":
                float(
                    materiality[position]
                ),

            "verification":
                float(
                    verification[position]
                ),
        })

    # --------------------------------------------------------
    # FOUR-ITEM RETURN
    # --------------------------------------------------------

    component_output = (
        rows
        +
        [
            {
                "component":
                    "CSS",
                **summary
            }
        ]
    )

    return (
        summary,
        component_output,
        top_rows,
        css_grid
    )


# ============================================================
# EVENT OUTPUT VALIDATION
# ============================================================

def evaluate_event_output(
    con: duckdb.DuckDBPyConnection,
    events_path: Path,
    settings: dict[str, Any]
) -> dict[str, Any]:

    relation = relation_sql(
        events_path
    )

    columns = describe_relation(
        con,
        relation
    )

    merchant = choose_column(
        columns,
        [
            "merchant_id",
            "merchant"
        ]
    )

    start = choose_column(
        columns,
        [
            "detection_time",
            "event_start_time",
            "start_time"
        ]
    )

    end = choose_column(
        columns,
        [
            "event_end_time",
            "end_time"
        ]
    )

    peak = choose_column(
        columns,
        [
            "peak_time",
            "peak_timestamp"
        ]
    )

    expected = choose_column(
        columns,
        [
            "expected_fraud_amount",
            "expected_amount"
        ]
    )

    transaction_count = choose_column(
        columns,
        [
            "transaction_count",
            "fraud_transaction_count"
        ]
    )

    peak_score = choose_column(
        columns,
        [
            "peak_fraud_attack_score",
            "peak_risk_score",
            "fraud_attack_score",
            "peak_tas"
        ]
    )

    required = {

        "merchant_id":
            merchant,

        "detection_time":
            start,
    }

    missing = [
        key
        for key, value
        in required.items()
        if value is None
    ]

    if missing:

        die(
            f"Event file {events_path} "
            f"is missing required fields: "
            f"{missing}. "
            f"Available: {columns}"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Use ident() for COLUMN NAMES,
    # not sql_quote().
    # --------------------------------------------------------

    detection_expr = (
        f"""
        TRY_CAST(
            {ident(start)}
            AS TIMESTAMP
        )
        """
    )

    expected_expr = (

        f"""
        TRY_CAST(
            {ident(expected)}
            AS DOUBLE
        )
        """

        if expected

        else
        "0.0"
    )

    tx_expr = (

        f"""
        TRY_CAST(
            {ident(transaction_count)}
            AS DOUBLE
        )
        """

        if transaction_count

        else
        "NULL"
    )

    peak_expr = (

        f"""
        TRY_CAST(
            {ident(peak_score)}
            AS DOUBLE
        )
        """

        if peak_score

        else
        "NULL"
    )

    end_expr = (

        f"""
        TRY_CAST(
            {ident(end)}
            AS TIMESTAMP
        )
        """

        if end

        else
        detection_expr
    )

    peak_expr_time = (

        f"""
        TRY_CAST(
            {ident(peak)}
            AS TIMESTAMP
        )
        """

        if peak

        else
        detection_expr
    )

    # --------------------------------------------------------
    # EVENT VIEW
    # --------------------------------------------------------

    con.execute(
        "DROP VIEW IF EXISTS phase2_events"
    )

    con.execute(
        f"""
        CREATE TEMP VIEW phase2_events AS

        SELECT

            CAST(
                {ident(merchant)}
                AS VARCHAR
            ) AS merchant_id,

            {detection_expr}
                AS detection_time,

            {end_expr}
                AS event_end_time,

            {peak_expr_time}
                AS peak_time,

            {expected_expr}
                AS expected_fraud_amount,

            {tx_expr}
                AS transaction_count,

            {peak_expr}
                AS peak_score

        FROM {relation}
        """
    )

    # --------------------------------------------------------
    # EVENT SUMMARY
    # --------------------------------------------------------

    row = con.execute(
        """
        SELECT

            COUNT(*) AS events,

            COUNT(
                DISTINCT merchant_id
            ) AS merchants,

            AVG(
                CASE
                    WHEN COALESCE(
                        transaction_count,
                        0
                    ) <= 1
                    THEN 1.0
                    ELSE 0.0
                END
            ) AS single_event_rate,

            AVG(
                COALESCE(
                    expected_fraud_amount,
                    0.0
                )
            ) AS mean_expected_fraud_amount,

            SUM(
                COALESCE(
                    expected_fraud_amount,
                    0.0
                )
            ) AS total_expected_fraud_amount,

            AVG(
                DATE_DIFF(
                    'second',
                    detection_time,
                    event_end_time
                ) / 60.0
            ) AS mean_duration_minutes,

            MAX(
                peak_score
            ) AS max_peak_score

        FROM phase2_events
        """
    ).fetchone()

    if row is None:

        row = (
            0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        )

    gt_row = con.execute(
        """
        SELECT COUNT(*)
        FROM fraud_episodes
        """
    ).fetchone()

    gt_events = (
        safe_int(gt_row[0])
        if gt_row
        else 0
    )

    predicted_events = safe_int(
        row[0]
    )

    # --------------------------------------------------------
    # EVENT MATCHING
    # --------------------------------------------------------

    con.execute(
        "DROP VIEW IF EXISTS event_candidates"
    )

    tolerance = max(
        0.0,
        safe_float(
            settings.get(
                "match_tolerance",
                60.0
            ),
            60.0
        )
    )

    con.execute(
        f"""
        CREATE TEMP VIEW event_candidates AS

        SELECT

            p.merchant_id,

            p.detection_time,

            f.start_time
                AS fraud_start_time,

            ABS(
                DATE_DIFF(
                    'second',
                    f.start_time,
                    p.detection_time
                )
            ) / 60.0
                AS delay_minutes,

            ROW_NUMBER() OVER (

                PARTITION BY
                    p.merchant_id,
                    p.detection_time

                ORDER BY
                    ABS(
                        DATE_DIFF(
                            'second',
                            f.start_time,
                            p.detection_time
                        )
                    )

            ) AS rn

        FROM phase2_events p

        JOIN fraud_episodes f

          ON p.merchant_id =
             f.merchant_id

         AND f.start_time
             BETWEEN
                p.detection_time
                -
                INTERVAL '{tolerance}' MINUTE

                AND

                p.detection_time
                +
                INTERVAL '{tolerance}' MINUTE

        WHERE
            p.detection_time IS NOT NULL
        """
    )

    matches = con.execute(
        """
        SELECT *
        FROM event_candidates
        WHERE rn = 1
        """
    ).fetchall()

    matched = len(
        matches
    )

    precision = (
        matched /
        max(
            predicted_events,
            1
        )
    )

    recall = (
        matched /
        max(
            gt_events,
            1
        )
    )

    delays = [

        safe_float(
            row_match[3],
            0.0
        )

        for row_match
        in matches
    ]

    return {

        "events":
            predicted_events,

        "merchants":
            safe_int(row[1]),

        "ground_truth_episodes":
            gt_events,

        "matched_events":
            matched,

        "event_precision":
            precision,

        "event_recall":
            recall,

        "mean_detection_delay_minutes":
            (
                float(
                    np.mean(delays)
                )
                if delays
                else None
            ),

        "single_event_rate":
            safe_float(row[2]),

        "mean_expected_fraud_amount":
            safe_float(row[3]),

        "total_expected_fraud_amount":
            safe_float(row[4]),

        "mean_duration_minutes":
            safe_float(row[5]),

        "max_peak_score":
            safe_float(row[6]),
    }


# ============================================================
# PERSISTENCE BACKTEST
# ============================================================

def persistence_backtest(
    con: duckdb.DuckDBPyConnection
) -> dict[str, Any]:

    row = con.execute(
        """
        SELECT

            AVG(
                CASE
                    WHEN actual_coordinated_episode_window
                    THEN COALESCE(
                        persistence,
                        0
                    )
                    ELSE NULL
                END
            ) AS coordinated_mean,

            AVG(
                CASE
                    WHEN NOT actual_coordinated_episode_window
                    THEN COALESCE(
                        persistence,
                        0
                    )
                    ELSE NULL
                END
            ) AS normal_mean,

            CORR(
                COALESCE(
                    persistence,
                    0
                ),
                CAST(
                    actual_coordinated_episode_window
                    AS INTEGER
                )
            ) AS correlation

        FROM validation_15
        """
    ).fetchone()

    if row is None:

        return {

            "mean_persistence_coordinated":
                0.0,

            "mean_persistence_normal":
                0.0,

            "persistence_lift":
                0.0,

            "persistence_label_correlation":
                0.0,
        }

    coordinated = safe_float(
        row[0]
    )

    normal = safe_float(
        row[1]
    )

    return {

        "mean_persistence_coordinated":
            coordinated,

        "mean_persistence_normal":
            normal,

        "persistence_lift":
            coordinated -
            normal,

        "persistence_label_correlation":
            safe_float(
                row[2]
            ),
    }


# ============================================================
# MULTISCALE DIAGNOSTIC
# ============================================================

def multiscale_diagnostic(
    con: duckdb.DuckDBPyConnection,
    available: dict[int, Path]
) -> dict[str, Any]:

    required = {
        5,
        15,
        60
    }

    if not required.issubset(
        set(available)
    ):

        return {

            "status":
                "incomplete",

            "available_windows":
                sorted(
                    available
                ),
        }

    results: dict[
        str,
        Any
    ] = {

        "status":
            "complete",

        "available_windows":
            [
                5,
                15,
                60
            ],
    }

    for window in [
        5,
        15,
        60
    ]:

        alias, _ = normalise_phase2_windows(
            con,
            available[window],
            window
        )

        count_row = con.execute(
            f"""
            SELECT COUNT(*)
            FROM {ident(alias)}
            """
        ).fetchone()

        max_row = con.execute(
            f"""
            SELECT MAX(tas)
            FROM {ident(alias)}
            """
        ).fetchone()

        results[
            f"rows_{window}m"
        ] = (
            safe_int(
                count_row[0]
            )
            if count_row
            else 0
        )

        results[
            f"max_tas_{window}m"
        ] = (
            safe_float(
                max_row[0]
            )
            if max_row
            else 0.0
        )

    return results


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "FraudSentinel AI Phase-2 "
            "validation using DuckDB"
        )
    )

    parser.add_argument(
        "--raw-data",
        default=None,
        help=(
            "Raw Phase-1 transaction "
            "CSV/Parquet"
        )
    )

    parser.add_argument(
        "--risk-output",
        default=None,
        help=(
            "Phase-1 transaction "
            "risk output CSV/Parquet"
        )
    )

    parser.add_argument(
        "--phase2-dir",
        default=None,
        help=(
            "Phase-2 artifacts directory"
        )
    )

    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Phase-2 YAML configuration"
        )
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Validation output directory"
        )
    )

    parser.add_argument(
        "--project-root",
        default=".",
        help=(
            "FraudSentinel project root"
        )
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Overwrite validation outputs"
        )
    )

    args = parser.parse_args()

    started = time.perf_counter()

    root = Path(
        args.project_root
    ).resolve()

    output_dir = (

        Path(
            args.output_dir
        ).resolve()

        if args.output_dir

        else

        root /
        "artifacts" /
        "phase2" /
        "validation_final"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    report: dict[
        str,
        Any
    ] = {

        "project":
            "FraudSentinel AI",

        "validator":
            "phase2_validation_duckdb_final",

        "status":
            "PASS",

        "sections":
            {},

        "errors":
            [],
    }

    con: (
        duckdb.DuckDBPyConnection
        | None
    ) = None

    try:

        # ----------------------------------------------------
        # INPUT DISCOVERY
        # ----------------------------------------------------

        (
            raw_path,
            risk_path,
            phase2_dir,
            config_path
        ) = discover_paths(
            args
        )

        config = read_yaml(
            config_path
        )

        settings = load_settings(
            config
        )

        report[
            "sections"
        ][
            "INPUTS"
        ] = {

            "raw_data":
                str(raw_path),

            "risk_output":
                str(risk_path),

            "phase2_dir":
                str(phase2_dir),

            "config":
                (
                    str(config_path)
                    if config_path.exists()
                    else
                    "NOT_FOUND_DEFAULTS_USED"
                ),
        }

        # ----------------------------------------------------
        # DUCKDB
        # ----------------------------------------------------

        con = connect()

        # ----------------------------------------------------
        # PHASE-1 AUDIT
        # ----------------------------------------------------

        (
            phase1_result,
            raw_rel,
            risk_rel
        ) = phase1_schema_audit(
            con,
            raw_path,
            risk_path
        )

        report[
            "sections"
        ][
            "PHASE1_AUDIT"
        ] = phase1_result

        # ----------------------------------------------------
        # GROUND TRUTH
        # ----------------------------------------------------

        truth_result = build_ground_truth(
            con,
            raw_rel,
            settings
        )

        report[
            "sections"
        ][
            "GROUND_TRUTH"
        ] = truth_result

        # ----------------------------------------------------
        # 15-MINUTE TRUTH
        # ----------------------------------------------------

        create_truth_15m(
            con
        )

        # ----------------------------------------------------
        # PHASE-2 ARTIFACT DISCOVERY
        # ----------------------------------------------------

        available = discover_phase2_windows(
            phase2_dir
        )

        if 15 not in available:

            die(
                "15-minute Phase-2 window "
                "artifact was not found; "
                "it is required for full evaluation."
            )

        # ----------------------------------------------------
        # NORMALISE WINDOWS
        # ----------------------------------------------------

        eval_views: dict[
            int,
            str
        ] = {}

        mappings: dict[
            int,
            dict[str, str | None]
        ] = {}

        for window, path in sorted(
            available.items()
        ):

            view, mapping = (
                normalise_phase2_windows(
                    con,
                    path,
                    window
                )
            )

            eval_views[
                window
            ] = view

            mappings[
                window
            ] = mapping

        # ----------------------------------------------------
        # 15-MINUTE EVALUATION
        # ----------------------------------------------------

        (
            fifteen_summary,
            component_rows,
            top_rows,
            css_grid
        ) = collect_window_metrics(
            con,
            15,
            eval_views[15],
            settings
        )

        report[
            "sections"
        ][
            "CSS_FAS"
        ] = fifteen_summary

        report[
            "sections"
        ][
            "COMPONENTS"
        ] = component_rows

        # ----------------------------------------------------
        # PERSISTENCE
        # ----------------------------------------------------

        report[
            "sections"
        ][
            "PERSISTENCE"
        ] = persistence_backtest(
            con
        )

        # ----------------------------------------------------
        # MULTISCALE
        # ----------------------------------------------------

        report[
            "sections"
        ][
            "MULTISCALE"
        ] = multiscale_diagnostic(
            con,
            available
        )

        # ----------------------------------------------------
        # EVENT VALIDATION
        # ----------------------------------------------------

        events_path = discover_events(
            phase2_dir
        )

        if events_path is not None:

            report[
                "sections"
            ][
                "EVENT_VALIDATION"
            ] = evaluate_event_output(
                con,
                events_path,
                settings
            )

        else:

            report[
                "sections"
            ][
                "EVENT_VALIDATION"
            ] = {

                "status":
                    "SKIPPED",

                "reason":
                    (
                        "No event artifact "
                        "found under "
                        "artifacts/phase2/events"
                    ),
            }

        # ----------------------------------------------------
        # PRODUCTION SANITY
        # ----------------------------------------------------

        verification_row = con.execute(
            """
            SELECT

                COUNT(*) AS rows,

                COUNT(*) FILTER (
                    WHERE verified
                ) AS verified_windows,

                COUNT(*) FILTER (
                    WHERE candidate
                ) AS candidate_windows,

                COUNT(*) FILTER (
                    WHERE critical
                ) AS critical_windows,

                COUNT(*) FILTER (
                    WHERE
                        tas < 0
                        OR tas > 100
                ) AS invalid_tas

            FROM validation_15
            """
        ).fetchone()

        if verification_row is None:

            raise ValidationError(
                "Unable to calculate "
                "Phase-2 production sanity metrics."
            )

        invalid_tas = safe_int(
            verification_row[4]
        )

        if invalid_tas > 0:

            raise ValidationError(
                "Phase-2 TAS contains values "
                "outside [0,100]."
            )

        report[
            "sections"
        ][
            "PRODUCTION_SANITY"
        ] = {

            "rows_15m":
                safe_int(
                    verification_row[0]
                ),

            "verified_windows":
                safe_int(
                    verification_row[1]
                ),

            "candidate_windows":
                safe_int(
                    verification_row[2]
                ),

            "critical_windows":
                safe_int(
                    verification_row[3]
                ),

            "invalid_tas":
                invalid_tas,
        }

        # ----------------------------------------------------
        # OUTPUT CSV FILES
        # ----------------------------------------------------

        write_csv(
            output_dir /
            "component_diagnostics.csv",
            component_rows
        )

        write_csv(
            output_dir /
            "top_css_windows.csv",
            top_rows
        )

        write_csv(
            output_dir /
            "css_threshold_grid.csv",
            css_grid
        )

    # ========================================================
    # EXPECTED VALIDATION FAILURE
    # ========================================================

    except ValidationError as exc:

        report[
            "status"
        ] = "FAIL"

        report[
            "errors"
        ].append(
            str(exc)
        )

        print(
            f"\nVALIDATION FAILED: {exc}",
            file=sys.stderr
        )

        traceback.print_exc()

        return_code = 1

    # ========================================================
    # UNEXPECTED ERROR
    # ========================================================

    except Exception as exc:

        report[
            "status"
        ] = "ERROR"

        report[
            "errors"
        ].append(
            f"{type(exc).__name__}: {exc}"
        )

        print(
            f"\nUNEXPECTED ERROR: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr
        )

        traceback.print_exc()

        return_code = 2

    # ========================================================
    # CLEANUP
    # ========================================================

    else:

        return_code = 0

    finally:

        if con is not None:

            try:
                con.close()

            except Exception:
                pass

        report[
            "runtime_seconds"
        ] = round(
            time.perf_counter()
            - started,
            3
        )

        write_json(
            output_dir /
            "phase2_validation_final.json",
            report
        )

        write_text_report(
            output_dir /
            "phase2_validation_final.txt",
            report
        )

    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "PHASE 2 VALIDATION"
    )

    print(
        "=" * 72
    )

    print(
        f"Status: {report['status']}"
    )

    print(
        "Runtime: "
        f"{report['runtime_seconds']:.2f} sec"
    )

    section_order = [

        "INPUTS",

        "PHASE1_AUDIT",

        "GROUND_TRUTH",

        "CSS_FAS",

        "COMPONENTS",

        "PERSISTENCE",

        "MULTISCALE",

        "EVENT_VALIDATION",

        "PRODUCTION_SANITY",
    ]

    for section_name in section_order:

        section = (
            report[
                "sections"
            ].get(
                section_name
            )
        )

        if not section:
            continue

        print(
            f"\n[{section_name}]"
        )

        if isinstance(
            section,
            dict
        ):

            for key, value in section.items():

                print(
                    f"{key}: {value}"
                )

        elif isinstance(
            section,
            list
        ):

            print(
                f"items: {len(section)}"
            )

        else:

            print(
                str(section)
            )

    if report["errors"]:

        print(
            "\n[ERRORS]"
        )

        for error in report[
            "errors"
        ]:

            print(
                error
            )

    print(
        f"\nReports written to: "
        f"{output_dir}"
    )

    return return_code


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )


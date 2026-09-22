from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence


from src.mlops.config import (
    PROJECT_ROOT,
    PIPELINE_VERSION,
    COMPONENT_VERSIONS,
    RUNS_DIR,
)

from src.mlops.logging.logger import get_logger


logger = get_logger("InferencePipeline")


# ============================================================
# PIPELINE STAGES
# ============================================================

@dataclass(frozen=True)
class PipelineStage:
    """
    Definition of one executable pipeline stage.
    """

    name: str
    module: str
    version: str
    artifacts: tuple[str, ...]


PIPELINE_STAGES: tuple[PipelineStage, ...] = (

    # ========================================================
    # PHASE 1 — TRANSACTION INTELLIGENCE
    # ========================================================

    PipelineStage(
        name="phase1",
        module="scripts.run_phase1",
        version=COMPONENT_VERSIONS["phase1"],
        artifacts=(
            "artifacts/phase1/risk_output/transaction_risk.csv",
        ),
    ),

    # ========================================================
    # PHASE 2 — TEMPORAL FRAUD-SPIKE INTELLIGENCE
    # ========================================================

    PipelineStage(
        name="phase2",
        module="scripts.run_phase2",
        version=COMPONENT_VERSIONS["phase2"],
        artifacts=(
            "artifacts/phase2/phase2_windows.csv",
            "artifacts/phase2/phase2_events.csv",
        ),
    ),

    # ========================================================
    # FUSION
    # ========================================================

    PipelineStage(
        name="fusion",
        module="scripts.run_fusion",
        version=COMPONENT_VERSIONS["fusion"],
        artifacts=(
            "artifacts/fusion/fraudsentinel_unified_risk.csv",
            "artifacts/fusion/fusion_events.csv",
        ),
    ),

    # ========================================================
    # XAI
    # ========================================================

    PipelineStage(
        name="xai",
        module="scripts.run_xai",
        version=COMPONENT_VERSIONS["xai"],
        artifacts=(
            "artifacts/xai/transaction_explanations.csv",
            "artifacts/xai/spike_explanations.csv",
            "artifacts/xai/unified_explanations.csv",
        ),
    ),

    # ========================================================
    # IMPACT / LOSS FORECAST
    # ========================================================

    PipelineStage(
        name="impact",
        module="scripts.run_impact",
        version=COMPONENT_VERSIONS["impact"],
        artifacts=(
            "artifacts/impact/impact_forecast.csv",
            "artifacts/impact/impact_scenarios.csv",
        ),
    ),

    # ========================================================
    # RESPONSE RECOMMENDATION
    # ========================================================

    PipelineStage(
        name="response",
        module="scripts.run_response",
        version=COMPONENT_VERSIONS["response"],
        artifacts=(
            "artifacts/response/response_recommendations.csv",
            "artifacts/response/response_actions.csv",
        ),
    ),
)


# ============================================================
# STAGE RESULT
# ============================================================

@dataclass
class StageResult:
    """
    Runtime result of one pipeline stage.
    """

    name: str
    module: str
    version: str

    status: str

    started_at: str
    completed_at: str

    duration_seconds: float

    return_code: int

    artifacts: Dict[str, Dict]

    stdout: str = ""
    stderr: str = ""

    error: Optional[str] = None


# ============================================================
# PIPELINE RESULT
# ============================================================

@dataclass
class PipelineResult:
    """
    Complete pipeline execution result.
    """

    project: str

    pipeline_version: str

    run_id: str

    status: str

    started_at: str
    completed_at: str

    duration_seconds: float

    stages: List[Dict]

    failed_stage: Optional[str]

    python_version: str

    executable: str


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> str:
    """
    Return the current UTC timestamp in ISO-8601 format.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


def resolve_artifact(
    relative_path: str,
) -> Path:
    """
    Resolve an artifact path against the project root.
    """

    return (
        PROJECT_ROOT
        / relative_path
    )


def artifact_exists(
    relative_path: str,
) -> bool:
    """
    Check whether an artifact exists and is non-empty.
    """

    path = resolve_artifact(
        relative_path
    )

    return (
        path.exists()
        and path.is_file()
        and path.stat().st_size > 0
    )


def inspect_artifacts(
    artifact_paths: Sequence[str],
) -> Dict[str, Dict]:
    """
    Inspect expected artifacts produced by a stage.
    """

    results: Dict[str, Dict] = {}

    for relative_path in artifact_paths:

        path = resolve_artifact(
            relative_path
        )

        exists = path.exists()

        size_bytes = (
            path.stat().st_size
            if exists
            else 0
        )

        results[relative_path] = {
            "path": str(path),
            "exists": bool(exists),
            "size_bytes": int(size_bytes),
            "status": (
                "PASS"
                if (
                    exists
                    and path.is_file()
                    and size_bytes > 0
                )
                else "FAIL"
            ),
        }

    return results


def artifacts_pass(
    artifacts: Dict[str, Dict],
) -> bool:
    """
    Return True only when every expected artifact passes.
    """

    if not artifacts:
        return False

    return all(
        info["status"] == "PASS"
        for info in artifacts.values()
    )


def validate_stage_inputs(
    stage: PipelineStage,
    extra_args: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """
    Validate required inputs before launching a stage.

    Currently Phase 2 requires the Phase 1 transaction-risk
    artifact. This prevents the subprocess from being launched
    with a missing dependency.
    """

    if stage.name == "phase2":

        phase1_input = resolve_artifact(
            "artifacts/phase1/risk_output/transaction_risk.csv"
        )

        if not (
            phase1_input.exists()
            and phase1_input.is_file()
            and phase1_input.stat().st_size > 0
        ):

            return (
                "Phase 2 input artifact is missing or empty: "
                f"{phase1_input}"
            )

    return None


def build_stage_arguments(
    stage: PipelineStage,
) -> List[str]:
    """
    Build command-line arguments required by each stage.

    Phase 2 consumes the transaction-level risk output
    generated by Phase 1.
    """

    arguments: List[str] = []

    # ========================================================
    # PHASE 2 INPUT
    # ========================================================

    if stage.name == "phase2":

        phase1_input = resolve_artifact(
            "artifacts/phase1/risk_output/transaction_risk.csv"
        )

        arguments.extend(
            [
                "--phase1-input",
                str(phase1_input),
            ]
        )

    return arguments


# ============================================================
# RUN SINGLE STAGE
# ============================================================

def run_stage(
    stage: PipelineStage,
    extra_args: Optional[Sequence[str]] = None,
) -> StageResult:
    """
    Execute one pipeline stage as a Python module.
    """

    stage_started = utc_now()

    start_time = time.perf_counter()

    logger.info(
        "=================================================="
    )

    logger.info(
        "Starting stage: %s",
        stage.name,
    )

    logger.info(
        "Module: %s",
        stage.module,
    )

    logger.info(
        "Version: %s",
        stage.version,
    )

    # ========================================================
    # VALIDATE INPUTS
    # ========================================================

    input_error = validate_stage_inputs(
        stage,
        extra_args,
    )

    if input_error is not None:

        duration = (
            time.perf_counter()
            - start_time
        )

        logger.error(
            "Stage input validation failed: %s",
            input_error,
        )

        return StageResult(
            name=stage.name,
            module=stage.module,
            version=stage.version,
            status="FAIL",
            started_at=stage_started,
            completed_at=utc_now(),
            duration_seconds=float(
                duration
            ),
            return_code=-1,
            artifacts=inspect_artifacts(
                stage.artifacts
            ),
            stdout="",
            stderr="",
            error=input_error,
        )

    # ========================================================
    # BUILD COMMAND
    # ========================================================

    command: List[str] = [
        sys.executable,
        "-m",
        stage.module,
    ]

    if extra_args:

        command.extend(
            list(extra_args)
        )

    logger.info(
        "Command: %s",
        " ".join(
            f'"{arg}"'
            if " " in arg
            else arg
            for arg in command
        ),
    )

    # ========================================================
    # EXECUTE
    # ========================================================

    try:

        process = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        stdout = (
            process.stdout
            or ""
        )

        stderr = (
            process.stderr
            or ""
        )

        # ====================================================
        # LOG STDOUT
        # ====================================================

        if stdout:

            logger.info(
                "[%s] stdout:\n%s",
                stage.name.upper(),
                stdout.rstrip(),
            )

        # ====================================================
        # LOG STDERR
        # ====================================================

        if stderr:

            logger.warning(
                "[%s] stderr:\n%s",
                stage.name.upper(),
                stderr.rstrip(),
            )

        # ====================================================
        # INSPECT ARTIFACTS
        # ====================================================

        artifacts = inspect_artifacts(
            stage.artifacts
        )

        duration = (
            time.perf_counter()
            - start_time
        )

        # ====================================================
        # DETERMINE STATUS
        # ====================================================

        if process.returncode != 0:

            status = "FAIL"

            error = (
                "Runner exited with "
                f"return code "
                f"{process.returncode}."
            )

        elif not artifacts_pass(
            artifacts
        ):

            status = "FAIL"

            missing = [
                path
                for path, info
                in artifacts.items()
                if info["status"] != "PASS"
            ]

            error = (
                "Expected artifacts missing "
                f"or empty: {missing}"
            )

        else:

            status = "PASS"

            error = None

        completed_at = utc_now()

        logger.info(
            "Stage %s → %s | %.2f seconds",
            stage.name,
            status,
            duration,
        )

        return StageResult(
            name=stage.name,
            module=stage.module,
            version=stage.version,
            status=status,
            started_at=stage_started,
            completed_at=completed_at,
            duration_seconds=float(
                duration
            ),
            return_code=int(
                process.returncode
            ),
            artifacts=artifacts,
            stdout=stdout,
            stderr=stderr,
            error=error,
        )

    except Exception as exc:

        duration = (
            time.perf_counter()
            - start_time
        )

        logger.exception(
            "Stage %s failed with exception.",
            stage.name,
        )

        return StageResult(
            name=stage.name,
            module=stage.module,
            version=stage.version,
            status="FAIL",
            started_at=stage_started,
            completed_at=utc_now(),
            duration_seconds=float(
                duration
            ),
            return_code=-1,
            artifacts=inspect_artifacts(
                stage.artifacts
            ),
            stdout="",
            stderr="",
            error=str(exc),
        )


# ============================================================
# SAVE PIPELINE RESULT
# ============================================================

def save_pipeline_result(
    result: PipelineResult,
) -> Path:
    """
    Persist pipeline execution metadata.
    """

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RUNS_DIR
        / f"{result.run_id}.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            asdict(result),
            file,
            indent=2,
        )

    return output_path


# ============================================================
# RUN COMPLETE PIPELINE
# ============================================================

def run_inference_pipeline(
    stop_on_failure: bool = True,
) -> PipelineResult:
    """
    Execute the complete FraudSentinel AI inference pipeline.

    Stage order:

        Phase 1
          ↓
        Phase 2
          ↓
        Fusion
          ↓
        XAI
          ↓
        Impact
          ↓
        Response
    """

    # ========================================================
    # CREATE RUN ID
    # ========================================================

    run_id = (
        datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        + "_"
        + uuid.uuid4().hex[:8]
    )

    pipeline_started = utc_now()

    start_time = time.perf_counter()

    # ========================================================
    # PIPELINE HEADER
    # ========================================================

    logger.info("")

    logger.info(
        "=================================================="
    )

    logger.info(
        " FraudSentinel AI — Inference Pipeline"
    )

    logger.info(
        "=================================================="
    )

    logger.info(
        "Run ID: %s",
        run_id,
    )

    logger.info(
        "Pipeline version: %s",
        PIPELINE_VERSION,
    )

    logger.info(
        "Project root: %s",
        PROJECT_ROOT,
    )

    logger.info(
        "Python executable: %s",
        sys.executable,
    )

    logger.info(
        "=================================================="
    )

    # ========================================================
    # STAGE RESULTS
    # ========================================================

    stage_results: List[StageResult] = []

    failed_stage: Optional[str] = None

    # ========================================================
    # EXECUTE STAGES
    # ========================================================

    for stage in PIPELINE_STAGES:

        # ----------------------------------------------------
        # Build stage-specific arguments
        # ----------------------------------------------------

        extra_args = build_stage_arguments(
            stage
        )

        # ----------------------------------------------------
        # Log dependency information
        # ----------------------------------------------------

        if stage.name == "phase2":

            phase1_input = resolve_artifact(
                "artifacts/phase1/risk_output/transaction_risk.csv"
            )

            logger.info(
                "Phase 2 input: %s",
                phase1_input,
            )

        # ----------------------------------------------------
        # Execute stage
        # ----------------------------------------------------

        result = run_stage(
            stage,
            extra_args=extra_args,
        )

        stage_results.append(
            result
        )

        # ----------------------------------------------------
        # Handle failure
        # ----------------------------------------------------

        if result.status != "PASS":

            failed_stage = stage.name

            logger.error(
                "Pipeline stage FAILED: %s",
                stage.name,
            )

            if stop_on_failure:

                logger.error(
                    "Stopping pipeline because "
                    "stop_on_failure=True."
                )

                break

    # ========================================================
    # PIPELINE DURATION
    # ========================================================

    duration = (
        time.perf_counter()
        - start_time
    )

    # ========================================================
    # PIPELINE STATUS
    # ========================================================

    pipeline_status = (
        "PASS"
        if (
            failed_stage is None
            and len(stage_results)
            == len(PIPELINE_STAGES)
            and all(
                stage.status == "PASS"
                for stage in stage_results
            )
        )
        else "FAIL"
    )

    pipeline_completed = utc_now()

    # ========================================================
    # CREATE PIPELINE RESULT
    # ========================================================

    result = PipelineResult(
        project="FraudSentinel AI",
        pipeline_version=PIPELINE_VERSION,
        run_id=run_id,
        status=pipeline_status,
        started_at=pipeline_started,
        completed_at=pipeline_completed,
        duration_seconds=float(
            duration
        ),
        stages=[
            asdict(stage)
            for stage in stage_results
        ],
        failed_stage=failed_stage,
        python_version=sys.version,
        executable=sys.executable,
    )

    # ========================================================
    # SAVE RUN METADATA
    # ========================================================

    output_path = save_pipeline_result(
        result
    )

    # ========================================================
    # PIPELINE SUMMARY
    # ========================================================

    logger.info("")

    logger.info(
        "=================================================="
    )

    logger.info(
        " PIPELINE COMPLETE"
    )

    logger.info(
        "=================================================="
    )

    logger.info(
        "Run ID        : %s",
        run_id,
    )

    logger.info(
        "Status        : %s",
        pipeline_status,
    )

    logger.info(
        "Duration      : %.2f seconds",
        duration,
    )

    logger.info(
        "Failed stage  : %s",
        failed_stage or "None",
    )

    logger.info(
        "Run metadata  : %s",
        output_path,
    )

    logger.info(
        "=================================================="
    )

    return result


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def main() -> int:
    """
    CLI entry point for the inference pipeline.
    """

    result = run_inference_pipeline(
        stop_on_failure=True
    )

    if result.status == "PASS":

        return 0

    return 1


# ============================================================
# MODULE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )
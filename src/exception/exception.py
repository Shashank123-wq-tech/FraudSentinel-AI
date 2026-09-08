"""
FraudSentinel AI
Application Exception Hierarchy
"""

from __future__ import annotations

from typing import Optional


class FraudSentinelException(Exception):
    """
    Base exception for expected FraudSentinel AI failures.

    Supports optional exception chaining metadata while remaining
    backward compatible with callers that pass only a message.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "FRAUDSENTINEL_ERROR",
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)

        self.message = message
        self.code = code
        self.cause = cause

    def __str__(self) -> str:
        if self.cause is not None:
            return (
                f"[{self.code}] "
                f"{self.message}: "
                f"{self.cause}"
            )

        return (
            f"[{self.code}] "
            f"{self.message}"
        )


class ConfigurationError(FraudSentinelException):
    """Configuration is missing, invalid, or inconsistent."""

    def __init__(
        self,
        message: str,
        *,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(
            message,
            code="CONFIGURATION_ERROR",
            cause=cause,
        )


class DataContractError(FraudSentinelException):
    """Input/output schema violates an expected contract."""

    def __init__(
        self,
        message: str,
        *,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(
            message,
            code="DATA_CONTRACT_ERROR",
            cause=cause,
        )


class DataQualityError(FraudSentinelException):
    """Data violates a quality constraint."""

    def __init__(
        self,
        message: str,
        *,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(
            message,
            code="DATA_QUALITY_ERROR",
            cause=cause,
        )


class ComputationError(FraudSentinelException):
    """A detector or mathematical transformation failed."""

    def __init__(
        self,
        message: str,
        *,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(
            message,
            code="COMPUTATION_ERROR",
            cause=cause,
        )


class ArtifactError(FraudSentinelException):
    """An artifact could not be created, read, or validated."""

    def __init__(
        self,
        message: str,
        *,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(
            message,
            code="ARTIFACT_ERROR",
            cause=cause,
        )


class PipelineError(FraudSentinelException):
    """A pipeline stage failed."""

    def __init__(
        self,
        message: str,
        *,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(
            message,
            code="PIPELINE_ERROR",
            cause=cause,
        )


__all__ = [
    "FraudSentinelException",
    "ConfigurationError",
    "DataContractError",
    "DataQualityError",
    "ComputationError",
    "ArtifactError",
    "PipelineError",
]
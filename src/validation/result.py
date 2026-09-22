from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class ValidationResult:

    validator: str

    check: str

    status: str

    message: str

    details: dict[str, Any]

    def to_dict(self) -> dict:

        return asdict(self)
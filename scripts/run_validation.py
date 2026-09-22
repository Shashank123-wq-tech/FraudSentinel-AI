from __future__ import annotations

from src.validation import (
    SystemValidator,
    ValidationConfig,
)


def main() -> None:

    validator = SystemValidator(
        config=ValidationConfig()
    )

    result = validator.validate()

    if result["status"] != "PASS":

        raise RuntimeError(
            "FraudSentinel AI system "
            "validation FAILED."
        )


if __name__ == "__main__":

    main()
import json

from src.mlops.monitoring.system_monitor import (
    run_system_monitor,
)


def main():

    print()
    print("=" * 60)
    print(" FraudSentinel AI — MLOps Monitoring")
    print("=" * 60)
    print()

    result = run_system_monitor()

    print(
        f"Overall status : "
        f"{result['overall_status']}"
    )

    print()

    print(
        "Artifact failures:"
    )

    failures = result["artifact_failures"]

    if failures:
        for failure in failures:
            print(f"  FAIL: {failure}")
    else:
        print("  None")

    print()

    data = result["data_monitoring"]

    print("DATA MONITORING")
    print(
        f"  Rows             : "
        f"{data.get('rows', 0):,}"
    )
    print(
        f"  Missing values   : "
        f"{data.get('missing_values', 0):,}"
    )
    print(
        f"  Invalid amounts  : "
        f"{data.get('invalid_amounts', 0):,}"
    )
    print(
        f"  Status           : "
        f"{data.get('status')}"
    )

    print()

    model = result["model_monitoring"]

    print("MODEL MONITORING")
    print(
        f"  Transactions     : "
        f"{model.get('transactions', 0):,}"
    )
    print(
        f"  Alerts           : "
        f"{model.get('alerts', 0):,}"
    )
    print(
        f"  Alert rate       : "
        f"{model.get('alert_rate', 0):.4%}"
    )
    print(
        f"  Fraud probability: "
        f"{model.get('mean_fraud_probability', 0):.6f}"
    )
    print(
        f"  Unified risk     : "
        f"{model.get('mean_unified_risk', 0):.6f}"
    )
    print(
        f"  Status           : "
        f"{model.get('status')}"
    )

    print()
    print("=" * 60)
    print(
        f" MLOPS STATUS: "
        f"{result['overall_status']}"
    )
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
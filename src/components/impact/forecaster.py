import numpy as np
import pandas as pd


def forecast_loss(
    events: pd.DataFrame,
    config,
):

    rows = []

    for _, row in events.iterrows():

        event_id = row["event_id"]

        merchant_id = row["merchant_id"]

        current_loss = float(
            max(
                row.get(
                    "current_expected_loss",
                    0.0,
                ),
                0.0,
            )
        )

        velocity = float(
            max(
                row.get(
                    "loss_velocity_per_hour",
                    0.0,
                ),
                0.0,
            )
        )

        growth_rate = float(
            max(
                row.get(
                    "growth_rate_per_hour",
                    0.0,
                ),
                0.0,
            )
        )

        current_fraud_count = float(
            max(
                row.get(
                    "expected_fraud_count",
                    0.0,
                ),
                0.0,
            )
        )

        fraud_velocity = float(
            max(
                row.get(
                    "fraud_txn_velocity_per_hour",
                    0.0,
                ),
                0.0,
            )
        )

        txn_velocity = float(
            max(
                row.get(
                    "transaction_velocity_per_hour",
                    0.0,
                ),
                0.0,
            )
        )

        amount_velocity = float(
            max(
                row.get(
                    "transaction_amount_velocity_per_hour",
                    0.0,
                ),
                0.0,
            )
        )

        avg_fraud_amount = float(
            max(
                row.get(
                    "average_expected_fraud_amount",
                    0.0,
                ),
                0.0,
            )
        )

        confidence = float(
            np.clip(
                row.get(
                    "forecast_confidence",
                    0.0,
                ),
                0,
                1,
            )
        )

        spike_state = str(
            row.get(
                "max_state",
                "UNKNOWN",
            )
        )

        spike_score = float(
            max(
                row.get(
                    "max_fas",
                    0.0,
                ),
                0.0,
            )
        )

        for horizon in config.horizons_minutes:

            hours = (
                horizon / 60.0
            )

            # -------------------------------------------------
            # CONTINUATION
            # -------------------------------------------------

            continuation = (
                velocity * hours
            )

            growth_multiplier = (
                1.0
                + growth_rate * hours
            )

            growth_multiplier = min(
                growth_multiplier,
                config.max_growth_multiplier,
            )

            projected_incremental_loss = (
                continuation
                * growth_multiplier
            )

            projected_loss = (
                current_loss
                + projected_incremental_loss
            )

            # -------------------------------------------------
            # FRAUD COUNT
            # -------------------------------------------------

            projected_fraud_count = (
                current_fraud_count
                + fraud_velocity * hours
            )

            # -------------------------------------------------
            # TRANSACTION COUNT
            # -------------------------------------------------

            projected_transactions = (
                txn_velocity * hours
            )

            # -------------------------------------------------
            # TRANSACTION AMOUNT
            # -------------------------------------------------

            projected_transaction_amount = (
                amount_velocity * hours
            )

            # -------------------------------------------------
            # UNCERTAINTY
            # -------------------------------------------------

            uncertainty = (
                1.0
                + (
                    1.0
                    - confidence
                ) * 0.75
            )

            lower_loss = (
                projected_loss
                / uncertainty
            )

            upper_loss = (
                projected_loss
                * uncertainty
            )

            # -------------------------------------------------
            # PROJECTED LOSS VELOCITY
            # -------------------------------------------------

            projected_loss_velocity = (
                projected_loss
                / max(
                    hours,
                    1e-6,
                )
            )

            rows.append(
                {
                    "event_id":
                        event_id,

                    "merchant_id":
                        merchant_id,

                    "forecast_horizon_minutes":
                        horizon,

                    "current_expected_loss":
                        current_loss,

                    "current_expected_fraud_count":
                        current_fraud_count,

                    "loss_velocity_per_hour":
                        velocity,

                    "fraud_transaction_velocity_per_hour":
                        fraud_velocity,

                    "projected_loss":
                        projected_loss,

                    "projected_loss_lower":
                        lower_loss,

                    "projected_loss_upper":
                        upper_loss,

                    "projected_fraud_transactions":
                        projected_fraud_count,

                    "projected_transactions":
                        projected_transactions,

                    "projected_transaction_amount":
                        projected_transaction_amount,

                    "projected_loss_velocity_per_hour":
                        projected_loss_velocity,

                    "average_expected_fraud_amount":
                        avg_fraud_amount,

                    "growth_rate_per_hour":
                        growth_rate,

                    "growth_multiplier":
                        growth_multiplier,

                    "forecast_confidence":
                        confidence,

                    "spike_score":
                        spike_score,

                    "spike_state":
                        spike_state,

                    "unique_cards":
                        row.get(
                            "unique_cards",
                            0,
                        ),

                    "coordination_score":
                        row.get(
                            "coordination_score",
                            0,
                        ),

                    "max_tas":
                        row.get(
                            "max_tas",
                            0,
                        ),

                    "max_fas":
                        row.get(
                            "max_fas",
                            0,
                        ),

                    "duration_minutes":
                        row.get(
                            "duration_minutes",
                            0,
                        ),

                    "windows":
                        row.get(
                            "windows",
                            0,
                        ),

                    "model_version":
                        config.model_version,
                }
            )

    return pd.DataFrame(rows)

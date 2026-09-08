import pandas as pd


def build_scenarios(
    forecast_df: pd.DataFrame,
    config,
):

    rows = []

    for _, row in forecast_df.iterrows():

        projected_loss = float(
            max(
                row["projected_loss"],
                0.0,
            )
        )

        # =====================================================
        # NO ACTION
        # =====================================================

        no_action_loss = (
            projected_loss
        )

        # =====================================================
        # MODERATE INTERVENTION
        # =====================================================

        moderate_loss = (
            projected_loss
            * config.moderate_loss_retention
        )

        moderate_prevented = (
            projected_loss
            - moderate_loss
        )

        # =====================================================
        # AGGRESSIVE INTERVENTION
        # =====================================================

        aggressive_loss = (
            projected_loss
            * config.aggressive_loss_retention
        )

        aggressive_prevented = (
            projected_loss
            - aggressive_loss
        )

        rows.extend(
            [
                {
                    "event_id":
                        row["event_id"],

                    "merchant_id":
                        row["merchant_id"],

                    "forecast_horizon_minutes":
                        row[
                            "forecast_horizon_minutes"
                        ],

                    "scenario":
                        "NO_ACTION",

                    "projected_loss":
                        no_action_loss,

                    "loss_prevented":
                        0.0,

                    "loss_retained":
                        no_action_loss,

                    "response_intensity":
                        0.0,

                    "forecast_confidence":
                        row[
                            "forecast_confidence"
                        ],

                    "model_version":
                        config.model_version,
                },

                {
                    "event_id":
                        row["event_id"],

                    "merchant_id":
                        row["merchant_id"],

                    "forecast_horizon_minutes":
                        row[
                            "forecast_horizon_minutes"
                        ],

                    "scenario":
                        "MODERATE_INTERVENTION",

                    "projected_loss":
                        moderate_loss,

                    "loss_prevented":
                        moderate_prevented,

                    "loss_retained":
                        moderate_loss,

                    "response_intensity":
                        0.40,

                    "forecast_confidence":
                        row[
                            "forecast_confidence"
                        ],

                    "model_version":
                        config.model_version,
                },

                {
                    "event_id":
                        row["event_id"],

                    "merchant_id":
                        row["merchant_id"],

                    "forecast_horizon_minutes":
                        row[
                            "forecast_horizon_minutes"
                        ],

                    "scenario":
                        "AGGRESSIVE_INTERVENTION",

                    "projected_loss":
                        aggressive_loss,

                    "loss_prevented":
                        aggressive_prevented,

                    "loss_retained":
                        aggressive_loss,

                    "response_intensity":
                        0.80,

                    "forecast_confidence":
                        row[
                            "forecast_confidence"
                        ],

                    "model_version":
                        config.model_version,
                },
            ]
        )

    return pd.DataFrame(rows)

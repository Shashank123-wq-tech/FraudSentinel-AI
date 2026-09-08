"""
FraudSentinel AI
Response Recommendation Runner

Run:
    python -m scripts.run_response
"""

import json
import uuid

from src.components.response.config import (
    RECOMMENDATION_PATH,
    ACTION_PATH,
    SUMMARY_PATH,
    RESPONSE_MODEL_VERSION,
)

from src.components.response.loader import (
    load_fusion,
    load_impact_forecast,
    load_impact_scenarios,
)

from src.components.response.recommender import (
    build_impact_features,
    merge_impact_into_fusion,
    calculate_recommendation_confidence,
    finalize_recommendations,
)

from src.components.response.scoring import (
    calculate_scores,
)

from src.components.response.policy import (
    apply_policy,
)


def main():

    print()
    print("=" * 70)
    print("FRAUDSENTINEL AI — RESPONSE RECOMMENDATION")
    print("=" * 70)

    # ========================================================
    # 1. LOAD INPUT ARTIFACTS
    # ========================================================

    print()
    print("[1/8] Loading Fusion artifact...")

    fusion = load_fusion()

    print(
        f"       Fusion rows       : {len(fusion):,}"
    )

    print()
    print("[2/8] Loading Impact Forecast artifact...")

    forecast = load_impact_forecast()

    print(
        f"       Forecast rows     : {len(forecast):,}"
    )

    print()
    print("[3/8] Loading Impact Scenario artifact...")

    scenarios = load_impact_scenarios()

    print(
        f"       Scenario rows     : {len(scenarios):,}"
    )

    # ========================================================
    # 2. BUILD EVENT-LEVEL IMPACT FEATURES
    # ========================================================

    print()
    print("[4/8] Building event-level impact features...")

    impact = build_impact_features(
        forecast=forecast,
        scenarios=scenarios,
    )

    print(
        f"       Impact events     : {len(impact):,}"
    )

    # ========================================================
    # 3. MERGE FUSION + IMPACT
    # ========================================================

    print()
    print("[5/8] Merging Fusion + Impact intelligence...")

    recommendations = merge_impact_into_fusion(
        fusion=fusion,
        impact=impact,
    )

    if recommendations.empty:

        print()
        print("No active fraud events were found.")

        empty_summary = {
            "project": "FraudSentinel AI",
            "component": "Response Recommendation",
            "model_version": RESPONSE_MODEL_VERSION,
            "events_processed": 0,
            "recommendations": 0,
            "actions": 0,
        }

        with open(
            SUMMARY_PATH,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                empty_summary,
                file,
                indent=2,
            )

        return

    print(
        f"       Events detected    : "
        f"{len(recommendations):,}"
    )

    # ========================================================
    # 4. CALCULATE RISK / RESPONSE SCORES
    # ========================================================

    print()
    print("[6/8] Calculating response scores...")

    recommendations = calculate_scores(
        recommendations
    )

    # ========================================================
    # 5. CALCULATE RECOMMENDATION CONFIDENCE
    # ========================================================

    print(
        "       Calculating recommendation confidence..."
    )

    recommendations = calculate_recommendation_confidence(
        recommendations
    )

    # ========================================================
    # 6. APPLY RESPONSE POLICY
    # ========================================================

    print(
        "       Applying response policy..."
    )

    recommendations = apply_policy(
        recommendations
    )

    # ========================================================
    # 7. FINALIZE ECONOMIC IMPACT
    # ========================================================

    print(
        "       Calculating economic impact..."
    )

    recommendations = finalize_recommendations(
        recommendations
    )

    # ========================================================
    # 8. IDENTIFIERS
    # ========================================================

    recommendations.insert(
        0,
        "recommendation_id",
        [
            f"REC-{uuid.uuid4().hex[:12].upper()}"
            for _ in range(len(recommendations))
        ],
    )

    recommendations["policy_version"] = (
        RESPONSE_MODEL_VERSION
    )

    # ========================================================
    # SAVE RECOMMENDATIONS
    # ========================================================

    recommendations.to_csv(
        RECOMMENDATION_PATH,
        index=False,
    )

    # ========================================================
    # BUILD ACTION TABLE
    # ========================================================

    action_columns = [
        "recommendation_id",
        "event_id",
        "merchant_id",
        "response_priority",
        "response_action",
        "recommended_control",
        "estimated_loss_prevented",
        "recommendation_confidence",
        "timestamp",
    ]

    available_action_columns = [
        column
        for column in action_columns
        if column in recommendations.columns
    ]

    actions = recommendations[
        available_action_columns
    ].copy()

    actions.insert(
        0,
        "action_id",
        [
            f"ACT-{uuid.uuid4().hex[:12].upper()}"
            for _ in range(len(actions))
        ],
    )

    actions.to_csv(
        ACTION_PATH,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    action_counts = (
        recommendations[
            "response_action"
        ]
        .value_counts()
        .to_dict()
    )

    priority_counts = (
        recommendations[
            "response_priority"
        ]
        .value_counts()
        .to_dict()
    )

    total_loss_prevented = float(
        recommendations[
            "estimated_loss_prevented"
        ].sum()
    )

    total_no_action_loss = float(
        recommendations[
            "no_action_loss_60m"
        ].sum()
    )

    total_expected_action_loss = float(
        recommendations[
            "expected_loss_if_action"
        ].sum()
    )

    mean_confidence = float(
        recommendations[
            "recommendation_confidence"
        ].mean()
    )

    summary = {
        "project": "FraudSentinel AI",
        "component": "Response Recommendation",
        "model_version": RESPONSE_MODEL_VERSION,

        "events_processed": int(
            len(recommendations)
        ),

        "recommendations_generated": int(
            len(recommendations)
        ),

        "actions_generated": int(
            len(actions)
        ),

        "total_no_action_loss_60m": (
            total_no_action_loss
        ),

        "total_expected_action_loss_60m": (
            total_expected_action_loss
        ),

        "estimated_loss_prevented": (
            total_loss_prevented
        ),

        "mean_recommendation_confidence": (
            mean_confidence
        ),

        "action_distribution": {
            str(key): int(value)
            for key, value in action_counts.items()
        },

        "priority_distribution": {
            str(key): int(value)
            for key, value in priority_counts.items()
        },

        "artifacts": {
            "recommendations": str(
                RECOMMENDATION_PATH
            ),

            "actions": str(
                ACTION_PATH
            ),

            "summary": str(
                SUMMARY_PATH
            ),
        },
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    # ========================================================
    # TERMINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("RESPONSE RECOMMENDATION COMPLETE")
    print("=" * 70)

    print(
        f"Events processed        : "
        f"{len(recommendations):,}"
    )

    print(
        f"Recommendations         : "
        f"{len(recommendations):,}"
    )

    print(
        f"Actions generated       : "
        f"{len(actions):,}"
    )

    print()
    print("-" * 70)
    print("RESPONSE ACTIONS")
    print("-" * 70)

    for action, count in action_counts.items():

        print(
            f"{str(action):28s}: "
            f"{int(count):,}"
        )

    print()
    print("-" * 70)
    print("RESPONSE PRIORITIES")
    print("-" * 70)

    for priority, count in priority_counts.items():

        print(
            f"{str(priority):28s}: "
            f"{int(count):,}"
        )

    print()
    print("-" * 70)
    print("ECONOMIC IMPACT")
    print("-" * 70)

    print(
        f"No-action loss (60m)    : "
        f"{total_no_action_loss:,.2f}"
    )

    print(
        f"Expected action loss    : "
        f"{total_expected_action_loss:,.2f}"
    )

    print(
        f"Estimated loss prevented: "
        f"{total_loss_prevented:,.2f}"
    )

    print(
        f"Recommendation confidence: "
        f"{mean_confidence:.4f}"
    )

    print()
    print("-" * 70)
    print("ARTIFACTS")
    print("-" * 70)

    print(
        f"Recommendations : "
        f"{RECOMMENDATION_PATH}"
    )

    print(
        f"Actions         : "
        f"{ACTION_PATH}"
    )

    print(
        f"Summary         : "
        f"{SUMMARY_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()

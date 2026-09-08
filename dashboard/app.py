import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# INTERNAL IMPORTS
# ============================================================

from src.components.dashboard.loader import (
    artifact_status,
    build_fusion_summary,
    build_merchant_ranking,
    build_risk_timeseries,
    build_spike_distribution,
    load_phase2_events,
    load_impact_forecast,
    load_impact_scenarios,
    load_response_recommendations,
    load_response_actions,
    load_top_transactions,
    load_xai_unified,
)

from src.components.dashboard.metrics import (
    calculate_impact_metrics,
    calculate_response_metrics,
)

from dashboard.components.charts import (
    risk_timeline,
    spike_timeline,
    merchant_ranking,
    response_distribution,
    priority_distribution,
    scenario_comparison,
    probability_distribution,
    risk_amount_scatter,
)

from dashboard.components.cards import (
    render_overview_metrics,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FraudSentinel AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1800px;
    }

    .dashboard-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
    }

    .dashboard-subtitle {
        color: #8b949e;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .status-online {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        font-weight: 700;
        background: rgba(46, 160, 67, 0.15);
        border: 1px solid rgba(46, 160, 67, 0.35);
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 750;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [5, 1]
)

with header_left:

    st.markdown(
        '<div class="dashboard-title">'
        '🛡️ FraudSentinel AI'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        'Autonomous • Explainable • Adaptive Fraud-Spike Intelligence'
        '</div>',
        unsafe_allow_html=True,
    )

with header_right:

    st.markdown(
        '<div class="status-online">'
        '● SYSTEM ACTIVE'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Control Center")

    if st.button(
        "🔄 Refresh dashboard",
        width="stretch",
    ):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    st.subheader("Artifact Status")

    statuses = artifact_status()

    for name, available in statuses.items():

        if available:
            st.success(
                f"✓ {name}"
            )
        else:
            st.error(
                f"✗ {name}"
            )

    st.divider()

    st.caption(
        "Dashboard reads existing FraudSentinel AI "
        "artifacts. It does not retrain models."
    )


# ============================================================
# LOAD DATA
# ============================================================

with st.spinner(
    "Loading FraudSentinel intelligence..."
):

    fusion_summary = build_fusion_summary()

    risk_ts = build_risk_timeseries()

    merchants = build_merchant_ranking()

    top_transactions = load_top_transactions(
        limit=100,
    )

    events = load_phase2_events()

    impact_forecast = load_impact_forecast()

    impact_scenarios = load_impact_scenarios()

    responses = load_response_recommendations()

    response_actions = load_response_actions()

    xai = load_xai_unified()


impact_metrics = calculate_impact_metrics(
    impact_forecast
)

response_metrics = calculate_response_metrics(
    responses
)


# ============================================================
# ACTIVE SPIKES
# ============================================================

active_spikes = 0

if not events.empty:

    if "spike_state" in events.columns:

        active_spikes = events[
            events["spike_state"]
            .astype(str)
            .isin(
                [
                    "EARLY_WARNING",
                    "VERIFIED_FRAUD_SPIKE",
                    "CRITICAL_ACTIVE_SPIKE",
                ]
            )
        ].shape[0]

    else:

        active_spikes = len(events)


# ============================================================
# KPI SECTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Executive Risk Overview'
    '</div>',
    unsafe_allow_html=True,
)

render_overview_metrics(
    fusion_summary,
    impact_metrics,
    response_metrics,
    active_spikes,
)


# ============================================================
# MAIN TABS
# ============================================================

tabs = st.tabs(
    [
        "🎯 Overview",
        "💳 Transactions",
        "🚨 Fraud Spikes",
        "💰 Financial Impact",
        "⚡ Response Intelligence",
        "🔍 XAI Investigation",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tabs[0]:

    st.subheader(
        "Real-time risk intelligence"
    )

    col1, col2 = st.columns(
        [2.2, 1]
    )

    with col1:

        st.plotly_chart(
            risk_timeline(risk_ts),
            width="stretch",
            key="overview_risk_timeline",
        )

    with col2:

        if not top_transactions.empty:

            st.plotly_chart(
                probability_distribution(
                    top_transactions
                ),
                width="stretch",
                key="overview_probability_distribution",
            )

    st.subheader(
        "Merchant risk intelligence"
    )

    st.plotly_chart(
        merchant_ranking(merchants),
        width="stretch",
        key="overview_merchant_ranking",
    )


# ============================================================
# TRANSACTIONS
# ============================================================

with tabs[1]:

    st.subheader(
        "Transaction Intelligence"
    )

    if top_transactions.empty:

        st.warning(
            "No transaction-level Fusion data available."
        )

    else:

        st.plotly_chart(
            risk_amount_scatter(
                top_transactions
            ),
            width="stretch",
            key="transactions_risk_amount_scatter",
        )

        st.subheader(
            "Highest-risk transactions"
        )

        display_columns = [
            "transaction_id",
            "timestamp",
            "merchant_id",
            "amount",
            "fraud_probability",
            "phase1_score",
            "phase2_score",
            "unified_risk_score",
            "risk_band",
            "spike_state",
            "response_action",
            "response_priority",
            "alert_flag",
        ]

        display_columns = [
            c
            for c in display_columns
            if c in top_transactions.columns
        ]

        st.dataframe(
            top_transactions[
                display_columns
            ],
            width="stretch",
            hide_index=True,
        )


# ============================================================
# SPIKES
# ============================================================

with tabs[2]:

    st.subheader(
        "Fraud Spike Intelligence"
    )

    if events.empty:

        st.warning(
            "No Phase 2 event artifact found."
        )

    else:

        state_col = (
            "spike_state"
            if "spike_state" in events.columns
            else None
        )

        if state_col:

            state_counts = (
                events[state_col]
                .astype(str)
                .value_counts()
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Total events",
                    f"{len(events):,}",
                )

            with c2:
                st.metric(
                    "Early warnings",
                    f"{state_counts.get('EARLY_WARNING', 0):,}",
                )

            with c3:
                st.metric(
                    "Verified spikes",
                    f"{state_counts.get('VERIFIED_FRAUD_SPIKE', 0):,}",
                )

            with c4:
                st.metric(
                    "Critical spikes",
                    f"{state_counts.get('CRITICAL_ACTIVE_SPIKE', 0):,}",
                )

        st.plotly_chart(
            spike_timeline(
                events
            ),
            width="stretch",
            key="spikes_timeline",
        )

        st.subheader(
            "Spike event registry"
        )

        spike_columns = [
            "event_id",
            "merchant_id",
            "start_time",
            "end_time",
            "duration_minutes",
            "windows",
            "transaction_count",
            "total_amount",
            "expected_fraud_count",
            "expected_fraud_amount",
            "unique_cards",
            "max_coordination_score",
            "max_tas",
            "max_fas",
        ]

        spike_columns = [
            c
            for c in spike_columns
            if c in events.columns
        ]

        st.dataframe(
            events[
                spike_columns
            ].sort_values(
                "max_fas",
                ascending=False,
            )
            if "max_fas" in events.columns
            else events[spike_columns],
            width="stretch",
            hide_index=True,
        )


# ============================================================
# FINANCIAL IMPACT
# ============================================================

with tabs[3]:

    st.subheader(
        "Loss / Impact Intelligence"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Current expected loss",
            f"${impact_metrics['current_loss']:,.2f}",
        )

    with c2:
        st.metric(
            "15-minute forecast",
            f"${impact_metrics['forecast_15']:,.2f}",
        )

    with c3:
        st.metric(
            "30-minute forecast",
            f"${impact_metrics['forecast_30']:,.2f}",
        )

    with c4:
        st.metric(
            "60-minute forecast",
            f"${impact_metrics['forecast_60']:,.2f}",
        )

    st.divider()

    no_action = 0.0
    moderate = 0.0
    aggressive = 0.0

    if not impact_scenarios.empty:

        scenario_col = None

        for candidate in [
            "scenario",
            "action",
            "scenario_name",
        ]:
            if candidate in impact_scenarios.columns:
                scenario_col = candidate
                break

        if scenario_col:

            loss_col = None

            for candidate in [
                "projected_loss",
                "loss",
                "expected_loss",
            ]:
                if candidate in impact_scenarios.columns:
                    loss_col = candidate
                    break

            if loss_col:

                values = (
                    impact_scenarios
                    .groupby(
                        impact_scenarios[
                            scenario_col
                        ].astype(str)
                    )[loss_col]
                    .sum()
                )

                no_action = values.get(
                    "NO_ACTION",
                    0.0,
                )

                moderate = values.get(
                    "MODERATE_INTERVENTION",
                    0.0,
                )

                aggressive = values.get(
                    "AGGRESSIVE_INTERVENTION",
                    0.0,
                )

    st.plotly_chart(
        scenario_comparison(
            no_action,
            moderate,
            aggressive,
        ),
        width="stretch",
        key="impact_scenario_comparison",
    )

    if not impact_forecast.empty:

        st.subheader(
            "Forecast detail"
        )

        st.dataframe(
            impact_forecast,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# RESPONSE
# ============================================================

with tabs[4]:

    st.subheader(
        "Response Intelligence"
    )

    if responses.empty:

        st.warning(
            "Response recommendations are unavailable."
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            st.plotly_chart(
                response_distribution(
                    responses
                ),
                width="stretch",
                key="response_action_distribution",
            )

        with col2:

            st.plotly_chart(
                priority_distribution(
                    responses
                ),
                width="stretch",
            )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Recommendations",
                f"{len(responses):,}",
            )

        with c2:
            st.metric(
                "Recommendation confidence",
                f"{response_metrics['confidence']:.1%}",
            )

        with c3:
            st.metric(
                "Estimated loss prevented",
                f"${response_metrics['prevented']:,.2f}",
            )

        st.subheader(
            "Recommended actions"
        )

        response_columns = [
            "event_id",
            "merchant_id",
            "action",
            "priority",
            "recommended_control",
            "recommendation_confidence",
            "expected_loss_if_action",
            "estimated_loss_prevented",
            "action_effectiveness",
        ]

        response_columns = [
            c
            for c in response_columns
            if c in responses.columns
        ]

        st.dataframe(
            responses[
                response_columns
            ],
            width="stretch",
            hide_index=True,
        )


# ============================================================
# XAI
# ============================================================

with tabs[5]:

    st.subheader(
        "Explainable AI Investigation"
    )

    if xai.empty:

        st.warning(
            "Unified XAI artifact is unavailable."
        )

    else:

        st.info(
            "Select an event or transaction to investigate "
            "the evidence generated by the XAI layer."
        )

        event_column = (
            "event_id"
            if "event_id" in xai.columns
            else None
        )

        if event_column:

            event_values = (
                xai[event_column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_event = st.selectbox(
                "Select event",
                event_values,
            )

            selected = xai[
                xai[event_column]
                .astype(str)
                .eq(selected_event)
            ]

        else:

            selected = xai.head(1)

        if not selected.empty:

            row = selected.iloc[0]

            st.subheader(
                f"Investigation: {selected_event}"
                if event_column
                else "Investigation"
            )

            left, right = st.columns(2)

            with left:

                st.markdown(
                    "### Risk Evidence"
                )

                evidence_columns = [
                    "unified_risk_score",
                    "fraud_probability",
                    "phase1_score",
                    "phase2_score",
                    "tas",
                    "fas",
                    "spike_state",
                    "risk_band",
                ]

                for column in evidence_columns:

                    if column in selected.columns:

                        value = row[column]

                        st.write(
                            f"**{column}:** {value}"
                        )

            with right:

                st.markdown(
                    "### Decision Evidence"
                )

                decision_columns = [
                    "phase1_primary_signal",
                    "phase2_primary_signal",
                    "temporal_spike_signal",
                    "verified_spike_signal",
                    "critical_spike_signal",
                    "active_event_signal",
                    "response_action",
                    "response_priority",
                ]

                for column in decision_columns:

                    if column in selected.columns:

                        value = row[column]

                        st.write(
                            f"**{column}:** {value}"
                        )

            st.markdown(
                "### Raw explanation record"
            )

            st.dataframe(
                selected,
                width="stretch",
                hide_index=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "FraudSentinel AI • Dashboard v1.0.0 • "
    "Defense-only fraud intelligence platform"
)
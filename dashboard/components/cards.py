import streamlit as st


def metric_card(
    label: str,
    value,
    delta=None,
):

    st.metric(
        label=label,
        value=value,
        delta=delta,
    )


def render_overview_metrics(
    fusion_summary: dict,
    impact_metrics: dict,
    response_metrics: dict,
    active_spikes: int,
):

    cols = st.columns(6)

    with cols[0]:
        st.metric(
            "Transactions",
            f"{fusion_summary.get('transactions', 0):,}",
        )

    with cols[1]:
        st.metric(
            "High-risk",
            f"{fusion_summary.get('high_risk', 0):,}",
        )

    with cols[2]:
        st.metric(
            "Active spikes",
            f"{active_spikes:,}",
        )

    with cols[3]:
        st.metric(
            "Expected fraud loss",
            f"${fusion_summary.get('expected_fraud_amount', 0):,.2f}",
        )

    with cols[4]:
        st.metric(
            "60-min forecast",
            f"${impact_metrics.get('forecast_60', 0):,.2f}",
        )

    with cols[5]:
        st.metric(
            "Loss prevented",
            f"${response_metrics.get('prevented', 0):,.2f}",
        )
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def risk_timeline(df: pd.DataFrame):

    fig = go.Figure()

    if df.empty:
        return fig

    if "avg_unified_risk" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["avg_unified_risk"],
                mode="lines",
                name="Unified Risk",
            )
        )

    if "avg_phase1_score" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["avg_phase1_score"],
                mode="lines",
                name="Phase 1",
            )
        )

    if "avg_phase2_score" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["avg_phase2_score"],
                mode="lines",
                name="Phase 2",
            )
        )

    fig.update_layout(
        title="Unified risk intelligence timeline",
        xaxis_title="Time",
        yaxis_title="Risk score",
        hovermode="x unified",
        height=480,
    )

    return fig


def spike_timeline(df: pd.DataFrame):

    if df.empty:
        return go.Figure()

    fig = go.Figure()

    if "max_fas" in df.columns:

        ordered = df.sort_values(
            "start_time"
            if "start_time" in df.columns
            else df.index
        )

        x = (
            ordered["start_time"]
            if "start_time" in ordered.columns
            else ordered.index
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=ordered["max_fas"],
                mode="lines+markers",
                name="Spike intensity",
            )
        )

    fig.update_layout(
        title="Fraud spike intensity over time",
        xaxis_title="Event start",
        yaxis_title="FAS / spike score",
        height=420,
    )

    return fig


def merchant_ranking(df: pd.DataFrame):

    if df.empty:
        return go.Figure()

    ordered = df.sort_values(
        "avg_unified_risk"
    )

    fig = px.bar(
        ordered,
        x="avg_unified_risk",
        y="merchant_id",
        orientation="h",
        hover_data=[
            "transactions",
            "expected_fraud_amount",
            "events",
            "max_unified_risk",
        ],
        title="Highest-risk merchants",
    )

    fig.update_layout(
        height=520,
        yaxis_title="Merchant",
        xaxis_title="Average unified risk",
    )

    return fig


def response_distribution(df: pd.DataFrame):

    if df.empty or "action" not in df.columns:
        return go.Figure()

    counts = (
        df["action"]
        .astype(str)
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "action",
        "count",
    ]

    fig = px.bar(
        counts,
        x="action",
        y="count",
        title="Recommended response actions",
    )

    fig.update_layout(
        height=400,
    )

    return fig


def priority_distribution(df: pd.DataFrame):

    if df.empty or "priority" not in df.columns:
        return go.Figure()

    counts = (
        df["priority"]
        .astype(str)
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "priority",
        "count",
    ]

    fig = px.pie(
        counts,
        names="priority",
        values="count",
        hole=0.55,
        title="Response priority distribution",
    )

    fig.update_layout(
        height=420,
    )

    return fig


def scenario_comparison(
    no_action: float,
    moderate: float,
    aggressive: float,
):

    df = pd.DataFrame(
        {
            "Scenario": [
                "No action",
                "Moderate intervention",
                "Aggressive intervention",
            ],
            "Projected loss": [
                no_action,
                moderate,
                aggressive,
            ],
        }
    )

    fig = px.bar(
        df,
        x="Scenario",
        y="Projected loss",
        title="60-minute projected loss by response scenario",
    )

    fig.update_layout(
        height=420,
        yaxis_title="Projected loss",
    )

    return fig


def probability_distribution(df: pd.DataFrame):

    if df.empty or "fraud_probability" not in df.columns:
        return go.Figure()

    fig = px.histogram(
        df,
        x="fraud_probability",
        nbins=40,
        title="Transaction fraud-probability distribution",
    )

    fig.update_layout(
        height=420,
        xaxis_title="Fraud probability",
        yaxis_title="Transactions",
    )

    return fig


def risk_amount_scatter(df: pd.DataFrame):

    if df.empty:
        return go.Figure()

    required = [
        "amount",
        "unified_risk_score",
    ]

    if not all(
        c in df.columns
        for c in required
    ):
        return go.Figure()

    plot_df = df.copy()

    plot_df["amount"] = pd.to_numeric(
        plot_df["amount"],
        errors="coerce",
    )

    plot_df["unified_risk_score"] = pd.to_numeric(
        plot_df["unified_risk_score"],
        errors="coerce",
    )

    plot_df = plot_df.dropna(
        subset=required
    )

    fig = px.scatter(
        plot_df,
        x="amount",
        y="unified_risk_score",
        size="fraud_probability"
        if "fraud_probability" in plot_df.columns
        else None,
        hover_data=[
            c
            for c in [
                "transaction_id",
                "merchant_id",
                "risk_band",
                "spike_state",
            ]
            if c in plot_df.columns
        ],
        title="Transaction amount vs unified risk",
    )

    fig.update_layout(
        height=500,
        xaxis_title="Transaction amount",
        yaxis_title="Unified risk score",
    )

    return fig
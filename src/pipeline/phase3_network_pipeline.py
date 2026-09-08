# src/pipeline/phase3_network_pipeline.py

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pandas as pd

from src.components.phase3.entity_resolution import (
    EntityResolver,
)

from src.components.phase3.event_subgraph import (
    EventSubgraphExtractor,
)

from src.components.phase3.graph_builder import (
    FraudNetworkGraphBuilder,
)

from src.components.phase3.link_features import (
    LinkFeatureEngineer,
)

from src.components.phase3.network_anomaly import (
    NetworkAnomalyDetector,
)

from src.components.phase3.coordinated_attack import (
    CoordinatedAttackDetector,
)


class Phase3NetworkIntelligence:
    """
    Complete Phase 3 network intelligence pipeline.

    Inputs:

        Phase 1:
            transaction_risk.csv

        Phase 2:
            merchant_spike_events.parquet

    Output:

        Network/community intelligence.
    """

    def __init__(
        self,
        output_dir: Path,
        random_seed: int = 42,
        model_version: str = "network_intelligence_v1",
        pre_window_minutes: int = 0,
        post_window_minutes: int = 0,
    ):

        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.random_seed = random_seed
        self.model_version = model_version

        self.entity_resolver = (
            EntityResolver()
        )

        self.subgraph_extractor = (
            EventSubgraphExtractor(
                pre_window_minutes=pre_window_minutes,
                post_window_minutes=post_window_minutes,
            )
        )

        self.graph_builder = (
            FraudNetworkGraphBuilder()
        )

        self.link_engineer = (
            LinkFeatureEngineer()
        )

        self.network_anomaly = (
            NetworkAnomalyDetector()
        )

        self.coordinated_detector = (
            CoordinatedAttackDetector()
        )

    # -----------------------------------------------------
    # Loading
    # -----------------------------------------------------

    @staticmethod
    def load_phase1(
        path: Path,
    ) -> pd.DataFrame:

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Phase 1 file not found: {path}"
            )

        df = pd.read_csv(path)

        return df

    @staticmethod
    def load_phase2(
        path: Path,
    ) -> pd.DataFrame:

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Phase 2 file not found: {path}"
            )

        df = pd.read_parquet(path)

        return df

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    @staticmethod
    def validate_phase2(
        df: pd.DataFrame,
    ) -> None:

        required = [
            "event_id",
            "merchant_id",
            "attack_start_time",
            "detection_time",
            "attack_end_time",
            "spike_score",
        ]

        missing = [
            c for c in required
            if c not in df.columns
        ]

        if missing:
            raise ValueError(
                "Phase 2 output missing columns: "
                f"{missing}"
            )

    # -----------------------------------------------------
    # Community Detection
    # -----------------------------------------------------

    @staticmethod
    def detect_communities(
        graph: nx.Graph,
    ) -> dict:

        if graph.number_of_nodes() == 0:
            return {}

        if graph.number_of_edges() == 0:
            return {
                node: 0
                for node in graph.nodes()
            }

        communities = nx.community.greedy_modularity_communities(
            graph,
            weight="transaction_count",
        )

        mapping = {}

        for community_id, community in enumerate(
            communities
        ):

            for node in community:
                mapping[node] = community_id

        return mapping

    @staticmethod
    def community_summary(
        graph: nx.Graph,
        community_mapping: dict,
        transactions: pd.DataFrame,
        event_id: str,
    ) -> pd.DataFrame:

        if not community_mapping:
            return pd.DataFrame()

        node_to_community = pd.Series(
            community_mapping,
            name="community_id",
        )

        rows = []

        for community_id in sorted(
            node_to_community.unique()
        ):

            nodes = [
                node
                for node, cid
                in community_mapping.items()
                if cid == community_id
            ]

            node_set = set(nodes)

            cards = [
                node
                for node in nodes
                if graph.nodes[node].get(
                    "node_type"
                ) == "CARD"
            ]

            merchants = [
                node
                for node in nodes
                if graph.nodes[node].get(
                    "node_type"
                ) == "MERCHANT"
            ]

            edges = [
                (u, v)
                for u, v
                in graph.edges()
                if u in node_set
                and v in node_set
            ]

            community_txn = transactions[
                transactions["card_id"].map(
                    lambda x:
                    f"CARD::{x}" in node_set
                )
                &
                transactions["merchant_id"].map(
                    lambda x:
                    f"MERCHANT::{x}" in node_set
                )
            ]

            total_amount = float(
                community_txn["amount"].sum()
            )

            high_risk_amount = float(
                community_txn.loc[
                    community_txn["predicted_fraud"] == 1,
                    "amount",
                ].sum()
            )

            avg_probability = float(
                community_txn[
                    "fraud_probability"
                ].mean()
            ) if not community_txn.empty else 0.0

            rows.append(
                {
                    "event_id": event_id,
                    "community_id": int(
                        community_id
                    ),
                    "node_count": len(nodes),
                    "card_count": len(cards),
                    "merchant_count": len(merchants),
                    "edge_count": len(edges),
                    "transaction_count": len(
                        community_txn
                    ),
                    "total_amount": total_amount,
                    "high_risk_count": int(
                        community_txn[
                            "predicted_fraud"
                        ].sum()
                    ),
                    "high_risk_amount": high_risk_amount,
                    "avg_fraud_probability": avg_probability,
                }
            )

        return pd.DataFrame(rows)

    # -----------------------------------------------------
    # Baseline
    # -----------------------------------------------------

    @staticmethod
    def build_baseline(
        transactions: pd.DataFrame,
        event: pd.Series,
    ) -> pd.DataFrame:

        merchant_id = str(
            event["merchant_id"]
        )

        start = pd.to_datetime(
            event["attack_start_time"],
            utc=True,
            errors="coerce",
        )

        if pd.isna(start):
            return transactions.iloc[0:0].copy()

        duration = (
            pd.to_datetime(
                event["attack_end_time"],
                utc=True,
                errors="coerce",
            )
            - start
        )

        if pd.isna(duration):
            duration = pd.Timedelta(
                minutes=15
            )

        baseline_end = start

        baseline_start = (
            start - duration
        )

        mask = (
            (transactions["merchant_id"] == merchant_id)
            &
            (transactions["timestamp"] >= baseline_start)
            &
            (transactions["timestamp"] < baseline_end)
        )

        return transactions.loc[
            mask
        ].copy()

    # -----------------------------------------------------
    # Main execution
    # -----------------------------------------------------

    def run(
        self,
        phase1_df: pd.DataFrame,
        phase2_df: pd.DataFrame,
    ) -> dict:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "FRAUDSENTINEL AI - PHASE 3"
        )

        print(
            "NETWORK / LINK INTELLIGENCE"
        )

        print(
            "=" * 70
        )

        # -----------------------------------------------
        # Validate and resolve Phase 1
        # -----------------------------------------------

        print(
            "\n[1/7] Resolving Phase 1 entities..."
        )

        transactions = (
            self.entity_resolver.resolve(
                phase1_df
            )
        )

        print(
            f"Transactions: {len(transactions):,}"
        )

        # -----------------------------------------------
        # Validate Phase 2
        # -----------------------------------------------

        print(
            "\n[2/7] Validating Phase 2 events..."
        )

        self.validate_phase2(
            phase2_df
        )

        events = phase2_df.copy()

        events["merchant_id"] = (
            events["merchant_id"]
            .astype(str)
            .str.strip()
        )

        events["attack_start_time"] = (
            pd.to_datetime(
                events["attack_start_time"],
                utc=True,
                errors="coerce",
            )
        )

        events["attack_end_time"] = (
            pd.to_datetime(
                events["attack_end_time"],
                utc=True,
                errors="coerce",
            )
        )

        print(
            f"Spike events: {len(events):,}"
        )

        # -----------------------------------------------
        # Process events
        # -----------------------------------------------

        print(
            "\n[3/7] Building event-centered networks..."
        )

        network_results = []
        all_nodes = []
        all_edges = []
        all_communities = []

        processed = 0
        skipped = 0

        for _, event in events.iterrows():

            event_id = str(
                event["event_id"]
            )

            event_transactions = (
                self.subgraph_extractor.extract(
                    transactions,
                    event,
                )
            )

            if event_transactions.empty:
                skipped += 1
                continue

            # -------------------------------------------
            # Baseline immediately before event
            # -------------------------------------------

            baseline_transactions = (
                self.build_baseline(
                    transactions,
                    event,
                )
            )

            # -------------------------------------------
            # Graph
            # -------------------------------------------

            graph = self.graph_builder.build(
                event_transactions
            )

            if graph.number_of_nodes() == 0:
                skipped += 1
                continue

            # -------------------------------------------
            # Link features
            # -------------------------------------------

            link_features = (
                self.link_engineer.transform(
                    event_transactions
                )
            )

            # -------------------------------------------
            # Network anomaly
            # -------------------------------------------

            network_metrics = (
                self.network_anomaly.analyze(
                    graph=graph,
                    transactions=event_transactions,
                    baseline_transactions=baseline_transactions,
                )
            )

            # -------------------------------------------
            # Communities
            # -------------------------------------------

            community_mapping = (
                self.detect_communities(
                    graph
                )
            )

            community_df = (
                self.community_summary(
                    graph=graph,
                    community_mapping=community_mapping,
                    transactions=event_transactions,
                    event_id=event_id,
                )
            )

            community_count = (
                community_df["community_id"]
                .nunique()
                if not community_df.empty
                else 0
            )

            # -------------------------------------------
            # Coordinated attack
            # -------------------------------------------

            coordination = (
                self.coordinated_detector.detect(
                    event=event,
                    transactions=event_transactions,
                    link_features=link_features,
                    network_metrics=network_metrics,
                    community_count=community_count,
                )
            )

            # -------------------------------------------
            # Evidence
            # -------------------------------------------

            new_cards = (
                set(
                    event_transactions[
                        "card_id"
                    ]
                )
                -
                set(
                    baseline_transactions[
                        "card_id"
                    ]
                )
            )

            evidence = {
                "transaction_count": int(
                    len(event_transactions)
                ),

                "unique_cards": int(
                    event_transactions[
                        "card_id"
                    ].nunique()
                ),

                "unique_merchants": int(
                    event_transactions[
                        "merchant_id"
                    ].nunique()
                ),

                "new_cards": int(
                    len(new_cards)
                ),

                "network_edges": int(
                    graph.number_of_edges()
                ),

                "network_nodes": int(
                    graph.number_of_nodes()
                ),

                "communities": int(
                    community_count
                ),

                "network_density": float(
                    network_metrics[
                        "network_density"
                    ]
                ),

                "network_growth": float(
                    network_metrics[
                        "network_growth"
                    ]
                ),
            }

            # -------------------------------------------
            # Top suspicious links
            # -------------------------------------------

            if not link_features.empty:

                top_links = (
                    link_features
                    .sort_values(
                        "link_risk_score",
                        ascending=False,
                    )
                    .head(10)
                )

                suspicious_links = (
                    top_links[
                        [
                            "card_id",
                            "merchant_id",
                            "transaction_count",
                            "total_amount",
                            "avg_fraud_probability",
                            "link_risk_score",
                        ]
                    ]
                    .to_dict(
                        orient="records"
                    )
                )

            else:
                suspicious_links = []

            # -------------------------------------------
            # Result
            # -------------------------------------------

            result = {
                "event_id": event_id,

                "merchant_id": str(
                    event["merchant_id"]
                ),

                "attack_start_time": event[
                    "attack_start_time"
                ],

                "detection_time": event[
                    "detection_time"
                ],

                "attack_end_time": event[
                    "attack_end_time"
                ],

                "spike_score": coordination[
                    "spike_score"
                ],

                "spike_ratio": float(
                    event.get(
                        "spike_ratio",
                        0.0,
                    )
                ),

                "spike_acceleration": float(
                    event.get(
                        "spike_acceleration",
                        0.0,
                    )
                ),

                "network_risk_score": coordination[
                    "network_risk_score"
                ],

                "coordination_score": coordination[
                    "coordination_score"
                ],

                "link_anomaly_score": coordination[
                    "link_anomaly_score"
                ],

                "network_anomaly_score": coordination[
                    "network_anomaly_score"
                ],

                "network_growth_score": coordination[
                    "network_growth_score"
                ],

                "suspicious_amount_score": coordination[
                    "suspicious_amount_score"
                ],

                "severity": coordination[
                    "severity"
                ],

                "ring_type": coordination[
                    "ring_type"
                ],

                "community_count": community_count,

                "node_count": graph.number_of_nodes(),

                "edge_count": graph.number_of_edges(),

                "transaction_count": len(
                    event_transactions
                ),

                "total_amount": float(
                    event_transactions[
                        "amount"
                    ].sum()
                ),

                "high_risk_count": int(
                    event_transactions[
                        "predicted_fraud"
                    ].sum()
                ),

                "high_risk_amount": float(
                    event_transactions.loc[
                        event_transactions[
                            "predicted_fraud"
                        ] == 1,
                        "amount",
                    ].sum()
                ),

                "evidence": json.dumps(
                    evidence
                ),

                "top_suspicious_links": json.dumps(
                    suspicious_links,
                    default=str,
                ),

                "model_version": str(
                    event.get(
                        "model_version",
                        "unknown",
                    )
                ),

                "network_detector_version": self.model_version,

                "pipeline_version": "phase3_v1",

            }

            network_results.append(
                result
            )

            # -------------------------------------------
            # Save graph components
            # -------------------------------------------

            node_df = (
                self.graph_builder.graph_to_nodes(
                    graph,
                    event_id,
                )
            )

            edge_df = (
                self.graph_builder.graph_to_edges(
                    graph,
                    event_id,
                )
            )

            if not node_df.empty:
                node_df["community_id"] = (
                    node_df["node_id"]
                    .map(community_mapping)
                )

                all_nodes.append(
                    node_df
                )

            if not edge_df.empty:
                all_edges.append(
                    edge_df
                )

            if not community_df.empty:
                all_communities.append(
                    community_df
                )

            processed += 1

            if processed % 100 == 0:
                print(
                    f"Processed events: "
                    f"{processed:,}"
                )

        # -----------------------------------------------
        # Output
        # -----------------------------------------------

        print(
            "\n[4/7] Creating output tables..."
        )

        network_df = pd.DataFrame(
            network_results
        )

        nodes_df = (
            pd.concat(
                all_nodes,
                ignore_index=True,
            )
            if all_nodes
            else pd.DataFrame()
        )

        edges_df = (
            pd.concat(
                all_edges,
                ignore_index=True,
            )
            if all_edges
            else pd.DataFrame()
        )

        communities_df = (
            pd.concat(
                all_communities,
                ignore_index=True,
            )
            if all_communities
            else pd.DataFrame()
        )

        # -----------------------------------------------
        # Save
        # -----------------------------------------------

        print(
            "\n[5/7] Saving Phase 3 artifacts..."
        )

        network_path = (
            self.output_dir
            / "network_intelligence.parquet"
        )

        nodes_path = (
            self.output_dir
            / "network_nodes.parquet"
        )

        edges_path = (
            self.output_dir
            / "network_edges.parquet"
        )

        communities_path = (
            self.output_dir
            / "communities.parquet"
        )

        network_df.to_parquet(
            network_path,
            index=False,
        )

        nodes_df.to_parquet(
            nodes_path,
            index=False,
        )

        edges_df.to_parquet(
            edges_path,
            index=False,
        )

        communities_df.to_parquet(
            communities_path,
            index=False,
        )

        # -----------------------------------------------
        # Summary
        # -----------------------------------------------

        print(
            "\n[6/7] Creating summary..."
        )

        if not network_df.empty:

            severity_counts = (
                network_df[
                    "severity"
                ]
                .value_counts()
                .to_dict()
            )

            ring_counts = (
                network_df[
                    "ring_type"
                ]
                .value_counts()
                .to_dict()
            )

            summary = {
                "phase": "phase3",
                "pipeline_version": "phase3_v1",
                "model_version": self.model_version,

                "input_phase1_transactions": int(
                    len(transactions)
                ),

                "input_phase2_events": int(
                    len(events)
                ),

                "processed_events": int(
                    processed
                ),

                "skipped_events": int(
                    skipped
                ),

                "output_events": int(
                    len(network_df)
                ),

                "critical_events": int(
                    (
                        network_df["severity"]
                        == "CRITICAL"
                    ).sum()
                ),

                "high_events": int(
                    (
                        network_df["severity"]
                        == "HIGH"
                    ).sum()
                ),

                "watch_events": int(
                    (
                        network_df["severity"]
                        == "WATCH"
                    ).sum()
                ),

                "low_events": int(
                    (
                        network_df["severity"]
                        == "LOW"
                    ).sum()
                ),

                "severity_distribution":
                    severity_counts,

                "ring_type_distribution":
                    ring_counts,
            }

        else:

            summary = {
                "phase": "phase3",
                "pipeline_version": "phase3_v1",
                "model_version": self.model_version,
                "input_phase1_transactions": int(
                    len(transactions)
                ),
                "input_phase2_events": int(
                    len(events)
                ),
                "processed_events": 0,
                "skipped_events": int(
                    skipped
                ),
                "output_events": 0,
            }

        summary_path = (
            self.output_dir
            / "phase3_summary.json"
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                summary,
                f,
                indent=2,
                default=str,
            )

        # -----------------------------------------------
        # Final report
        # -----------------------------------------------

        print(
            "\n[7/7] Phase 3 completed."
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "PHASE 3 SUMMARY"
        )

        print(
            "=" * 70
        )

        print(
            f"Phase 1 transactions : "
            f"{len(transactions):,}"
        )

        print(
            f"Phase 2 events       : "
            f"{len(events):,}"
        )

        print(
            f"Processed events     : "
            f"{processed:,}"
        )

        print(
            f"Skipped events       : "
            f"{skipped:,}"
        )

        print(
            f"Network outputs      : "
            f"{len(network_df):,}"
        )

        print(
            f"\nOutput directory:"
            f"\n{self.output_dir}"
        )

        return {
            "network_intelligence": network_df,
            "nodes": nodes_df,
            "edges": edges_df,
            "communities": communities_df,
            "summary": summary,
        }
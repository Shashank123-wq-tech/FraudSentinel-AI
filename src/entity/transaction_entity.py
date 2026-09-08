"""
src/entity/transaction_entity.py
Canonical transaction schema (Section 2 / 4 of the Phase 2 spec).
"""

CANONICAL_TRANSACTION_COLUMNS = [
    "transaction_id", "timestamp", "merchant_id", "merchant_name", "card_id",
    "customer_id", "amount", "category", "customer_lat", "customer_long",
    "merchant_lat", "merchant_long", "fraud_label",  # fraud_label = eval-only
]

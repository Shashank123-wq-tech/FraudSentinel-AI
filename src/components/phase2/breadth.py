import numpy as np
import pandas as pd


def add_breadth_signal(
    windows: pd.DataFrame,
    card_saturation: float = 3.0,
) -> pd.DataFrame:
    """
    Calculate breadth / coordination evidence.

    Breadth answers:

        "How widely distributed is the abnormal activity?"

    It does NOT answer:

        "Is this fraud?"

    Therefore it is only a supporting component.
    """

    x = windows.copy()

    # ------------------------------------------------------------
    # Unique-card breadth
    # ------------------------------------------------------------

    unique_card_component = (
        1.0
        - np.exp(
            -x["unique_cards"]
            / max(card_saturation, 1e-8)
        )
    )

    # ------------------------------------------------------------
    # New-card breadth
    # ------------------------------------------------------------

    new_card_component = (
        x["new_card_rate"]
        .clip(0, 1)
    )

    # ------------------------------------------------------------
    # Combined breadth
    # ------------------------------------------------------------

    x["breadth_score"] = (
        0.70 * unique_card_component
        + 0.30 * new_card_component
    ).clip(0, 1)

    # ------------------------------------------------------------
    # Coordination
    # ------------------------------------------------------------

    x["coordination_score"] = (
        1.0
        - np.exp(
            -x["unique_cards"]
            / max(card_saturation, 1e-8)
        )
    ).clip(0, 1)

    return x
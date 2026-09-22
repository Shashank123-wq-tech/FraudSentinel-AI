import streamlit as st

from src.components.copilot.copilot import FraudSentinelCopilot


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FraudSentinel AI Copilot",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ FraudSentinel AI Copilot")

st.caption(
    "Defense-only fraud intelligence assistant "
    "grounded in FraudSentinel API evidence."
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


if "copilot" not in st.session_state:

    try:

        st.session_state.copilot = (
            FraudSentinelCopilot()
        )

    except Exception as exc:

        st.error(
            f"Copilot initialization failed: {exc}"
        )

        st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("System")

    st.success(
        "FraudSentinel Intelligence API"
    )

    st.caption(
        "The Copilot retrieves evidence from "
        "the existing FraudSentinel API before "
        "generating an explanation."
    )

    st.divider()

    if st.button(
        "Clear conversation",
        width="stretch",
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# DISPLAY CONVERSATION HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask FraudSentinel about risk, spikes, XAI, impact or response..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Store user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------------------
    # Assistant response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing FraudSentinel evidence..."
        ):

            try:

                # ------------------------------------------------
                # Preserve previous conversation
                # ------------------------------------------------

                history = [
                    {
                        "role": message["role"],
                        "content": message["content"],
                    }
                    for message
                    in st.session_state.messages[:-1]
                ]


                # ------------------------------------------------
                # Call Copilot
                #
                # IMPORTANT:
                # FraudSentinelCopilot.ask() expects
                # `conversation`, not `conversation_history`.
                # ------------------------------------------------

                result = (
                    st.session_state
                    .copilot
                    .ask(
                        question=question,
                        conversation=history,
                    )
                )


                # ------------------------------------------------
                # Extract answer
                # ------------------------------------------------

                answer = result.get(
                    "answer",
                    "No answer was generated.",
                )

                st.markdown(answer)


                # ------------------------------------------------
                # Evidence
                # ------------------------------------------------

                evidence_context = result.get(
                    "evidence_context",
                    {},
                )

                with st.expander(
                    "🔎 Evidence retrieved"
                ):

                    if evidence_context:

                        st.json(
                            evidence_context
                        )

                    else:

                        st.info(
                            "No evidence was returned."
                        )


                # ------------------------------------------------
                # Metadata
                # ------------------------------------------------

                model_name = result.get(
                    "model",
                    "unknown",
                )

                intent = result.get(
                    "intent",
                    "general",
                )

                route_confidence = result.get(
                    "route_confidence",
                    0.0,
                )

                tools_used = result.get(
                    "tools_used",
                    [],
                )


                st.caption(
                    f"Model: {model_name}"
                )

                st.caption(
                    f"Intent: {intent} | "
                    f"Route confidence: "
                    f"{float(route_confidence):.2f}"
                )

                if tools_used:

                    st.caption(
                        "Tools used: "
                        + ", ".join(
                            str(tool)
                            for tool in tools_used
                        )
                    )


                # ------------------------------------------------
                # Store assistant answer
                # ------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )


            except Exception as exc:

                # ------------------------------------------------
                # Display error
                # ------------------------------------------------

                error_message = (
                    f"Copilot request failed: {exc}"
                )

                st.error(
                    error_message
                )


                # ------------------------------------------------
                # Store error in conversation
                # ------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )
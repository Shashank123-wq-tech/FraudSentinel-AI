import streamlit as st

from src.components.copilot.copilot import (
    FraudSentinelCopilot,
)


st.set_page_config(
    page_title="FraudSentinel AI Copilot",
    page_icon="🛡️",
    layout="wide",
)


st.title("🛡️ FraudSentinel AI Copilot")

st.caption(
    "Defense-only fraud intelligence assistant "
    "grounded in FraudSentinel API evidence."
)


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


with st.sidebar:

    st.header("System")

    st.success(
        "FraudSentinel Intelligence API"
    )

    st.caption(
        "The Copilot retrieves evidence from "
        "the existing API before generating "
        "an explanation."
    )

    if st.button(
        "Clear conversation",
        width="stretch",
    ):

        st.session_state.messages = []

        st.rerun()


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


question = st.chat_input(
    "Ask FraudSentinel about risk, spikes, XAI, impact or response..."
)


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing FraudSentinel evidence..."
        ):

            try:

                history = [
                    {
                        "role": message["role"],
                        "content": message["content"],
                    }
                    for message
                    in st.session_state.messages[:-1]
                ]

                result = (
                    st.session_state
                    .copilot
                    .ask(
                        question,
                        conversation_history=history,
                    )
                )

                answer = result["answer"]

                st.markdown(answer)

                with st.expander(
                    "🔎 Evidence retrieved"
                ):

                    st.json(
                        result["evidence_context"]
                    )

                st.caption(
                    f"Model: {result['model']}"
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as exc:

                error_message = (
                    f"Copilot request failed: {exc}"
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )
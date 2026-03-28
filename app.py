"""
Streamlit parent coaching chatbot application.

This file is the entry point for the parent_coach_bot module. It renders a
Hebrew-language chat interface that guides a parent through a four-question
behavioral-analysis protocol about an interaction with their child, using Azure
OpenAI GPT-4o to evaluate answers and generate supportive feedback.

Update this file when the UI, conversation flow, hard validation rules, or LLM
prompting logic changes.
Run with: streamlit run app.py
"""

import logging
import os
import streamlit as st
from openai import OpenAI

from llm_client import initialize_client, get_deployment, get_temperature, get_max_tokens
from protocol import PROTOCOL
from validator import validate_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "אתה מאמן הורות תומך ואמפתי שעוזר להורים לנתח אינטראקציות עם ילדיהם "
    "באמצעות פרוטוקול מובנה. אתה תמיד מגיב בעברית. אתה לא שיפוטי, לא ביקורתי, "
    "ומעודד חשיבה עצמית והתפתחות."
)

STEP_EVENT = 0     # Waiting for parent to describe the event
STEP_Q1 = 1        # Protocol question 1
STEP_Q2 = 2        # Protocol question 2
STEP_Q3 = 3        # Protocol question 3
STEP_Q4 = 4        # Protocol question 4
STEP_SUMMARY = 5   # Final summary produced

# ── LLM helpers ───────────────────────────────────────────────────────────────

def _call_llm(client: OpenAI, messages: list[dict]) -> str:
    """
    Send a list of chat messages to the LLM and return the text response.

    Args:
        client: Authenticated OpenAI client.
        messages: List of {"role": ..., "content": ...} dicts.

    Returns:
        Response text from the model.
    """
    response = client.chat.completions.create(
        model=get_deployment(),
        messages=messages,
        temperature=get_temperature(),
        max_tokens=get_max_tokens(),
    )
    return response.choices[0].message.content.strip()


def generate_empathy_response(client: OpenAI, event_description: str) -> str:
    """
    Generate a short empathetic acknowledgment of the parent's event description.

    Args:
        client: Authenticated OpenAI client.
        event_description: The parent's description of the event.

    Returns:
        Empathetic Hebrew response string (2-3 sentences, no advice or questions).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"הורה תיאר את האירוע הבא עם ילדו:\n\n\"{event_description}\"\n\n"
                "כתוב תגובה קצרה (2-3 משפטים) שמביעה הבנה ואמפתיה כלפי ההורה. "
                "אל תשאל שאלות ואל תציע עצות – רק הכר ברגע."
            ),
        },
    ]
    return _call_llm(client, messages)


def generate_guidance(client: OpenAI, question: str, feedback: str) -> str:
    """
    Generate a gentle, supportive message asking the parent to refine their answer.

    Args:
        client: Authenticated OpenAI client.
        question: The Hebrew protocol question that was asked.
        feedback: The evaluation feedback explaining what was missing.

    Returns:
        Hebrew guidance string. Does not repeat the full question.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"הערכת תשובת ההורה לשאלה:\n\"{question}\"\n\n"
                f"המשוב הפנימי:\n{feedback}\n\n"
                "כתוב הודעה קצרה, תומכת ולא שיפוטית שמסבירה להורה מה כדאי לדייק, "
                "ומזמינה אותו לנסות שוב. אל תחזור על השאלה המלאה – רק הכוון עדין."
            ),
        },
    ]
    return _call_llm(client, messages)


def generate_summary(
    client: OpenAI,
    event_description: str,
    answers: dict[int, str],
) -> str:
    """
    Generate a short reflective summary after all four protocol questions are answered.

    The summary covers: the event, the child's action, the child's goal, whether
    the goal was achieved, and whether the behavior pattern was reinforced.

    Args:
        client: Authenticated OpenAI client.
        event_description: The parent's original event description.
        answers: Dict mapping question id (1-4) to the accepted answer.

    Returns:
        Hebrew summary string with a warm, empowering tone.
    """
    qa_block = "\n".join(
        f"שאלה {p['id']}: {p['question']}\nתשובה: {answers[p['id']]}"
        for p in PROTOCOL
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "להלן סיכום שיחת האימון עם ההורה:\n\n"
                f"תיאור האירוע:\n{event_description}\n\n"
                f"{qa_block}\n\n"
                "כתוב סיכום קצר (4-5 משפטים) שכולל:\n"
                "• מה הייתה הפעולה השלילית של הילד\n"
                "• מה הייתה המטרה שלו\n"
                "• האם המטרה הושגה\n"
                "• האם דפוס הפעולה חוזק\n"
                "• המלצה אחת בונה קדימה\n"
                "הטון יהיה חמים, תומך ואמפתי."
            ),
        },
    ]
    return _call_llm(client, messages)


# ── Session state helpers ──────────────────────────────────────────────────────

def _init_session() -> None:
    """Initialize all session state variables on first run."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "step" not in st.session_state:
        st.session_state.step = STEP_EVENT
    if "event_description" not in st.session_state:
        st.session_state.event_description = ""
    if "protocol_answers" not in st.session_state:
        # Maps question id (1-4) to the accepted answer string
        st.session_state.protocol_answers: dict[int, str] = {}
    if "attempt_counts" not in st.session_state:
        # Maps question id (1-4) to number of failed attempts so far
        st.session_state.attempt_counts: dict[int, int] = {}
    if "client" not in st.session_state:
        # Show available secret keys (not values) to help diagnose missing secrets
        secret_keys = list(st.secrets.keys()) if st.secrets else []
        logger.info("st.secrets keys available: %s", secret_keys)

        required = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT_NAME"]
        missing = [k for k in required if k not in st.secrets and not os.getenv(k)]
        if missing:
            st.error(
                "**Missing secrets:** " + ", ".join(f"`{k}`" for k in missing) + "\n\n"
                "Go to **Manage app → Settings → Secrets** and paste:\n"
                "```toml\n"
                'AZURE_OPENAI_ENDPOINT = "https://api.openai.com/v1"\n'
                'AZURE_OPENAI_API_KEY = "sk-proj-..."\n'
                'AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o"\n'
                'OPENAI_TEMPERATURE = "0.7"\n'
                'OPENAI_MAX_TOKENS = "4000"\n'
                "```\n\n"
                f"Keys currently found in st.secrets: `{secret_keys}`"
            )
            st.stop()
        st.session_state.client = initialize_client()


def _add_message(role: str, content: str) -> None:
    """
    Append a message to the session chat history.

    Args:
        role: "assistant" or "user".
        content: Message text.
    """
    st.session_state.messages.append({"role": role, "content": content})


def _current_protocol_step() -> dict:
    """
    Return the PROTOCOL entry for the current conversation step.

    Returns:
        Protocol dict with keys 'id', 'question', 'guidelines'.
    """
    # step 1 → PROTOCOL[0], step 2 → PROTOCOL[1], etc.
    return PROTOCOL[st.session_state.step - 1]


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(
        page_title="Parent Coaching Assistant",
        page_icon="🤝",
        layout="centered",
    )

    # Right-to-left support for Hebrew
    st.markdown(
        "<style>body, .stChatMessage { direction: rtl; text-align: right; }</style>",
        unsafe_allow_html=True,
    )

    st.title("🤝 Parent Coaching Assistant")
    st.caption("עוזר אימון הורים | מבוסס על פרוטוקול מובנה")

    # On Streamlit Cloud, secrets live in st.secrets (not os.environ).
    # Copying them into os.environ lets llm_client.py read them via os.getenv.
    for _key, _val in st.secrets.items():
        os.environ.setdefault(_key, str(_val))

    _init_session()

    client: OpenAI = st.session_state.client

    # ── Render existing chat history ──────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Show 3-part opening when conversation is fresh ────────────────────────
    if not st.session_state.messages:
        opening_lines = [
            "כל הכבוד שעצרת רגע והחלטת להתייעץ. זה לא מובן מאליו.",
            "אני מזמין אותך להתמקד בסיטואציה אחת שקרתה עכשיו או לאחרונה, "
            "שבה הילד התנהג בצורה שהיית רוצה לשנות.",
            "תאר/י לי מה קרה.",
        ]
        for line in opening_lines:
            _add_message("assistant", line)
            with st.chat_message("assistant"):
                st.markdown(line)

    # ── Handle user input ─────────────────────────────────────────────────────
    user_input = st.chat_input("כתוב/י כאן...")

    if not user_input:
        return

    with st.chat_message("user"):
        st.markdown(user_input)
    _add_message("user", user_input)

    step = st.session_state.step
    print(f"\n[DEBUG] step={step} | answer={user_input!r}")

    # ── Step 0: Parent describes the event ───────────────────────────────────
    if step == STEP_EVENT:
        st.session_state.event_description = user_input

        with st.spinner("חושב..."):
            empathy = generate_empathy_response(client, user_input)
            first_q = PROTOCOL[0]["question"]
            bot_reply = (
                f"{empathy}\n\n---\n\n**שאלה 1 מתוך 4:**\n\n{first_q}"
            )

        _add_message("assistant", bot_reply)
        with st.chat_message("assistant"):
            st.markdown(bot_reply)

        st.session_state.step = STEP_Q1

    # ── Steps 1-4: Protocol questions ────────────────────────────────────────
    elif STEP_Q1 <= step <= STEP_Q4:
        protocol_entry = _current_protocol_step()
        q_id = protocol_entry["id"]

        # Current attempt count for this question (0-based before this attempt)
        attempt_count = st.session_state.attempt_counts.get(q_id, 0)

        with st.spinner("מעריך את התשובה..."):
            evaluation = validate_answer(
                client=client,
                question_id=q_id,
                question=protocol_entry["question"],
                guidelines=protocol_entry["guidelines"],
                answer=user_input,
                attempt_count=attempt_count,
            )

        alignment = evaluation.get("alignment", "not_aligned")
        confidence = evaluation.get("confidence", 0.5)
        should_accept = evaluation.get("should_accept", False)
        feedback = evaluation.get("feedback", "")

        print(
            f"[DEBUG] q_id={q_id} | alignment={alignment} | confidence={confidence:.2f} "
            f"| should_accept={should_accept} | attempts={attempt_count}"
        )
        logger.info(
            "step=%d | q_id=%d | alignment=%s | should_accept=%s | attempts=%d",
            step, q_id, alignment, should_accept, attempt_count,
        )

        if should_accept:
            # Store the accepted answer, reset attempt count, and advance
            st.session_state.protocol_answers[q_id] = user_input
            st.session_state.attempt_counts[q_id] = 0
            next_step = step + 1

            if next_step <= STEP_Q4:
                next_entry = PROTOCOL[next_step - 1]
                bot_reply = (
                    f"תודה!\n\n---\n\n"
                    f"**שאלה {next_step} מתוך 4:**\n\n{next_entry['question']}"
                )
                st.session_state.step = next_step
            else:
                # All four questions answered → produce summary
                with st.spinner("מכין סיכום..."):
                    summary = generate_summary(
                        client,
                        st.session_state.event_description,
                        st.session_state.protocol_answers,
                    )
                bot_reply = (
                    "תודה!\n\n---\n\n"
                    "✨ **סיכום השיחה:**\n\n"
                    f"{summary}"
                )
                st.session_state.step = STEP_SUMMARY

        else:
            # Increment attempt count for next try
            st.session_state.attempt_counts[q_id] = attempt_count + 1

            if alignment == "invalid_format":
                # Q4 hard rule: echo the fixed feedback directly, no LLM guidance call
                bot_reply = feedback
            else:
                with st.spinner("מכין הנחיה..."):
                    bot_reply = generate_guidance(
                        client,
                        protocol_entry["question"],
                        feedback,
                    )

        _add_message("assistant", bot_reply)
        with st.chat_message("assistant"):
            st.markdown(bot_reply)

    # ── Step 5: Summary already shown ────────────────────────────────────────
    elif step == STEP_SUMMARY:
        closing = (
            "תודה רבה על השיחה הזו. 🙏 "
            "אם תרצה/י לדון באירוע נוסף, רענן/י את הדף ונתחיל מחדש."
        )
        _add_message("assistant", closing)
        with st.chat_message("assistant"):
            st.markdown(closing)


if __name__ == "__main__":
    main()

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

import json
import logging
import streamlit as st
from openai import OpenAI

from llm_client import initialize_client, get_deployment, get_temperature, get_max_tokens
from protocol import PROTOCOL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "אתה מאמן הורות תומך ואמפתי שעוזר להורים לנתח אינטראקציות עם ילדיהם "
    "באמצעות פרוטוקול מובנה. אתה תמיד מגיב בעברית. אתה לא שיפוטי, לא ביקורתי, "
    "ומעודד חשיבה עצמית והתפתחות."
)

EVAL_SYSTEM_PROMPT = (
    "You evaluate whether a parent's answer follows a structured parenting "
    "reflection protocol. Be strict but fair. Always respond with valid JSON only."
)

STEP_EVENT = 0     # Waiting for parent to describe the event
STEP_Q1 = 1        # Protocol question 1
STEP_Q2 = 2        # Protocol question 2
STEP_Q3 = 3        # Protocol question 3
STEP_Q4 = 4        # Protocol question 4
STEP_SUMMARY = 5   # Final summary produced

# Accepted alignment value that causes the bot to advance to the next step
ALIGNED = "aligned"

# Q4 only accepts one of these two answers (after stripping whitespace)
Q4_VALID_ANSWERS = {"כן", "לא"}


# ── Hard validation rules (applied before calling the LLM) ───────────────────

def _hard_validate_q1(answer: str) -> dict | None:
    """
    Apply the hard rule for question 1: answer must not contain the word 'לא'.

    Args:
        answer: The parent's raw answer string.

    Returns:
        A result dict if the rule fires (automatically not_aligned), else None.
    """
    if "לא" in answer.split():
        return {
            "alignment": "not_aligned",
            "feedback": "התשובה צריכה לתאר מה הילד כן עשה, ולא מה הוא לא עשה.",
        }
    return None


def _hard_validate_q4(answer: str) -> dict | None:
    """
    Apply the hard rule for question 4: only 'כן' or 'לא' are accepted.

    Args:
        answer: The parent's raw answer string.

    Returns:
        A result dict prompting the parent to re-answer if invalid, else None.
        Uses a special alignment value "invalid_format" so the bot stays on Q4
        and does not call the LLM.
    """
    if answer.strip() not in Q4_VALID_ANSWERS:
        return {
            "alignment": "invalid_format",
            "feedback": "אנא ענה/י רק כן או לא.",
        }
    return None


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


def evaluate_answer(
    client: OpenAI,
    question_id: int,
    question: str,
    guidelines: str,
    parent_answer: str,
) -> dict:
    """
    Evaluate whether the parent's answer aligns with the protocol guidelines via LLM.

    Hard validation rules for Q1 and Q4 are applied before calling the LLM.

    Args:
        client: Authenticated OpenAI client.
        question_id: The protocol question id (1-4).
        question: The Hebrew question text.
        guidelines: Alignment guidelines for this question.
        parent_answer: The parent's answer to evaluate.

    Returns:
        Dict with keys "alignment" ("aligned" | "partially_aligned" | "not_aligned"
        | "invalid_format") and "feedback" (Hebrew string).
    """
    # Apply hard rules first — no LLM call needed if they trigger
    if question_id == 1:
        hard_result = _hard_validate_q1(parent_answer)
        if hard_result:
            return hard_result

    if question_id == 4:
        hard_result = _hard_validate_q4(parent_answer)
        if hard_result:
            return hard_result

    evaluation_prompt = (
        f"Protocol question:\n{question}\n\n"
        f"Alignment guidelines:\n{guidelines}\n\n"
        f"Parent's answer:\n{parent_answer}\n\n"
        "Evaluate whether the parent's answer follows the guidelines.\n"
        "Respond ONLY with valid JSON, no extra text:\n"
        '{"alignment": "aligned | partially_aligned | not_aligned", '
        '"feedback": "short explanation in Hebrew"}'
    )
    messages = [
        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
        {"role": "user", "content": evaluation_prompt},
    ]
    raw = _call_llm(client, messages)

    # Strip markdown fences if the model wraps the JSON
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())


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
    if "client" not in st.session_state:
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

    # ── Show initial bot prompt when conversation is fresh ────────────────────
    if not st.session_state.messages:
        welcome = "תאר/י בקצרה אירוע שקרה עם הילד/ה."
        _add_message("assistant", welcome)
        with st.chat_message("assistant"):
            st.markdown(welcome)

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

        with st.spinner("מעריך את התשובה..."):
            evaluation = evaluate_answer(
                client=client,
                question_id=q_id,
                question=protocol_entry["question"],
                guidelines=protocol_entry["guidelines"],
                parent_answer=user_input,
            )

        alignment = evaluation.get("alignment", "not_aligned")
        feedback = evaluation.get("feedback", "")

        print(f"[DEBUG] question_id={q_id} | alignment={alignment} | feedback={feedback!r}")
        logger.info("step=%d | q_id=%d | alignment=%s", step, q_id, alignment)

        if alignment == ALIGNED:
            # Store the accepted answer and move forward
            st.session_state.protocol_answers[q_id] = user_input
            next_step = step + 1

            if next_step <= STEP_Q4:
                next_entry = PROTOCOL[next_step - 1]
                bot_reply = (
                    f"תודה! ✓\n\n---\n\n"
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
                    "תודה! ✓\n\n---\n\n"
                    "✨ **סיכום השיחה:**\n\n"
                    f"{summary}"
                )
                st.session_state.step = STEP_SUMMARY

        else:
            # partially_aligned, not_aligned, or invalid_format
            if alignment == "invalid_format":
                # Q4 hard rule: just echo the feedback directly, no LLM guidance call
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

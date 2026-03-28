"""
Streamlit parent coaching chatbot application.

This file is the entry point for the parent_coach_bot module. It renders a
Hebrew-language chat interface that guides a parent through a structured
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
STEP_Q5 = 5        # Protocol question 5
STEP_SUMMARY = 6   # Summary shown; waiting for yes/no on another scenario
STEP_CLOSED = 7    # User declined another scenario; chat input ignored

NUM_QUESTIONS = 5  # Total number of protocol questions

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


def generate_alternative_child_goals(
    client: OpenAI,
    event_description: str,
    q1_answer: str,
    previous_q2_goal: str,
    q3_answer: str,
) -> str:
    """
    Suggest a few plausible alternative phrasings of the child's goal for a Q2 restart.

    Used when Q4 cannot be tied to the stated goal relative to Q3 — the parent
    is sent back to Q2 with ideas, not prescriptions.

    Args:
        client: Authenticated OpenAI client.
        event_description: Parent's event narrative.
        q1_answer: Child's action (Q1).
        previous_q2_goal: The goal the parent had stated in Q2 before restart.
        q3_answer: Parent's described reaction (Q3).

    Returns:
        Hebrew markdown: short intro line plus a bulleted list (3–4 items).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "ההורה מתאר/ת סיטואציה ואת הפעולה של הילד, מטרה שהציע/ה לילד, ואת תגובת ההורה. "
                "נראה שמטרת הילד שהוצעה והפעולה שההורא תיאר/ה לא מתיישבות — התוצאה לא ניתנת לייחס "
                "למטרה בהקשר.\n\n"
                f"תיאור האירוע:\n{event_description}\n\n"
                f"פעולת הילד (Q1):\n{q1_answer}\n\n"
                f"מטרת הילד שההורה ציין/ה קודם (Q2):\n{previous_q2_goal}\n\n"
                f"תגובת ההורה (Q3):\n{q3_answer}\n\n"
                "כתוב בעברית פסקה קצרה ואז רשימת נקודות (3–4 פריטים) של דוגמאות אפשריות "
                "למטרות שהילד אולי ניסה להשיג — רלוונטיות לסיטואציה. "
                "אל תטיף מוסר; אל תאמר שההורה טעה. "
                "הדגש: אלה הצעות לשיקול ההורה, לא תשובה נכונה אחת. "
                "השתמש בפורמט markdown עם כותרת קצרה ואז שורות המתחילות ב-• או -."
            ),
        },
    ]
    return _call_llm(client, messages)


def generate_guidance_q4(
    client: OpenAI,
    question: str,
    feedback: str,
    child_goal: str,
    q3_answer: str,
    parent_answer: str,
) -> str:
    """
    Ask the parent to restate Q4 in terms of the child's goal (not only affect).

    Used when q4_outcome_unclear without protocol restart — e.g. attention goal but
    answer only says 'kept crying'; expected sharpening: 'got my attention', etc.

    Args:
        client: Authenticated OpenAI client.
        question: Hebrew Q4 protocol question.
        feedback: Validator feedback (internal).
        child_goal: Accepted Q2 answer.
        q3_answer: Accepted Q3 answer.
        parent_answer: Parent's latest Q4 text.

    Returns:
        Short Hebrew guidance; does not restart Q2.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"שאלת הפרוטוקול: \"{question}\"\n\n"
                f"מטרת הילד (Q2):\n{child_goal}\n\n"
                f"תגובת ההורה (Q3):\n{q3_answer}\n\n"
                f"תשובת ההורה לשאלה על התוצאה (Q4):\n{parent_answer}\n\n"
                f"משוב פנימי:\n{feedback}\n\n"
                "התשובה לא מנוסחת ביחס למטרת הילד — רק תיאור רגש/התנהגות (למשל המשך בכי) "
                "בלי לומר אם **מטרת הילד** הושגה או לא.\n\n"
                "כתוב הודעה קצרה בעברית (3–5 משפטים):\n"
                "הסבר בעדינות שצריך לדייק: התוצאה צריכה להתייחס למטרה שציינת לילד. "
                "לדוגמה אם המטרה היא תשומת לב או רחמים — לתאר האם הילד קיבל תשומת לב "
                "(גם אם דרך צעקה או המשך תגובה), לא רק 'המשיך לבכות'. "
                "אל תאתחל מחדש פרוטוקול משאלה 2; רק בקש לנסח מחדש את התוצאה ביחס למטרה. "
                "אל תחזור על ניסוח השאלה המלא."
            ),
        },
    ]
    return _call_llm(client, messages)


def generate_guidance(
    client: OpenAI,
    question: str,
    feedback: str,
    event_context: str = "",
    q1_answer: str = "",
    child_goal: str = "",
    parent_answer: str = "",
) -> str:
    """
    Generate a gentle, supportive message asking the parent to refine their answer.

    Q2: When event_context + q1_answer are supplied, suggests situation-specific goal examples.
    Q3: When child_goal is also supplied, gives a brief explanation of the mismatch and
        politely asks again for a reaction that directly relates to the child's stated goal.

    Args:
        client: Authenticated OpenAI client.
        question: The Hebrew protocol question that was asked.
        feedback: The evaluation feedback explaining what was missing.
        event_context: The parent's original event description (optional).
        q1_answer: The accepted answer to Q1 — the child's action (optional).
        child_goal: The accepted answer to Q2 — the child's goal (optional).
        parent_answer: The parent's latest answer text (optional; used for Q3).

    Returns:
        Hebrew guidance string. Does not repeat the full question.
    """
    if child_goal:
        content = (
            f"שאלת הפרוטוקול: \"{question}\"\n\n"
            f"מטרת הילד (לפי תיאור ההורה בשאלה 2 — זו מטרת הילד, לא מטרת ההורה הנפרדת):\n"
            f"{child_goal}\n\n"
            f"מה שההורה כתב עכשיו:\n{parent_answer}\n\n"
            f"הערכה פנימית — למה התשובה לא עומדת בדרישה:\n{feedback}\n\n"
            "הקשר: מטרת ההורה ומטרת הילד עלולות להתנגש — אין כאן עמדה מוסרית או חינוכית. "
            "רק בודקים אם הפעולה שתיאר ההורה קשורה ישירות למטרת הילד: עזרה להשגה, או חסימה — "
            "או שלא קשורה (אז צריך פעולה אחרת).\n\n"
            "כתוב הודעה קצרה בעברית (2–4 משפטים בלבד):\n"
            "1) הסבר בקצרה למה התשובה לא מתאימה (למשל: הפעולה לא עוזרת ולא חוסמת את מטרת הילד — "
            "רק לא קשורה אליה).\n"
            "2) בקשה מנומסת לנסות שוב: לתאר משהו שההורה עשה שקשור ישירות למטרת הילד "
            "שלמעלה — פעולה שעזרה לילד לכיוון המטרה או שחסמה אותה.\n"
            "אל תוסיף עידוד ארוך, אל תשאל שאלות פתוחות נוספות, ואל תחזור על ניסוח השאלה המלא."
        )
    elif event_context and q1_answer:
        # Q2-specific guidance: include situation context and suggest relevant examples
        content = (
            f"הערכת תשובת ההורה לשאלה:\n\"{question}\"\n\n"
            f"המשוב הפנימי:\n{feedback}\n\n"
            f"תיאור הסיטואציה:\n{event_context}\n\n"
            f"הפעולה שהילד עשה (Q1):\n{q1_answer}\n\n"
            "כתוב הודעה קצרה, תומכת ולא שיפוטית שמסבירה להורה שהמטרה שציין לא ממש "
            "מתאימה לסיטואציה. הצע 2-3 דוגמאות ספציפיות למטרות שהיו הגיוניות "
            "בהקשר של הסיטואציה שתואר, ומזמינה אותו לנסות שוב. "
            "אל תחזור על השאלה המלאה."
        )
    else:
        content = (
            f"הערכת תשובת ההורה לשאלה:\n\"{question}\"\n\n"
            f"המשוב הפנימי:\n{feedback}\n\n"
            "כתוב הודעה קצרה, תומכת ולא שיפוטית שמסבירה להורה מה כדאי לדייק, "
            "ומזמינה אותו לנסות שוב. אל תחזור על השאלה המלאה – רק הכוון עדין."
        )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
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


def _wants_another_scenario(text: str) -> bool | None:
    """
    Interpret the parent's reply after the summary (another scenario or stop).

    Args:
        text: User message.

    Returns:
        True if they want a new scenario, False if they want to stop, None if unclear.
    """
    t = text.strip().lower()
    if not t:
        return None
    if t.startswith("לא") or t.startswith("no"):
        return False
    if "לא תודה" in t or "מספיק" in t or "לא רוצה" in t or "לא עכשיו" in t:
        return False
    yes_markers = (
        "כן", "בטח", "בוודאי", "בשמחה", "יאללה", "קדימה", "התחל", "נסה", "אשמח",
    )
    if any(m in t for m in yes_markers):
        return True
    if "רוצה" in t and "לא" not in t.split()[:2]:
        return True
    return None


def _reset_session_for_new_scenario() -> None:
    """
    Clear protocol state and chat so the parent can describe a new event from scratch.

    Keeps the OpenAI client in session. Next run shows the opening prompts again.
    """
    st.session_state.messages = []
    st.session_state.event_description = ""
    st.session_state.protocol_answers = {}
    st.session_state.attempt_counts = {}
    st.session_state.step = STEP_EVENT


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
            "התמקד/י בפעולה שהילד שלך עושה עכשיו, או עשה בסיטואציה שבה אתה מנסה לתקן – "
            "פעולה שמרגישה בלתי נסבלת ושהיית רוצה לשנות.",
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

    if st.session_state.step == STEP_CLOSED:
        return

    step = st.session_state.step

    # After summary: another scenario (fresh context) or goodbye — before logging user turn
    if step == STEP_SUMMARY:
        choice = _wants_another_scenario(user_input)
        if choice is True:
            _reset_session_for_new_scenario()
            st.rerun()
            return
        elif choice is False:
            with st.chat_message("user"):
                st.markdown(user_input)
            _add_message("user", user_input)
            goodbye = (
                "תודה רבה על השיחה. 🙏 "
                "אם תרצה/י לחזור בעתיד, רענן/י את הדף או התחל/י סיטואציה חדשה."
            )
            _add_message("assistant", goodbye)
            with st.chat_message("assistant"):
                st.markdown(goodbye)
            st.session_state.step = STEP_CLOSED
        else:
            with st.chat_message("user"):
                st.markdown(user_input)
            _add_message("user", user_input)
            clarify = (
                "לא הבנתי בבירור. "
                "כתוב/י **כן** אם תרצה/י **סיטואציה חדשה** (מתחילים מאפס עם אירוע אחר), "
                "או **לא** כדי לסיים לעת עתה."
            )
            _add_message("assistant", clarify)
            with st.chat_message("assistant"):
                st.markdown(clarify)
        return

    with st.chat_message("user"):
        st.markdown(user_input)
    _add_message("user", user_input)

    print(f"\n[DEBUG] step={step} | answer={user_input!r}")

    # ── Step 0: Parent describes the event ───────────────────────────────────
    if step == STEP_EVENT:
        st.session_state.event_description = user_input

        with st.spinner("חושב..."):
            empathy = generate_empathy_response(client, user_input)
            first_q = PROTOCOL[0]["question"]
            bot_reply = (
                f"{empathy}\n\n---\n\n**שאלה 1 מתוך {NUM_QUESTIONS}:**\n\n{first_q}"
            )

        _add_message("assistant", bot_reply)
        with st.chat_message("assistant"):
            st.markdown(bot_reply)

        st.session_state.step = STEP_Q1

    # ── Steps 1-5: Protocol questions ────────────────────────────────────────
    elif STEP_Q1 <= step <= STEP_Q5:
        protocol_entry = _current_protocol_step()
        q_id = protocol_entry["id"]

        # Current attempt count for this question (0-based before this attempt)
        attempt_count = st.session_state.attempt_counts.get(q_id, 0)

        # Build context block for validation — Q3/Q4 need goal; Q4 also needs Q3 answer
        if q_id == 3:
            child_goal = st.session_state.protocol_answers.get(2, "")
            validation_context = (
                f"Event: {st.session_state.event_description}\n"
                f"Child's action (Q1): {st.session_state.protocol_answers.get(1, '')}\n"
                f"Child's goal (Q2): {child_goal}"
            )
        elif q_id == 4:
            validation_context = (
                f"Event: {st.session_state.event_description}\n"
                f"Child's action (Q1): {st.session_state.protocol_answers.get(1, '')}\n"
                f"Child's goal (Q2): {st.session_state.protocol_answers.get(2, '')}\n"
                f"Parent's action (Q3): {st.session_state.protocol_answers.get(3, '')}"
            )
        else:
            validation_context = st.session_state.event_description

        with st.spinner("מעריך את התשובה..."):
            evaluation = validate_answer(
                client=client,
                question_id=q_id,
                question=protocol_entry["question"],
                guidelines=protocol_entry["guidelines"],
                answer=user_input,
                attempt_count=attempt_count,
                event_context=validation_context,
            )

        alignment = evaluation.get("alignment", "not_aligned")
        confidence = evaluation.get("confidence", 0.5)
        should_accept = evaluation.get("should_accept", False)
        feedback = evaluation.get("feedback", "")
        restart_at_q2 = evaluation.get("restart_at_q2", False)

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

            if next_step <= STEP_Q5:
                next_entry = PROTOCOL[next_step - 1]

                if next_step == STEP_Q2:
                    # Before Q2 explain WHY understanding the child's goal matters
                    bot_reply = (
                        "תודה!\n\n"
                        "לכל פעולה שילד עושה יש מטרה או צורך שהוא מנסה להשיג. "
                        "כדי לשנות את ההתנהגות, צריך לוודא שהילד יכול להגיע לאותה מטרה "
                        "רק דרך התנהגות אחרת.\n\n"
                        "בוא/י נצא לגלות מה המטרה שהילד ניסה להשיג עם הפעולה הזו.\n\n"
                        f"---\n\n**שאלה {next_step} מתוך {NUM_QUESTIONS}:**\n\n{next_entry['question']}"
                    )
                else:
                    bot_reply = (
                        f"תודה!\n\n---\n\n"
                        f"**שאלה {next_step} מתוך {NUM_QUESTIONS}:**\n\n{next_entry['question']}"
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
                    f"{summary}\n\n"
                    "---\n\n"
                    "**רוצה/י לנסות סיטואציה נוספת?**\n\n"
                    "אם כן — נבנה הקשר חדש מאפס: תיאור אירוע חדש ואותו פרוטוקול מההתחלה.\n\n"
                    "כתוב/י **כן** או **בטח** להתחלה חדשה, או **לא** לסיום."
                )
                st.session_state.step = STEP_SUMMARY

        else:
            if restart_at_q2 and q_id == 4:
                # Q4 outcome cannot be tied to child's goal vs Q3 — restart protocol at Q2
                prev_goal = st.session_state.protocol_answers.get(2, "")
                q1_stored = st.session_state.protocol_answers.get(1, "")
                q3_stored = st.session_state.protocol_answers.get(3, "")
                for _k in (2, 3, 4):
                    st.session_state.protocol_answers.pop(_k, None)
                for _k in (2, 3, 4):
                    st.session_state.attempt_counts[_k] = 0
                st.session_state.step = STEP_Q2
                q2_text = PROTOCOL[1]["question"]
                with st.spinner("מכין הצעות למטרת הילד..."):
                    alternatives = generate_alternative_child_goals(
                        client,
                        st.session_state.event_description,
                        q1_stored,
                        prev_goal,
                        q3_stored,
                    )
                bot_reply = (
                    "### מתחילים מחדש את הפרוטוקול משאלה 2\n\n"
                    "התשובה על התוצאה (שאלה 4) לא מאפשרת להבין אם **מטרת הילד** הושגה או לא "
                    "בהתאם לפעולה שתיארת בשאלה 3. "
                    "נראה שהמטרה שציינת לילד והפעולה שלך לא נשמעות מתואמות — "
                    "אולי צריך לדייק מה הילד באמת ניסה להשיג.\n\n"
                    "לכל פעולה שילד עושה יש מטרה או צורך שהוא מנסה להשיג. "
                    "כדי לשנות את ההתנהגות, צריך לוודא שהילד יכול להגיע לאותה מטרה רק דרך התנהגות אחרת.\n\n"
                    f"{alternatives}\n\n"
                    f"---\n\n**שאלה {STEP_Q2} מתוך {NUM_QUESTIONS}:**\n\n{q2_text}"
                )
            elif q_id == 4 and evaluation.get("q4_outcome_unclear"):
                # Vague Q4 — sharpen outcome relative to goal; do not restart at Q2
                st.session_state.attempt_counts[q_id] = attempt_count + 1
                with st.spinner("מכין הנחיה..."):
                    bot_reply = generate_guidance_q4(
                        client,
                        protocol_entry["question"],
                        feedback,
                        child_goal=st.session_state.protocol_answers.get(2, ""),
                        q3_answer=st.session_state.protocol_answers.get(3, ""),
                        parent_answer=user_input,
                    )
            else:
                # Increment attempt count for next try
                st.session_state.attempt_counts[q_id] = attempt_count + 1

                if alignment == "invalid_format":
                    # Q5 hard rule: echo the fixed feedback directly, no LLM guidance call
                    bot_reply = feedback
                else:
                    with st.spinner("מכין הנחיה..."):
                        # Q2: pass event + Q1 so guidance can suggest goal examples
                        # Q3: pass event + Q1 + Q2 so guidance links to the child's goal
                        if q_id == 2:
                            bot_reply = generate_guidance(
                                client,
                                protocol_entry["question"],
                                feedback,
                                event_context=st.session_state.event_description,
                                q1_answer=st.session_state.protocol_answers.get(1, ""),
                            )
                        elif q_id == 3:
                            bot_reply = generate_guidance(
                                client,
                                protocol_entry["question"],
                                feedback,
                                event_context=st.session_state.event_description,
                                q1_answer=st.session_state.protocol_answers.get(1, ""),
                                child_goal=st.session_state.protocol_answers.get(2, ""),
                                parent_answer=user_input,
                            )
                        else:
                            bot_reply = generate_guidance(
                                client,
                                protocol_entry["question"],
                                feedback,
                            )

        _add_message("assistant", bot_reply)
        with st.chat_message("assistant"):
            st.markdown(bot_reply)

if __name__ == "__main__":
    main()

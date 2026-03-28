"""
Answer validation module for the parent coaching chatbot.

This file belongs to the parent_coach_bot module and is responsible for all
validation logic applied to parent answers during the protocol flow.

Validation runs in three layers:
1. Hard rules — deterministic checks applied before the LLM (Q1 word ban, Q4 format).
2. LLM evaluation — chain-of-thought reasoning hidden inside the model; the model
   returns a structured JSON verdict with alignment, confidence, and feedback.
3. should_accept override — post-LLM logic that decides whether to accept the answer
   based on alignment + confidence, and applies the soft-pass mechanism after 2 attempts.

Update this file when validation rules, acceptance policy, or the LLM prompt change.
"""

import json
import logging
from openai import OpenAI

from llm_client import get_deployment, get_temperature, get_max_tokens

logger = logging.getLogger(__name__)

# Q4 only accepts one of these exact strings (stripped)
_Q4_VALID = {"כן", "לא"}

# Confidence threshold above which a not_aligned verdict causes a hard reject
_CONFIDENCE_REJECT_THRESHOLD = 0.8

# Number of failed attempts after which the answer is force-accepted (soft pass)
_SOFT_PASS_AFTER = 2

_EVAL_SYSTEM_PROMPT = (
    "You are a precise evaluator of parenting protocol responses. "
    "Carefully analyze the parent's answer internally step by step before deciding. "
    "Be permissive — avoid over-rejecting reasonable or interpretable answers. "
    "Do not expose your internal reasoning. "
    "Respond with valid JSON only, no extra text."
)


def _call_llm(client: OpenAI, messages: list[dict]) -> str:
    """
    Send messages to the LLM and return the raw response string.

    Args:
        client: Authenticated OpenAI client.
        messages: List of {"role": ..., "content": ...} dicts.

    Returns:
        Raw response text from the model.
    """
    response = client.chat.completions.create(
        model=get_deployment(),
        messages=messages,
        temperature=get_temperature(),
        max_tokens=get_max_tokens(),
    )
    return response.choices[0].message.content.strip()


def _q4_affect_only_without_goal_link(answer: str) -> bool:
    """
    Detect Q4 answers that only describe ongoing emotion/behavior, not goal outcome.

    Used as a safety net when the LLM still accepts vague answers like
    "הוא המשיך לבכות" for a goal such as "help me with friends".

    Args:
        answer: Parent's raw Q4 answer.

    Returns:
        True if the text is short, matches affect-only patterns, and lacks goal hints.
    """
    a = answer.strip()
    if len(a) > 72:
        return False
    affect_markers = (
        "המשיך לבכות",
        "המשיכה לבכות",
        "ממשיך לבכות",
        "המשיך לצעוק",
        "המשיך להתעצבן",
        "נשאר לבכות",
    )
    if not any(m in a for m in affect_markers):
        return False
    goal_hints = (
        "עזר",
        "מטרה",
        "חבר",
        "השיג",
        "לא עזר",
        "קיבל",
        "מול",
        "התערב",
        "סירב",
    )
    if any(h in a for h in goal_hints):
        return False
    return True


def _parse_json(raw: str) -> dict:
    """
    Parse a JSON string returned by the LLM, stripping markdown fences if present.

    Args:
        raw: Raw string from the LLM.

    Returns:
        Parsed dict.
    """
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _compute_should_accept(alignment: str, confidence: float, attempt_count: int) -> bool:
    """
    Decide whether to accept the parent's answer based on alignment + confidence.

    Acceptance policy:
    - aligned → always accept
    - partially_aligned → accept (close enough)
    - not_aligned + confidence < threshold → accept (benefit of the doubt)
    - not_aligned + confidence >= threshold → reject (model is confident it's wrong)
    - Soft pass override: accept regardless after attempt_count >= _SOFT_PASS_AFTER

    Args:
        alignment: One of "aligned", "partially_aligned", "not_aligned".
        confidence: Float 0.0–1.0 from the LLM.
        attempt_count: How many times the parent has already tried this question.

    Returns:
        True if the answer should be accepted and the flow should advance.
    """
    if attempt_count >= _SOFT_PASS_AFTER:
        logger.info("Soft pass triggered after %d attempts.", attempt_count)
        return True

    if alignment in ("aligned", "partially_aligned"):
        return True

    # not_aligned: only hard-reject when model is highly confident
    return confidence < _CONFIDENCE_REJECT_THRESHOLD


def validate_answer(
    client: OpenAI,
    question_id: int,
    question: str,
    guidelines: str,
    answer: str,
    attempt_count: int,
    event_context: str = "",
) -> dict:
    """
    Validate a parent's answer against the protocol guidelines.

    Applies hard rules first, then calls the LLM for evaluation, then
    computes the final should_accept decision.

    Args:
        client: Authenticated OpenAI client.
        question_id: Protocol question id (1–4).
        question: Hebrew question text shown to the parent.
        guidelines: English alignment guidelines for this question.
        answer: The parent's raw answer string.
        attempt_count: Number of previous failed attempts on this question
                       (0 on the first try).
        event_context: Situation text for coherence checks; for Q3 includes event,
                       Q1, and Q2 (child's goal) so relevance help/block/irrelevant
                       is judged against the child's goal only.

    Returns:
        Dict with keys:
          - alignment (str): "aligned" | "partially_aligned" | "not_aligned" | "invalid_format"
          - confidence (float): 0.0–1.0 (0.0 for hard-rule results)
          - reasoning_summary (str): brief internal reasoning (not shown to user)
          - feedback (str): Hebrew explanation shown to the parent when not accepted
          - should_accept (bool): whether the flow should advance past this question
          - restart_at_q2 (bool, optional): True only when Q4 fails and Q2/Q3 must be
            re-clarified — app restarts at Q2. False when Q4 only needs sharpening.
          - q4_outcome_unclear (bool, optional): True on Q4 when outcome vs goal is unclear.
    """
    # ── Layer 1: Hard rules ───────────────────────────────────────────────────

    # Q1: child's action must be positive (no negation)
    if question_id == 1 and "לא" in answer.split():
        return {
            "alignment": "not_aligned",
            "confidence": 1.0,
            "reasoning_summary": "Hard rule: answer contains the word 'לא'.",
            "feedback": "התשובה צריכה לתאר מה הילד כן עשה, ולא מה הוא לא עשה.",
            "should_accept": False,
        }

    # Q3: parent's reaction must also be a positive action (no negation)
    if question_id == 3 and "לא" in answer.split():
        return {
            "alignment": "not_aligned",
            "confidence": 1.0,
            "reasoning_summary": "Hard rule: parent's reaction contains the word 'לא'.",
            "feedback": "נסה/י לתאר מה עשית בפועל, ולא מה לא עשית.",
            "should_accept": False,
        }

    # Q5: reinforcement question accepts only כן or לא
    if question_id == 5 and answer.strip() not in _Q4_VALID:
        return {
            "alignment": "invalid_format",
            "confidence": 1.0,
            "reasoning_summary": "Hard rule: Q5 requires exactly כן or לא.",
            "feedback": "אנא ענה/י רק כן או לא.",
            "should_accept": False,
        }

    # ── Layer 2: LLM evaluation ───────────────────────────────────────────────

    context_block = (
        f"Situation context (use this to judge coherence):\n{event_context}\n\n"
        if event_context else ""
    )
    coherence_instruction = (
        "IMPORTANT: First check coherence — does the answer make sense given the "
        "situation context above? If not, mark not_aligned regardless of format. "
        "Only check format/type after coherence passes.\n\n"
        if event_context and question_id not in (3, 4) else ""
    )
    q3_instruction = ""
    if question_id == 3:
        q3_instruction = (
            "Q3 — CHILD'S GOAL ONLY (read carefully):\n"
            "The goal named in context is the CHILD'S goal as the parent stated in Q2. "
            "It is NOT the parent's separate goal or value (e.g. independence vs. help). "
            "Parent and child goals may conflict — do not judge morality or pedagogy.\n"
            "ATTENTION GOALS: If Q2 is about attention/connection (e.g. תשומת לב, שתשים לב, "
            "שתגיב לו), then parental actions that ENGAGE the child — including yelling back, "
            "arguing, scolding — are aligned or partially_aligned as ON-THREAD, because they "
            "still give attention and a response. Do NOT reject 'I screamed back' as irrelevant "
            "when the child's goal was attention. Irrelevant would be ignoring / no response / "
            "complete withdrawal while the goal was attention.\n"
            "OTHER GOALS: The parent's action must be on the SAME THREAD as that goal. "
            "Generic yelling to be left alone with no link to the stated goal is often not_aligned.\n"
            "  • aligned/partially_aligned if the action helped toward the goal OR refused/blocked "
            "it in a way that addresses that goal, OR (for attention goals) engaged the child;\n"
            "  • not_aligned if irrelevant to the goal thread or only withdrawal/ignore when the "
            "goal was not attention.\n"
            "Also check plausibility against the situation context.\n\n"
        )
    q4_instruction = ""
    if question_id == 4:
        q4_instruction = (
            "Q4 — OUTCOME RELATIVE TO THE CHILD'S GOAL:\n"
            "Context includes the child's goal (Q2) and the parent's stated action (Q3).\n"
            "The parent must describe whether THAT goal (Q2) was achieved or not — not only "
            "the child's mood.\n"
            "Set q4_outcome_unclear=true and not_aligned when the answer does not state whether "
            "the Q2 goal was achieved — e.g. only affect ('המשיך לבכות') without goal wording.\n"
            "For ATTENTION goals: 'kept crying' may imply continued attention — still unclear "
            "until phrased in goal terms; set q4_restart_at_q2=FALSE (sharpen only).\n"
            "Set q4_restart_at_q2=TRUE only when Q3 cannot be mapped to Q2 at all — need new "
            "goal at Q2. Otherwise q4_restart_at_q2=FALSE.\n\n"
        )
    json_block = (
        "{\n"
        '  "alignment": "aligned | partially_aligned | not_aligned",\n'
        '  "confidence": <float 0.0-1.0>,\n'
        '  "reasoning_summary": "<one sentence internal summary>",\n'
        '  "feedback": "<short Hebrew explanation for the parent if not aligned>",\n'
        '  "should_accept": <true|false>\n'
        "}"
    )
    if question_id == 4:
        json_block = (
            "{\n"
            '  "alignment": "aligned | partially_aligned | not_aligned",\n'
            '  "confidence": <float 0.0-1.0>,\n'
            '  "reasoning_summary": "<one sentence internal summary>",\n'
            '  "feedback": "<short Hebrew explanation for the parent if not aligned>",\n'
            '  "should_accept": <true|false>,\n'
            '  "q4_outcome_unclear": <true|false>,\n'
            '  "q4_restart_at_q2": <true|false>\n'
            "}\n"
            "q4_outcome_unclear: true if you cannot judge goal achievement from the answer.\n"
            "q4_restart_at_q2: true ONLY if Q2/Q3 are fundamentally misaligned and protocol "
            "must restart at Q2. FALSE if the parent only needs to phrase outcome in goal terms "
            "(e.g. attention goal + vague 'kept crying' — sharpen Q4, do NOT restart)."
        )
    prompt = (
        f"{context_block}"
        f"{q3_instruction}"
        f"{q4_instruction}"
        f"{coherence_instruction}"
        f"Protocol question:\n{question}\n\n"
        f"Alignment guidelines:\n{guidelines}\n\n"
        f"Parent's answer:\n{answer}\n\n"
        "Evaluate the answer. Think step by step internally, then respond ONLY with "
        f"this JSON (no extra text):\n{json_block}"
    )
    messages = [
        {"role": "system", "content": _EVAL_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    raw = _call_llm(client, messages)
    result = _parse_json(raw)

    if question_id == 4 and _q4_affect_only_without_goal_link(answer):
        result["q4_outcome_unclear"] = True
        result["q4_restart_at_q2"] = False

    # ── Layer 3: Override should_accept with our acceptance policy ────────────

    alignment = result.get("alignment", "not_aligned")
    confidence = float(result.get("confidence", 0.5))

    if question_id == 4 and result.get("q4_outcome_unclear") is True:
        result["alignment"] = "not_aligned"
        restart = result.get("q4_restart_at_q2") is True
        result["restart_at_q2"] = restart
        if restart:
            # Q2/Q3 thread broken — full restart; no soft-pass
            result["should_accept"] = False
        else:
            # Vague Q4 — sharpen only; do not use confidence-based accept (would pass junk)
            result["should_accept"] = attempt_count >= _SOFT_PASS_AFTER
    else:
        result["restart_at_q2"] = False
        result["should_accept"] = _compute_should_accept(alignment, confidence, attempt_count)

    return result

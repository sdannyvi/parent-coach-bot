"""
Protocol definition for the parent coaching chatbot.

This file belongs to the parent_coach_bot module and defines the structured
five-question protocol used to guide parents through a behavioral analysis of
an interaction with their child.

Questions:
  1. What negative action did the child do?
  2. What was the child's goal?
  3. How did the parent react / what did the parent do?
  4. What was the outcome relative to the child's goal?
  5. Did the parent reinforce the negative pattern?

Update this file when the coaching protocol changes, questions are revised,
or validation guidelines are adjusted by domain experts.
"""

# Each step defines the question id (1-5), the Hebrew question shown to the parent,
# and the English guidelines used by the LLM validator to evaluate alignment.
PROTOCOL: list[dict] = [
    {
        "id": 1,
        "question": "מה הפעולה שהילד עשה שהכי הוציאה אותך משלוותך?",
        "guidelines": """
The answer must describe a concrete action performed by the child.

The word "לא" (no/not) must NOT appear anywhere in the answer.

Bad example (negation — describes what the child did NOT do):
"לא הלך להתקלח"

Good examples (positive action — describes what the child DID do):
"התעלם ממני והמשיך לשחק במחשב"
"צעק עליי"
"זרק את הצעצוע"

If the answer contains the word "לא", it is automatically not_aligned.
""",
    },
    {
        "id": 2,
        "question": "מה לדעתך הילד ניסה להשיג עם הפעולה הזו?",
        "guidelines": """
STEP 1 — COHERENCE (evaluate this first, no exceptions):
Ask: "Given the situation and the child's specific action, is it genuinely plausible
that the child performed that action in order to achieve THIS goal?"

If the answer does not logically follow from the described event and action → not_aligned.
No format pattern or keyword can override a failed coherence check.

STEP 2 — FORMAT (only if coherence passed):
The answer may be phrased in any of these ways:
- Something the parent should do ("שתתערב", "שאוותר")
- A noun describing what the child wanted ("תמיכה", "עזרה", "ניצחון")
- An outcome the child wanted to reach ("שיוכל להמשיך לשחק")

One format that is NOT acceptable: avoidance framing (what the child wanted to avoid
rather than achieve). Ask the parent to reframe in positive terms.
""",
    },
    {
        "id": 3,
        "question": "איך הגבת? מה עשית באותו רגע?",
        "guidelines": """
The answer must describe a concrete action the PARENT took in response to the child.

The CHILD'S goal is given in context (from Q2 — as the parent described the child's goal).
This is NOT the parent's own goal or parenting ideal. Parent and child goals may conflict;
take no moral or educational stance about who "should" want what.

STEP 1 — RELEVANCE TO THE CHILD'S GOAL ONLY (check this first):
The parent's action must sit on the SAME THREAD as the child's stated goal — the same
need or situation (e.g. if the goal is "help me with my friends", the response must be
about that help request / that situation, not a unrelated parental reaction).

Ask whether the parent's action, in relation to that goal, did one of:
  (A) helped the child achieve or move toward that goal, OR
  (B) clearly refused, blocked, or opposed that goal in a way that engages the situation
      (e.g. "I said I won't come with you to them", "I refused to speak to the friends").

ATTENTION / CONNECTION GOALS (read when Q2 is about תשומת לב, קשר, שיראו אותו, שתגיב לו):
If the child's goal is to get the parent's attention or engagement, then any parental
response that **engages** the child is on-thread — including **yelling back, arguing,
scolding, or a loud response**. That still "addresses" the attention goal (the parent
responded; the child is not ignored). Do NOT mark these as irrelevant sidestep.
Irrelevant here would be **pure ignoring, walking away with no response, refusing to
engage at all** while the stated goal was attention.

NOT sufficient (often not_aligned as irrelevant) — for NON-attention goals:
- Only generic anger, yelling to be left alone, or "stop bothering me" with no link to
  what the child wanted in the stated goal — that does not show help OR a clear
  refusal tied to that goal; it sidesteps the thread.

If the action does NEITHER — it is irrelevant to the child's goal → not_aligned.
Ask the parent to describe a different action of theirs that directly connects to the
child's goal (as help/block on that same thread), not a random parallel behavior.

STEP 2 — POSITIVE ACTION (only if relevance passed):
The word "לא" must NOT appear in the answer.
The answer must describe what the parent DID, not what they did NOT do.

Bad examples:
"לא עשיתי כלום" — negation
"התעלמתי" — may be inaction and/or irrelevant to the child's goal
"gave a snack" when the child's goal was peer intervention — irrelevant to that goal

Good examples (directly tied to the child's stated goal as help or block):
Helping toward the goal, or refusing / blocking it — both are "relevant" if the link is clear.
For an attention goal: responding, even by yelling back, is relevant; ignoring is not.

If the answer contains the word "לא", it is automatically not_aligned.
""",
    },
    {
        "id": 4,
        "question": "מה הייתה התוצאה ביחס למטרה?",
        "guidelines": """
The child's goal is in context (Q2). The parent's action is in context (Q3).

The answer must allow a reader to judge whether the CHILD'S goal was achieved or
not, given what the parent did in Q3. This is not about the parent's goals or values.

STEP 1 — OUTCOME VS. CHILD'S GOAL (must name the thread of Q2):
The answer must make it possible to judge whether THE CHILD'S STATED GOAL IN Q2
was achieved or not (given Q3). Generic child behavior or emotion is NOT enough.

Accept (aligned/partially_aligned) only when the outcome clearly implies that goal was
reached OR not — e.g. "לא עזרתי לו מולם", "הלכנו יחד אליהם", "כן קיבל את העזרה".

STEP 2 — CANNOT TELL FROM THIS ANSWER → q4_outcome_unclear = true:
Set not_aligned when the answer does NOT tie the outcome to whether the Q2 goal was met.
Examples that are TOO VAGUE (do not accept until sharpened):
- Only continuing affect: "הוא המשיך לבכות", "המשיך לצעוק" — without stating in goal terms
  whether the goal was achieved (e.g. for an attention goal: "קיבל תשומת לב", "המשכתי
  להגיב לו" / "לא קיבל" — not only "kept crying").

ATTENTION / CONNECTION GOALS: Ongoing crying or screaming after the parent responded may
**imply** the child still got attention or engagement; the answer is still not_aligned
until the parent phrases the result **relative to the goal** (e.g. "קיבל את תשומת הלב
כי המשכתי לצעוק / להגיב"). Do NOT restart the whole protocol at Q2 for this — only ask
for a sharper answer (see q4_restart_at_q2 below).

STEP 3 — q4_restart_at_q2 (JSON, only when q4_outcome_unclear is true):
Set q4_restart_at_q2 to TRUE only when Q3 and Q2 are so misaligned that the parent must
re-clarify the child's goal at Q2. Set to FALSE when the only issue is vague wording —
the parent should restate Q4 in terms of the goal, without restarting at Q2.

STEP 4:
Reject blank or totally unrelated text.
""",
    },
    {
        "id": 5,
        "question": "האם חיזקתם את דפוס הפעולה השלילי? כן או לא",
        "guidelines": """
The only acceptable answers are "כן" or "לא".

Rule:
If the child's action achieved their goal → the answer should be "כן".
If the child's action did not achieve their goal → the answer should be "לא".
""",
    },
]

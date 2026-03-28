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

The word "לא" (no/not) must NOT appear anywhere in the answer.

The answer must be framed as a positive action (what the parent DID), not as
avoidance (what the parent did NOT do or avoided doing).

Bad examples (negation or avoidance):
"לא עשיתי כלום"
"לא הגבתי"
"התעלמתי" — this describes inaction; ask the parent to describe what they actively did

Good examples (positive action):
"הסכמתי ועזרתי לו"
"הסברתי לו בשקט"
"יצאתי מהחדר"
"ויתרתי לו"
"עמדתי על שלי"
"לקחתי אותו הצידה ודיברתי איתו"

If the answer contains the word "לא", it is automatically not_aligned.
""",
    },
    {
        "id": 4,
        "question": "מה הייתה התוצאה ביחס למטרה?",
        "guidelines": """
The answer should describe whether the child's goal was achieved or not,
in light of how the parent responded.

Be very permissive here. Accept any answer that conveys whether the goal was
reached — even loosely. Do NOT loop or over-coach on this question.

Good examples (accept all of these):
"הוא קיבל מה שרצה"
"היא לא קיבלה מה שרצתה"
"הוא המשיך לשחק" (implies goal achieved)
"היא נכנסה למקלחת" (implies goal not achieved)
"ויתרתי לו"
"כן" / "לא"

Only reject if the answer is completely unrelated to the child's goal
(e.g. describes an unrelated event, or is blank).
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

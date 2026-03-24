"""
Protocol definition for the parent coaching chatbot.

This file belongs to the parent_coach_bot module and defines the structured
four-question protocol used to guide parents through a behavioral analysis of
an interaction with their child.

The protocol focuses on identifying negative child behaviors, understanding the
child's goal, evaluating whether the goal was achieved, and determining whether
the parent inadvertently reinforced the behavior pattern.

Update this file when the coaching protocol changes, questions are revised,
or new alignment guidelines are provided by domain experts.
"""

# Each step defines the question id (1-4), the Hebrew question shown to the parent,
# and the English guidelines the LLM uses to evaluate alignment.
PROTOCOL: list[dict] = [
    {
        "id": 1,
        "question": "כתבו פעולה שלילית שהילד/ה עשה/תה",
        "guidelines": """
The text must describe an action the child performed.

The word "לא" (no/not) must NOT appear anywhere in the answer.

Bad example (describes what the child did NOT do):
"לא הלך להתקלח"

Good example (describes what the child DID do instead):
"התעלם ממני והמשיך לשחק במחשב"

If the answer contains the word "לא", it is automatically not_aligned.
""",
    },
    {
        "id": 2,
        "question": "מה הייתה המטרה או הצורך שהילד ניסה להשיג עם הפעולה הזו?",
        "guidelines": """
The answer must describe what the child wanted to ACHIEVE (a positive goal),
not something the child wanted to AVOID.

Rule 1 — Positive framing required:
The goal must be stated as something the child wanted to gain, reach, or make happen.
Goals framed as avoidance (what the child did NOT want to do) are not aligned.

Bad examples (avoidance framing — not aligned):
"שלא יצטרך לעשות שיעורים"
"שלא יצטרך להתקלח"
"להימנע מהכאב"

Good examples (achievement framing — aligned):
"שיוכל להמשיך לשחק"
"שההורה יוותר לו"
"לקבל עוד זמן מסך"

If the answer uses avoidance framing, return not_aligned and in your Hebrew feedback:
- Acknowledge what the parent wrote.
- Explain that the goal should describe what the child wanted to achieve, not avoid.
- Give a concrete positive reframing as an example.
  For instance: instead of "שלא יצטרך לעשות שיעורים" → "שיוכל לשחק במחשב".

Rule 2 — Describes child's goal, not parent's reaction:
The answer must focus on what the child wanted, not on how the parent felt or reacted.
Answers that describe the parent's feelings or reactions are not aligned.
""",
    },
    {
        "id": 3,
        "question": "מה הייתה התוצאה ביחס למטרה?",
        "guidelines": """
The answer must describe what the PARENT did in response to the child's action, and
whether that response resulted in the child achieving their goal or not.

A good answer contains two parts:
1. What the parent did (their reaction/response).
2. Whether the child's goal was ultimately achieved as a result.

Good examples:
"ויתרתי לו והוא המשיך לשחק" — parent gave in → child achieved goal.
"צעקתי עליו, והוא בכה, אבל לא נכנס למקלחת" — parent reacted, but child still did not comply → goal not clearly achieved.
"התעקשתי והוא בסוף נכנס למקלחת" — parent stood firm → child did not achieve goal.

Bad examples (missing the parent's response or missing the outcome relative to goal):
"הוא המשיך לשחק" — describes only the child, not what the parent did.
"כעסתי" — describes only the parent's feeling, not the outcome.
"הוא קיבל ארטיק קרח" — outcome unrelated to the child's original goal.

If the parent describes only their own emotion without stating what they actually did,
or only the child's final state without linking it to the parent's response, the answer
is not aligned. Both elements — parent's action and outcome relative to goal — must be present.
""",
    },
    {
        "id": 4,
        "question": "האם חיזקתם את דפוס הפעולה השלילי? כן או לא",
        "guidelines": """
The rule:

If the child's action achieved their goal → the answer must be "כן" (yes, the pattern was reinforced).
If the child's action did not achieve their goal → the answer must be "לא" (no, the pattern was not reinforced).

Only the answers "כן" or "לא" are acceptable.
""",
    },
]

"""
Protocol definition for the parent coaching chatbot.

This file belongs to the parent_coach_bot module and defines the structured
four-question protocol used to guide parents through a behavioral analysis of
an interaction with their child.

The protocol focuses on identifying the child's negative action, understanding
the child's goal, evaluating whether the goal was achieved, and determining
whether the parent inadvertently reinforced the behavior pattern.

Update this file when the coaching protocol changes, questions are revised,
or validation guidelines are adjusted by domain experts.
"""

# Each step defines the question id (1-4), the Hebrew question shown to the parent,
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
The answer must describe the goal or need the child was trying to fulfill.

Be permissive. Accept BOTH:
- External outcomes: "to watch TV", "to keep playing", "to get ice cream"
- Internal states: "quiet", "to be left alone", "calm"

The answer should describe what the child wanted, not what the parent felt.

Avoidance framing is NOT aligned (what the child wanted to avoid rather than achieve):
Bad: "שלא יצטרך לעשות שיעורים" → ask for positive reframe, e.g. "שיוכל לשחק"
Bad: "שלא יצטרך להתקלח"

Positive framing IS aligned (what the child wanted to gain or experience):
Good: "שיוכל להמשיך לשחק"
Good: "שקט"
Good: "שיוותרו לו"
Good: "לצפות בטלוויזיה"
Good: "להישאר לבד"

When in doubt, accept the answer.
""",
    },
    {
        "id": 3,
        "question": "מה הייתה התוצאה ביחס למטרה?",
        "guidelines": """
The answer should describe whether the child's goal was achieved or not.

Be very permissive here. Accept any answer that conveys whether the goal was
reached — even loosely. Do NOT loop or over-coach on this question.

Good examples (accept all of these):
"הוא קיבל מה שרצה"
"היא לא קיבלה מה שרצתה"
"הוא המשיך לשחק" (implies goal achieved)
"היא נכנסה למקלחת" (implies goal not achieved)
"ויתרתי לו"
"לא ויתרתי"
"כן" / "לא"

Only reject if the answer is completely unrelated to the child's goal
(e.g. describes an unrelated event, or is blank).
""",
    },
    {
        "id": 4,
        "question": "האם חיזקתם את דפוס הפעולה השלילי? כן או לא",
        "guidelines": """
The only acceptable answers are "כן" or "לא".

Rule:
If the child's action achieved their goal → the answer should be "כן".
If the child's action did not achieve their goal → the answer should be "לא".
""",
    },
]

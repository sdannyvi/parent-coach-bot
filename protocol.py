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
STEP 1 — COHERENCE CHECK (most important, evaluate this first):
The goal must make sense given the specific situation and child action already described.
Ask yourself: "Given what happened, is it plausible that the child did this action
in order to achieve THIS goal?"

If the goal is implausible or clearly unrelated to the described situation → not_aligned.
Do NOT rely on the format examples below to override this coherence check.

Example of coherent answer:
Event: child lost a game and pulled parent's hand toward friends.
Good: "שתתערב לטובתו", "שתגן עליו", "שיקבל עזרה", "תמיכה"
Bad: "לאכול ארטיק", "זמן מסך", "שקט" — these have nothing to do with the situation.

Example of coherent answer:
Event: child ignored parent and kept playing on the computer.
Good: "שיוכל להמשיך לשחק", "זמן מסך", "שאוותר לו"
Bad: "שיאכל ארוחת ערב", "תמיכה רגשית" — unrelated to the described action.

STEP 2 — FORMAT CHECK (only if coherence passed):
Accept any of these formats as long as the content passed the coherence check:
- Actions the parent should perform: "שתתערב", "שאוותר לו", "שאצא מהחדר"
- Nouns: "תמיכה", "שקט", "עזרה", "ניצחון"
- Outcomes the child wanted: "שיוכל להמשיך לשחק", "שיקבל עזרה"

Avoidance framing is NOT aligned:
Bad: "שלא יצטרך לעשות שיעורים" → ask to reframe as "שיוכל לשחק"
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

from pydantic import BaseModel


class AllInOneOutput(BaseModel):
    statements: list[str]


PROMPT = """
You are an Expert User Profiler and Semantic Distiller.
Your primary task is to analyze a raw conversation thread, identify all contributions made by the participant designated as 'SUBJECT', and distill these contributions into a flat list of concise, atomic, self-contained statements representing their Knowledge, Opinion, Attributes, or Intent (KOAI).

Input:
A JSON structure representing the raw conversation thread (list of message objects), where the target participant is identified by the username "SUBJECT".

Output Format:
Generate a single JSON array (a flat list of strings). Each string must represent a single, atomic KOAI statement derived from the SUBJECT's actions, prefixed with the appropriate category tag.

Core Requirements:

1.  SUBJECT Focus: Analyze only the contributions of 'SUBJECT'. Contributions from other participants serve only to provide necessary context for understanding SUBJECT's actions.
2.  Abstract Conversational Format:
    *   The output MUST be a flat list, removing all conversational structure (replies, nesting, turn-taking).
    *   Use Direct Phrasing: State the inferred knowledge, opinion, attribute, or intent directly. Avoid conversational introductions like "States that...", "Expresses...", "Mentions...", "Asks if...", "Responds by...". Focus on the *content* of the inference. Do not add illustrative examples using `(e.g., ...)` or similar phrasing; integrate concepts directly.
3.  Maximum Semantic Resolution & Embedded Context:
    *   Each output statement must be semantically complete and self-contained.
    *   Identify the minimal necessary semantic context (domain, core concepts actually discussed/implied, relevant situation type) from the original conversation required to fully understand the SUBJECT's demonstrated KOAI.
    *   Embed this necessary context directly within the statement string itself through careful phrasing or bracketed additions focused on clarification, not illustration.
    *   The goal is to retain full semantic nuance without needing to refer back to the original chat log structure. Omit only truly ephemeral details (like specific usernames of *others* - refer to them generically if needed, e.g., 'participant', 'advice-seeker').
4.  Atomicity: Each statement in the output list must represent a single, distinct piece of knowledge, opinion, attribute, or intent. If a single action by the SUBJECT demonstrates multiple distinct points (e.g., two different pieces of knowledge), generate a separate statement for each. Do not combine distinct points into one statement using 'and' or similar conjunctions.

KOAI Framework Definitions:

*   KNOWLEDGE: Statements reflecting factual understanding or know-how demonstrated by the SUBJECT that aligns with the LLM's general knowledge base. Must be phrased as knowledge the SUBJECT possesses.
*   OPINION: Statements reflecting the SUBJECT's beliefs, judgments, or preferences, particularly where they might differ from neutral facts or the LLM's baseline perspective. Must be phrased as an opinion held by the SUBJECT.
*   ATTRIBUTE: Descriptions of the SUBJECT's characteristics, possessions, or non-cognitive states derived from their statements (e.g., location, ownership, stated personal traits) that aren't primarily knowledge or opinion.
*   INTENT: [Context-Dependent!] Describes the SUBJECT's immediate purpose or goal for acting *in that specific moment/situation*. This is driven by the immediate context and perception of others. `INTENT` statements must retain necessary situational context (phrased generically) to accurately capture the *why* behind that particular action (e.g., "Wants to achieve X [in situation Y] by doing Z").

Example:

Input Raw Conversation JSON:
[
  {
    "user": "fqn",
    "time": "26-08-2015 13:45",
    "content": "Title: Did I miss the dip now? Or is this the \"dead cat bounce\"?\nSubreddit: stocks\nBody: A transfer just cleared, so now I have some money ready to buy some stocks on Robinhood. Should I hold off for a few weeks or months to see what's going to happen? Or is now a good time to buy?\n\n",
    "replies": [
      {
        "user": "fqn",
        "time": "26-08-2015 13:57",
        "content": "I might be new to this, but this really looks like a dead cat bounce. Maybe I'll check back in a few weeks. My money is probably a lot safer as a cash right now.",
        "replies": [
          {
            "user": "fqn",
            "time": "26-08-2015 14:02",
            "content": "But on the other hand, perhaps I could buy some stocks now, and then sell them just before they start going down again? Sounds pretty risky. But trades are all free on Robinhood.",
            "replies": [
              {
                "user": "SUBJECT",
                "time": "26-08-2015 14:06",
                "content": "Trade commissions don't matter if you're losing money. What is your goal? Are you holding long-term (several years)? Or are you trying to make short-term profits?",
                "replies": []
              }
            ]
          }
        ]
      }
    ]
  }
]

Correct JSON Output (Flat list, atomic, format-agnostic, semantically rich & self-contained):
```json
{
  "statements": [
    "OPINION: Prioritizes avoiding capital loss over minimizing transaction costs in investing/trading, particularly questioning the value of low costs if underlying risk is high.",
    "KNOWLEDGE: Understands investment transaction costs like commissions and the concept of capital loss.",
    "KNOWLEDGE: Recognizes that investment advice must align with the recipient's specific goals, distinguishing between approaches for long-term holding versus short-term profit seeking.",
    "INTENT: Wants to help an advice-seeker clarify their investment strategy by prompting for core objectives [like time horizon], perceiving a potential mismatch between stated means [like focusing on low costs] and unstated goals."
  ]
}
```
---

BEGIN TASK

Input:
"""

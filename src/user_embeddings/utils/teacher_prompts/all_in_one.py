from pydantic import BaseModel


class AllInOneOutput(BaseModel):
    statements: list[str]


PROMPT = """
You are an Expert User Profiler and Semantic Distiller.
Your primary task is to analyze a raw conversation thread, identify all contributions made by the participant designated as 'SUBJECT', and distill these contributions into a flat list of concise, atomic, self-contained statements representing their Knowledge, Opinion, Attributes, or Intent (KOAI).

Input:
A JSON structure representing the raw conversation thread (list of message objects), where the target participant is identified by the username "SUBJECT".

Output Format:
Generate a single JSON object containing a single key "statements" whose value is a flat list of strings. Each string must represent a single, atomic KOAI statement derived from the SUBJECT's actions, prefixed with the appropriate category tag (KNOWLEDGE, OPINION, ATTRIBUTE, INTENT).

Core Requirements:

1.  SUBJECT Focus: Analyze only the contributions of 'SUBJECT'. Contributions from other participants/sources serve *only* to provide necessary context for understanding SUBJECT's actions or the information SUBJECT is engaging with.

2.  Abstract Conversational Format & NO QUOTING:
    *   The output MUST be a flat list, removing all conversational structure (replies, nesting, turn-taking).
    *   Use Direct Phrasing for INFERRED MEANING: State the inferred knowledge, opinion, attribute, or intent directly. Avoid conversational introductions like "States that...", "Expresses...", "Mentions...", "Asks if...", "Responds by...". Focus on the *content* of the inference.
    *   Crucially, DO NOT embed verbatim quotes from the SUBJECT's input within the output statements. Focus entirely on the distilled semantic *meaning* or *implication* of the SUBJECT's contribution, not the specific words used. Rephrase the essence in your own words, maintaining the KOAI category.

3.  Maximum Semantic Resolution, Embedded Informational Context, and GENERALIZATION:
    *   Each output statement must be semantically complete and self-contained.
    *   Identify the minimal necessary semantic context (domain, core concepts, type of information/query being addressed, relevant situation type) required to fully understand the SUBJECT's demonstrated KOAI. This context might come from other parts of the input (e.g., preceding messages, the original post).
    *   Embed this necessary context directly within the statement string itself through careful phrasing or bracketed additions focused on clarification (e.g., `[regarding topic X]`, `[when addressing alternative viewpoints]`, `[in the context of financial planning]`). Critically, phrase this context in terms of the *topic* or *information* being discussed, not the conversational act of replying. For example, instead of "Replying to user X's question about Y", prefer phrasing like "Provides information about Y [in response to a query]" or "Addresses the topic of Y [where differing opinions exist]".
    *   The goal is to retain full semantic nuance without needing to refer back to the original interaction log structure or specific conversational participants.
    *   Generalize Unknown Specifics: Abstract away specific entities or details that are likely unknown outside the immediate context or highly specific, *unless* that detail is essential for the core semantic meaning or the discussion is *about* that specific detail. If SUBJECT mentions 'player John Doe of the Exampletown Eagles', and 'John Doe' is likely an unknown specific but 'Exampletown Eagles' is a known entity or the essential context, the statement should refer to 'a player from the Exampletown Eagles' or generalize appropriately (e.g., 'a specific sports player'). Use generic terms like 'a specific company', 'a particular software' when the specific name adds no generalizable value or is likely unknown.

4.  Atomicity: Each statement in the output list must represent a single, distinct piece of knowledge, opinion, attribute, or intent. If a single action by the SUBJECT demonstrates multiple distinct points (e.g., two different pieces of knowledge), generate a separate statement for each. Do not combine distinct points into one statement using 'and' or similar conjunctions.

KOAI Framework Definitions:

*   KNOWLEDGE: Statements reflecting factual understanding or know-how demonstrated by the SUBJECT that aligns with general knowledge. Must be phrased as knowledge the SUBJECT possesses. (e.g., "KNOWLEDGE: Understands the concept of capital loss in investing.")
*   OPINION: Statements reflecting the SUBJECT's beliefs, judgments, preferences, or subjective interpretations. Must be phrased as an opinion held by the SUBJECT. (e.g., "OPINION: Believes high transaction costs are acceptable if investment quality is high.")
*   ATTRIBUTE: Descriptions of the SUBJECT's characteristics, possessions, roles, or non-cognitive states derived directly from their statements or context. (e.g., "ATTRIBUTE: Identifies as a long-term investor.", "ATTRIBUTE: Uses the Robinhood platform.")
*   INTENT: [Context-Dependent!] Describes the SUBJECT's immediate purpose or goal for acting *in that specific moment/situation*, often related to influencing or interacting with the information space or others implicitly. Must retain necessary situational context (phrased generically, potentially including generalized roles like 'advice-seeker' if interaction is key) to capture the *why*. (e.g., "INTENT: Aims to clarify investment goals [when conflicting information is present] by asking about time horizon.")

Example:

Input Raw Conversation JSON:
```json
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
```

Correct JSON Output (Flat list, atomic, format-agnostic, semantically rich, generalized, NO QUOTES, informational context where appropriate, retains necessary interaction context for INTENT):
```json
{
  "koa": [
    "OPINION: Prioritizes avoiding capital loss over minimizing transaction costs in investing/trading, particularly questioning the value of low costs if underlying risk is high.",
    "KNOWLEDGE: Understands investment transaction costs like commissions and the concept of capital loss.",
    "KNOWLEDGE: Recognizes that investment advice must align with specific goals, distinguishing between approaches for long-term holding versus short-term profit seeking.",
  ],
  "intents":[
    "INTENT: Wants to help a beginner investor clarify their investment strategy by prompting for core objectives like time horizon, perceiving a potential mismatch between stated means (like focusing on low costs) and unstated goals."
  ]
}
```

---

BEGIN TASK

Input:
"""

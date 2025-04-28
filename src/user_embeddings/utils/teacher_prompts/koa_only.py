from typing import List

from pydantic import BaseModel, Field


class PromptOutput(BaseModel):
    """Pydantic model for validating the output of the KOA extraction prompt."""

    koa: List[str] = Field(
        ...,
        description="List of extracted Knowledge, Opinion, or Attribute statements.",
    )


PROMPT = """
You are an Expert User Profiler and Semantic Distiller focusing on Knowledge, Opinions, and Attributes.
Your primary task is to analyze a raw conversation thread, identify contributions made by 'SUBJECT', and distill these contributions into a flat list of concise, atomic, self-contained statements representing ONLY their **Knowledge (K), Opinion (O), or Attributes (A)**. **DO NOT extract Intents.**

Input:
A JSON structure representing the raw conversation thread (list of message objects), where the target participant is identified by the username "SUBJECT".

Output Format:
Generate a single JSON object containing a single key "koa" whose value is a flat list of strings. Each string must represent a single, atomic K, O, or A statement derived from the SUBJECT's actions, prefixed with the appropriate category tag (KNOWLEDGE, OPINION, ATTRIBUTE).

Core Requirements for KOA Extraction:

1.  **SUBJECT Focus:** Analyze only the contributions of 'SUBJECT'. Other context serves only to understand SUBJECT's statements.
2.  **Abstract Format & NO QUOTING:**
    *   Output MUST be a flat list.
    *   Use Direct Phrasing for INFERRED MEANING. Avoid conversational introductions.
    *   **Crucially, DO NOT embed verbatim quotes**. Rephrase the semantic essence.
3.  **Maximum Semantic Resolution, Embedded Informational Context, and GENERALIZATION:**
    *   Statements must be semantically complete and self-contained.
    *   Embed necessary semantic context (domain, core concepts, topic) using careful phrasing or brackets (e.g., `[regarding topic X]`). Focus context on the *topic*, not the conversation.
    *   **Generalize Unknown Specifics:** Abstract away unknown entities/details unless essential (e.g., 'a player from the Exampletown Eagles' instead of 'John Doe'). Use generic terms.
4.  **Atomicity:** Each statement must be a single, distinct K, O, or A point. No 'and'.
5.  **IGNORE INTENT:** Do not generate any statements describing the immediate purpose or goal of the SUBJECT's action in the conversation. Focus solely on the knowledge conveyed, opinions expressed, or attributes revealed.

KOAI Definitions (Focus on K, O, A):

*   **KNOWLEDGE:** Factual understanding/know-how possessed by SUBJECT. (e.g., "KNOWLEDGE: Understands the concept of capital loss in investing.")
*   **OPINION:** SUBJECT's beliefs, judgments, preferences, subjective interpretations. (e.g., "OPINION: Believes high transaction costs are acceptable if investment quality is high.")
*   **ATTRIBUTE:** SUBJECT's characteristics, possessions, roles, non-cognitive states. (e.g., "ATTRIBUTE: Identifies as a long-term investor.")

Example (Input JSON is the same as before):

Correct JSON Output for KOA:
```json
{
  "koa": [
    "OPINION: Prioritizes avoiding capital loss over minimizing transaction costs in investing/trading, particularly questioning the value of low costs if underlying risk is high.",
    "KNOWLEDGE: Understands investment transaction costs like commissions and the concept of capital loss.",
    "KNOWLEDGE: Recognizes that investment advice must align with specific goals, distinguishing between approaches for long-term holding versus short-term profit seeking."
    // NOTE: No INTENT statement included here
  ]
}
```

---

BEGIN TASK

Input:
"""

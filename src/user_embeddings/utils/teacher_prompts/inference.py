from pydantic import BaseModel


class InferenceOutput(BaseModel):
    context: str
    actions: list[list[str]]


PROMPT = """
You are an Expert Information Extractor and Normalizer. Your goal is to analyze a structured summary of a conversation thread, focusing on the participant designated as 'SUBJECT'. Abstract the SUBJECT's specific actions into *direct statements* representing their knowledge, opinions, explicitly mentioned attributes (possessions, experiences, location), and core intent demonstrated *within that specific interaction*. Avoid introductory phrases like "States that..." or "Displays...". Your output MUST retain the exact overall JSON structure of the input, transforming only the content of the 'actions' list.

Input:
A JSON object produced by the 'Interaction Analyzer and Structure Synthesizer'. This object contains:
- "context": An NL summary of the conversation preceding the SUBJECT's actions or interleaved between them.
- "actions": A list of NL strings, where each string describes one of the SUBJECT's specific, atomic contributions.

Output Format Specification:
Generate a single JSON object with the exact same top-level keys as the input ('context', 'actions').
- The value for the "context" key should be copied verbatim from the input "context" value.
- The value for the "actions" key should be a list where each element is itself a list of strings.
- The outer list of "actions" corresponds 1-to-1 with the input "actions" list.
- Each *inner list* contains one or more direct NL statements, representing the inferences abstracted from the *single corresponding action* in the input. Each statement MUST start with a capitalized category prefix followed by a colon.

Example Output Structure:
{
  "context": "The exact same context string as provided in the input.",
  "actions": [
    [ // Inferences derived from the FIRST action string in the input
      "CATEGORY: Direct statement 1 about Knowledge/Opinion/Attribute/Intent.",
      "CATEGORY: Direct statement 2 about Knowledge/Opinion/Attribute/Intent."
    ],
    [ // Inferences derived from the SECOND action string in the input
      "CATEGORY: Direct statement 3 about Knowledge/Opinion/Attribute/Intent."
    ]
    // ... one inner list for each action string in the input actions list ...
  ]
}


Key Principles for Abstraction and Output Generation:

1.  Mirror Top-Level Structure: The output JSON MUST have the keys 'context' and 'actions'.
2.  Copy Context: The input 'context' string MUST be copied directly to the output 'context' field without modification for this task.
3.  Transform Actions to Lists of Direct Statements: Each string in the input 'actions' list maps to an *inner list* of direct inference strings in the output 'actions' list.
4.  Direct Phrasing: State the inferred knowledge, opinion, attribute, or intent directly. Do NOT use introductory phrases like "States that...", "Expresses...", "Displays knowledge...", "Mentions...", "Appears to be...".
5.  Focus Categories (Use CAPITALIZED Prefixes): Generate statements reflecting these aspects, prefixed with the capitalized category name followed by a colon:
    *   `KNOWLEDGE`: (e.g., "KNOWLEDGE: Python dictionaries map keys to values.")
    *   `OPINION`: (e.g., "OPINION: Current stock market volatility is high.")
    *   `ATTRIBUTE`: (e.g., "ATTRIBUTE: Owns an electric vehicle.") *Only if explicitly stated.*
    *   `INTENT`: (e.g., "INTENT: Seek advice.", "INTENT: Provide a solution.")
6.  Generalizable Language: Use platform-agnostic terms. Avoid Reddit-specific jargon.
7.  Interaction-Bound: Base abstractions SOLELY on the text provided for the specific action being processed and its surrounding input context.
8.  Conciseness: Keep inference statements brief and factual based on the input action. One input action might yield one or multiple inference statements.
9.  Output ONLY JSON: Your entire response must be the single, valid JSON object described above, enclosed in ```json ```.

---
EXAMPLE:

Input JSON (from Separation Step):
{
  "context": "A participant ('fqn') initiated a discussion seeking advice on timing stock purchases after a recent market dip, questioning if it's a buying opportunity or a 'dead cat bounce'. This participant subsequently expressed concern it might be a bounce and considered holding cash, but then weighed the possibility of short-term trading enabled by free commissions on their platform.",
  "actions": [
    "SUBJECT asserts that avoiding losses is more critical than trade commission costs.",
    "SUBJECT prompts the initiating participant ('fqn') to define their investment objective, specifically asking whether the goal is long-term holding or seeking short-term profits."
  ]
}

Correct JSON Output (Direct Statements, Capitalized Categories, Actions as Lists):
```json
{
  "context": "A participant ('fqn') initiated a discussion seeking advice on timing stock purchases after a recent market dip, questioning if it's a buying opportunity or a 'dead cat bounce'. This participant subsequently expressed concern it might be a bounce and considered holding cash, but then weighed the possibility of short-term trading enabled by free commissions on their platform.",
  "actions": [
    [
      "OPINION: SUBJECT believes loss avoidance outweighs commission costs in trading.",
      "KNOWLEDGE: SUBJECT understands trade commissions and capital loss concepts.",
    ],
    [
      "KNOWLEDGE: SUBJECT understands difference between long-term and short-term investment objectives.",
      "INTENT: SUBJECT seeks clarification to provide tailored advice."
    ]
  ]
}
```

BEGIN TASK

Input:
"""

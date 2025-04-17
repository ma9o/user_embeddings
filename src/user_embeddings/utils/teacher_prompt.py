import json

TEACHER_PROMPT = """
You are an Expert Interaction Analyzer and Structure Synthesizer. Your goal is to analyze a conversational context, focusing on the contributions of the participant designated as 'SUBJECT', and represent the flow of interaction as a nested JSON structure composed of natural language summaries and actions.

Input:
A JSON structure representing the conversation thread (list of message objects), where the target participant is identified by the username "SUBJECT".

Output Format Specification:
Generate a single JSON object with the following structure:
{
  "context": "Natural language string summarizing the initial context before the SUBJECT's first action relevant to this structure.",
  "actions": [
    // This list contains a mix of strings and nested objects, ordered chronologically.
    "String representing a single, atomic action/contribution by the SUBJECT.",
    // ... more action strings if the SUBJECT made multiple points in the same comment ...
    {
      "context": "Natural language string summarizing the accumulated context from non-SUBJECT participants between the previous SUBJECT action and the next one at this level.",
      "actions": [
        // Recursive structure: contains more action strings and potentially further nested context objects.
        "Another SUBJECT action string.",
        {
            "context": "...",
            "actions": [...]
        }
      ]
    },
    "Yet another SUBJECT action string (if context didn't change)."
  ]
}

Key Principles for Output Generation:

1.  **Focus on SUBJECT:** The structure revolves around the participant identified as "SUBJECT" in the input `user_context`. Their messages are broken down into atomic action strings.
2.  **Context Summarization:** Messages from ALL OTHER participants are summarized into concise, natural language `context` strings. A `context` field captures the essence of conversational turns by non-SUBJECT participants that occur *between* the SUBJECT's actions *at a given nesting level*. The style should be similar to the action strings.
3.  **Structure Follows Flow:** The nesting reflects the conversation's reply structure *only when* non-SUBJECT participant messages (context) interleave with the SUBJECT's messages. If the SUBJECT replies multiple times without intervening context *at that level*, their action strings appear sequentially in the same `actions` list.
4.  **Action Strings:** Each action string describing the SUBJECT's contribution should be:
    *   **Atomic:** Represent a single logical point, assertion, question, or reaction from the SUBJECT's message.
    *   **Factual & Concise:** Describe what the SUBJECT did or expressed impersonally. Use precise verbs (identifies, clarifies, argues, asks, confirms, provides, disputes, etc.). Avoid vague terms like "says."
    *   **Self-Contained (within message):** Capture the essence of the SUBJECT's point *from that specific message*. Does NOT need to repeat context from parent `context` fields.
5.  **Chronological Order:** All elements within any `actions` list MUST be ordered according to the original timestamps of the messages they represent.
6.  **Consistent NL Style:** Both `context` summaries and `action` strings should use a similar, factual, descriptive natural language style suitable for projection into a shared semantic space.

Processing Steps (Conceptual):

1.  **Identify Initial Context:** Determine the content of the initial message(s) and any subsequent messages preceding the *first* message by the participant "SUBJECT". Summarize this into the root `context` string.
2.  **Process Conversation Tree:** Traverse the conversation chronologically.
3.  **Accumulate & Summarize Context:** As you encounter messages from non-SUBJECT participants, mentally accumulate their points. When a "SUBJECT" message is encountered or the branch ends, summarize the accumulated non-SUBJECT points into a `context` string if needed.
4.  **Handle SUBJECT Message:** When you encounter a message where the participant is "SUBJECT":
    *   If accumulated non-SUBJECT context exists for the current level, finalize its summary string and create a nested `{"context": ..., "actions": []}` object. Add this object to the current level's `actions` list. The `actions` list of this *new object* becomes the current target for subsequent actions within this branch.
    *   Generate the atomic action string(s) for the "SUBJECT"'s current message. Append these strings to the *currently active* `actions` list.
    *   Recursively process the replies to the "SUBJECT"'s message, adding results to the *currently active* `actions` list.
5.  **Handle Non-SUBJECT Message Replies:** When processing a non-SUBJECT participant's message, also recursively process its replies, adding results to the *same parent actions list* you were using before processing the non-SUBJECT participant.

---
EXAMPLE:

Input:
```json
[
  {
    "user": "fqn",
    "time": "26-08-2015 13:57",
    "content": "Title: Did I miss the dip now? Or is this the \"dead cat bounce\"?\nBody: A transfer just cleared, so now I have some money ready to buy some stocks. Should I hold off for a few weeks or months to see what's going to happen? Or is now a good time to buy?\n\n",
    "replies": [
      {
        "user": "fqn",
        "time": "26-08-2015 13:57",
        "content": "I might be new to this, but this really looks like a dead cat bounce. Maybe I'll check back in a few weeks. My money is probably a lot safer as a cash right now.",
        "replies": [
          {
            "user": "fqn",
            "time": "26-08-2015 14:02",
            "content": "But on the other hand, perhaps I could buy some stocks now, and then sell them just before they start going down again? Sounds pretty risky. But trades are all free [on the platform].",
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

Correct JSON Output:
```json
{
  "context": "A participant ('fqn') initiated a discussion seeking advice on timing stock purchases after a recent market dip, questioning if it's a buying opportunity or a 'dead cat bounce'. This participant subsequently expressed concern it might be a bounce and considered holding cash, but then weighed the possibility of short-term trading enabled by free commissions on their platform.",
  "actions": [
    "SUBJECT asserts that avoiding losses is more critical than trade commission costs.",
    "SUBJECT prompts the initiating participant ('fqn') to define their investment objective, specifically asking whether the goal is long-term holding or seeking short-term profits."
  ]
}
```

---

BEGIN TASK

Input:
"""


def get_teacher_prompt(user_context_raw: str) -> str:
    user_context = json.loads(user_context_raw)
    return f"{TEACHER_PROMPT}\n{user_context}"


def _extract_last_json(text: str) -> dict | None:
    """
    Finds and parses the last valid top-level JSON object enclosed in {}
    within a string, correctly handling nested structures.

    Args:
        text: The string potentially containing JSON objects and other text.

    Returns:
        The parsed JSON object as a dictionary, or None if no valid JSON object
        is found.
    """
    end_index = len(text) - 1
    while True:
        # Find the last '}' at or before end_index
        end_index = text.rfind("}", 0, end_index + 1)
        if end_index == -1:
            return None  # No more '}' found

        # Try to find the matching '{' for this '}' by tracking brace levels
        start_index = -1
        brace_level = 0
        for i in range(end_index, -1, -1):
            char = text[i]
            if char == "}":
                brace_level += 1
            elif char == "{":
                brace_level -= 1
                if brace_level == 0:
                    start_index = i
                    break  # Found the matching '{'

        if start_index != -1:
            # Extract the potential JSON substring
            potential_json_str = text[start_index : end_index + 1]
            try:
                # Attempt to parse the substring as JSON
                # Consider using repair_json if slightly malformed JSON is expected
                parsed_json = json.loads(potential_json_str)
                # Ensure it's a dictionary (object), not just an array or primitive
                if isinstance(parsed_json, dict):
                    return parsed_json  # Return the first valid JSON object found from the end
            except json.JSONDecodeError:
                # If parsing fails, this segment wasn't a valid JSON object.
                # Continue searching from before this '}' in the next iteration.
                pass

        # Move search backward: look for a '}' before the one we just processed
        end_index -= 1
        if end_index < 0:
            return None  # Reached beginning of string without finding valid JSON


def parse_teacher_prompt_output(output: str) -> dict:
    # Find the last occurrence of a json object in the output delimited by {}, starting from the bottom of the output
    parsed_data = _extract_last_json(output)
    if parsed_data is None:
        # Consider how to handle cases where no JSON is found.
        # Raise an error, return an empty dict, or handle differently.
        raise ValueError("No valid JSON object found in the output.")
    return parsed_data

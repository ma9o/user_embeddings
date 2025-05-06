from typing import Dict, Optional

# Import utility shared or needed by these funcs
from user_embeddings.utils.parsing import parse_llm_json_output


def create_judge_prompt(
    constraints_prompt: str,  # Specific prompt detailing constraints
    input_data: str,
    model_output: str,
) -> str:
    """Creates a prompt for a judge model to identify constraint violations using f-strings."""
    prompt = f"""You are an expert evaluator tasked with identifying violations of specific constraints in a Large Language Model (LLM) output based on a given input, and a set of constraints.

INPUT DATA GIVEN TO THE MODEL:
---
{input_data}
---

MODEL OUTPUT TO EVALUATE:
---
{model_output}
---

CONSTRAINTS TO CHECK:
---
{constraints_prompt}
---

TASK:
1. Carefully review the MODEL OUTPUT.
2. Compare it against the CONSTRAINTS TO CHECK, considering the INPUT DATA.
3. Identify *all* constraints that the MODEL OUTPUT failed to meet.

OUTPUT FORMAT:
Provide your evaluation as a JSON object where each key is a unique identifier string for the violated constraint and the value is a brief string explaining the violation.
The key MUST follow the format `CATEGORY.MainSection.SubSection` (e.g., `OUTPUT_FORMATTING.2.1`, `SEMANTIC_DISTILLATION.3.4`), referencing the corresponding section and subsection numbers from the 'CONSTRAINTS TO CHECK'. Use the ALL_CAPS category name and at most two numerical parts (e.g., `OUTPUT_FORMATTING.2` or `OUTPUT_FORMATTING.2.3` are valid, but `OUTPUT_FORMATTING.2.3.1` is NOT).
If no constraints were violated, return an empty JSON object (`{{}}`).

Example (Constraints violated):
```json
{{
  "OUTPUT_FORMATTING.2.3": "Explain in detail where the violation happened.",
  "ATOMICITY.4.1": "Explain in detail where the violation happened.",
  "SEMANTIC_DISTILLATION.3.4.2": "Explain in detail where the violation happened."
}}
```

Example (No constraints violated):
```json
{{}}
```

Return ONLY the JSON object and nothing else.
"""
    return prompt


def parse_judge_output(judge_response: str) -> Optional[Dict[str, str]]:
    """Parses the constraint judge's dictionary response using the utility function."""
    parsed_json = parse_llm_json_output(judge_response, expect_type=dict)

    if parsed_json is None:
        print(f"Error parsing constraint judge output. Raw output:\n{judge_response}")
        return None

    if not isinstance(parsed_json, dict):
        print(f"Warning: Constraint judge output is not a dictionary: {parsed_json}")
        return None

    violations_dict: Dict[str, str] = {}
    valid = True
    for key, value in parsed_json.items():
        if not isinstance(key, str) or not isinstance(value, str):
            print(
                f"Warning: Constraint judge dictionary contains non-string key or value: ({type(key)}) {{key}}: ({type(value)}) {{value}}"
            )
            valid = False
            break
        violations_dict[key] = value

    if not valid:
        return None

    return violations_dict

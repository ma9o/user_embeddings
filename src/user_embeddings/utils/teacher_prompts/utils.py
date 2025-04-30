import json
from pathlib import Path

PROMPT_DIR = Path(__file__).parent.parent.parent.parent.parent / "prompts"


def load_prompt(prompt_name: str, version: str = "latest") -> tuple[str, str]:
    if version != "latest":
        prompt_path = PROMPT_DIR / prompt_name / f"{version}.txt"
    else:
        # Sort files alphabetically and take the latest
        # Fix: Use .glob() to get an iterable list of paths
        prompt_files = list((PROMPT_DIR / prompt_name).glob("*.txt"))
        if not prompt_files:
            raise FileNotFoundError(
                f"No prompt files found in {PROMPT_DIR / prompt_name}"
            )
        prompt_path = sorted(prompt_files)[-1]

    version_str = prompt_path.stem.split(".")[0]

    print(f"Loading prompt {prompt_name} from {prompt_path}")

    with open(prompt_path, "r") as f:
        return f.read(), version_str


def get_prompt(prompt: str, user_context_raw: str) -> str:
    try:
        json.loads(user_context_raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in user_context_raw: {e}") from e
    return f"{prompt}\n{user_context_raw}\n```"

import importlib.resources
import json
import logging
from typing import Tuple

# PROMPT_DIR = Path(__file__).parent.parent.parent.parent.parent / "prompts" # Removed

logger = logging.getLogger(__name__)


def load_prompt(prompt_name: str, version: str = "latest") -> Tuple[str, str]:
    prompt_package_path = importlib.resources.files("user_embeddings").joinpath(
        "prompts", prompt_name
    )

    if version != "latest":
        prompt_file_path = prompt_package_path.joinpath(f"{version}.txt")
        if not prompt_file_path.is_file():
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_name}/{version}.txt"
            )
    else:
        prompt_files = [
            p
            for p in prompt_package_path.iterdir()
            if p.is_file() and p.name.endswith(".txt")
        ]
        if not prompt_files:
            raise FileNotFoundError(
                f"No prompt files found in user_embeddings.prompts.{prompt_name}"
            )
        # Sort by name to get the latest version (assuming version is in filename e.g., v1.txt, v2.txt or 1.0.0.txt)
        prompt_file_path = sorted(prompt_files, key=lambda p: p.name)[-1]

    version_str = (
        prompt_file_path.stem
    )  # .split(".")[0] is not needed if stem is just the version

    logger.info(f"Loading prompt {prompt_name} (version: {version_str})")

    return prompt_file_path.read_text(encoding="utf-8"), version_str


def get_prompt(prompt: str, user_context_raw: str) -> str:
    try:
        json.loads(user_context_raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in user_context_raw: {e}") from e
    return f"{prompt}\n{user_context_raw}\n```"

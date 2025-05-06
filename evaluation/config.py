from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import BaseModel

# Import workflow utilities - needed for WORKFLOWS type hint
from user_embeddings.utils.llm.workflow_executor import WorkflowStage
from user_embeddings.utils.teacher_prompts import (
    all_in_one as all_in_one_module,
)

# Import teacher prompts - needed for AVAILABLE_PROMPTS and WORKFLOWS
from user_embeddings.utils.teacher_prompts import (
    inference as inference_module,
)
from user_embeddings.utils.teacher_prompts import (
    intent_only as intent_only_module,
)
from user_embeddings.utils.teacher_prompts import (
    koa_only as koa_only_module,
)

# --- Shared Prompt Mapping --- (Now includes version)
# Used by both llm_rank_benchmark and prompt_adherence
# Stores tuple: (prompt_text, prompt_version)
AVAILABLE_PROMPTS: Dict[str, Tuple[str, str]] = {
    "inference": (inference_module.PROMPT, inference_module.VERSION),
    # "separation": (separation_module.PROMPT, separation_module.VERSION),
    "intent_only": (intent_only_module.PROMPT, intent_only_module.VERSION),
    "koa_only": (koa_only_module.PROMPT, koa_only_module.VERSION),
    "all_in_one": (all_in_one_module.PROMPT, all_in_one_module.VERSION),
    # Add any NEW prompts here, ensuring they export PROMPT and VERSION
    # e.g., "constraint_checker_v1": (constraint_checker_v1.PROMPT, constraint_checker_v1.VERSION),
}

# --- Shared Pydantic Output Model Mapping ---
# Used by run_and_parse_test_models in both scripts
AVAILABLE_OUTPUT_MODELS: Dict[str, type[BaseModel]] = {
    "koa_only": koa_only_module.PromptOutput,
    "intent_only": intent_only_module.PromptOutput,
    "inference": inference_module.PromptOutput,
    "all_in_one": all_in_one_module.PromptOutput,
}

# --- Shared Input Formatter Mapping ---
# Used by workflow_executor to format inputs for specific tasks
# Maps task_id (prompt name) -> Callable[[Dict[str, Any]], str]
# Define your formatter functions (e.g., in the prompt modules) and import them here.
AVAILABLE_INPUT_FORMATTERS: Dict[str, callable] = {
    # Example:
    # "inference": inference_module.format_inference_input,
    "intent_only": intent_only_module.format_intent_only_input,
}


# --- Shared Workflow Definitions ---
# Used by both llm_rank_benchmark and prompt_adherence
WORKFLOWS: Dict[str, List[WorkflowStage]] = {
    "serial_separation_inference": [
        {
            "stage": 1,
            "tasks": [{"prompt": "separation", "input_from": ["__RAW_INPUT__"]}],
        },
        {"stage": 2, "tasks": [{"prompt": "inference", "input_from": ["separation"]}]},
    ],
    "concurrent_intent_koa": [
        {
            "stage": 1,
            "tasks": [
                {"prompt": "intent_only", "input_from": ["__RAW_INPUT__"]},
                {"prompt": "koa_only", "input_from": ["__RAW_INPUT__"]},
            ],
        },
    ],
    "single_all_in_one": [
        {
            "stage": 1,
            "tasks": [{"prompt": "all_in_one", "input_from": ["__RAW_INPUT__"]}],
        }
    ],
    "single_intent_only": [
        {
            "stage": 1,
            "tasks": [{"prompt": "intent_only", "input_from": ["__RAW_INPUT__"]}],
        }
    ],
    "inference_only": [
        {
            "stage": 1,
            "tasks": [{"prompt": "inference", "input_from": ["__RAW_INPUT__"]}],
        }
    ],
    "koa_only": [
        {"stage": 1, "tasks": [{"prompt": "koa_only", "input_from": ["__RAW_INPUT__"]}]}
    ],
    "inference_with_intent": [
        {
            "stage": 1,
            "tasks": [{"prompt": "inference", "input_from": ["__RAW_INPUT__"]}],
        },
        {
            "stage": 2,
            "tasks": [
                {"prompt": "intent_only", "input_from": ["inference", "__RAW_INPUT__"]}
            ],
        },
    ],
}


# --- Common Default Values ---
DEFAULT_JUDGE_MODEL = "google/gemini-2.5-pro-preview-03-25"
DEFAULT_NUM_SAMPLES = 10
DEFAULT_SEED = None
DEFAULT_BASE_DATA_DIR = Path(
    "./data"
)  # Base data dir for constructing specific output dirs

from pathlib import Path
from typing import Dict, List, Tuple

# Import workflow utilities - needed for WORKFLOWS type hint
from user_embeddings.utils.llm.workflow_executor import PromptStage

# Import teacher prompts - needed for AVAILABLE_PROMPTS and WORKFLOWS
from user_embeddings.utils.teacher_prompts import (
    all_in_one as all_in_one_module,
)
from user_embeddings.utils.teacher_prompts import (
    inference as inference_module,
)
from user_embeddings.utils.teacher_prompts import (
    intent_only as intent_only_module,
)
from user_embeddings.utils.teacher_prompts import (
    koa_only as koa_only_module,
)

# Import Pydantic models for AVAILABLE_OUTPUT_MODELS
# Moved these imports up
# from user_embeddings.utils.teacher_prompts import intent_only as intent_only_module
# from user_embeddings.utils.teacher_prompts import koa_only as koa_only_module
# from user_embeddings.utils.teacher_prompts.deprecated import inference, separation

# --- Shared Prompt Mapping --- (Now includes version)
# Used by both llm_rank_benchmark and prompt_adherence
# Stores tuple: (prompt_text, prompt_version)
AVAILABLE_PROMPTS: Dict[str, Tuple[str, str]] = {
    "all_in_one": (all_in_one_module.PROMPT, all_in_one_module.VERSION),
    "inference": (inference_module.PROMPT, inference_module.VERSION),
    # "separation": (separation_module.PROMPT, separation_module.VERSION),
    "intent_only": (intent_only_module.PROMPT, intent_only_module.VERSION),
    "koa_only": (koa_only_module.PROMPT, koa_only_module.VERSION),
    # Add any NEW prompts here, ensuring they export PROMPT and VERSION
    # e.g., "constraint_checker_v1": (constraint_checker_v1.PROMPT, constraint_checker_v1.VERSION),
}

# --- Shared Pydantic Output Model Mapping ---
# Used by run_and_parse_test_models in both scripts
AVAILABLE_OUTPUT_MODELS: Dict[str, type] = {
    "koa_only": koa_only_module.PromptOutput,
    "intent_only": intent_only_module.PromptOutput,
    # Add other models here if they are defined and used in workflows
    # "separation": separation_module.PromptOutput,
    # " inference": inference_module.PromptOutput,
    # "all_in_one": all_in_one_module.PromptOutput,
}

# --- Shared Workflow Definitions ---
# Used by both llm_rank_benchmark and prompt_adherence
WORKFLOWS: Dict[str, List[PromptStage]] = {
    "serial_separation_inference": [
        {
            "stage": 1,
            "prompts": ["separation"],
            "input_from": None,
            "input_formatter": None,
        },
        {
            "stage": 2,
            "prompts": ["inference"],
            "input_from": ["separation"],
            "input_formatter": "format_single_input",
        },
    ],
    "concurrent_intent_koa": [
        {
            "stage": 1,
            "prompts": ["intent_only", "koa_only"],
            "input_from": None,
            "input_formatter": None,
        },
    ],
    "single_all_in_one": [
        {
            "stage": 1,
            "prompts": ["all_in_one"],
            "input_from": None,
            "input_formatter": None,
        }
    ],
    "single_intent_only": [
        {
            "stage": 1,
            "prompts": ["intent_only"],
            "input_from": None,
            "input_formatter": None,
        }
    ],
    "inference_only": [
        {
            "stage": 1,
            "prompts": ["inference"],
            "input_from": None,
            "input_formatter": None,
        }
    ],
    # Add other common workflows here
}


# --- Common Default Values ---
DEFAULT_JUDGE_MODEL = "google/gemini-2.5-pro-preview-03-25"
DEFAULT_NUM_SAMPLES = 10
DEFAULT_SEED = None
DEFAULT_BASE_DATA_DIR = Path(
    "./data"
)  # Base data dir for constructing specific output dirs

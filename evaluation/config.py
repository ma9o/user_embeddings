from pathlib import Path
from typing import Dict, List, Tuple, Type

from pydantic import BaseModel  # Needed for AVAILABLE_OUTPUT_MODELS type hint

# Import workflow utilities - needed for WORKFLOWS type hint
from user_embeddings.utils.llm.workflow_executor import (
    WorkflowStage,  # Use the new type
)

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
    "intent_only": (intent_only_module.PROMPT, intent_only_module.VERSION),
    "koa_only": (koa_only_module.PROMPT, koa_only_module.VERSION),
}

# --- Shared Pydantic Output Model Mapping ---
# Used by the workflow executor. Key MUST match the task_id in WORKFLOWS.
AVAILABLE_OUTPUT_MODELS: Dict[str, Type[BaseModel]] = {
    # Task IDs need to be unique across all workflows if they share models
    # Example task IDs used below:
    "extract_intent": intent_only_module.PromptOutput,
    "extract_koa": koa_only_module.PromptOutput,
    # "separate_context": separation_module.PromptOutput, # If separation prompt/model existed
    # "infer_from_separated": inference_module.PromptOutput, # If inference prompt/model existed
    # "process_all_in_one": all_in_one_module.PromptOutput, # If all_in_one model existed
}

# --- Shared Workflow Definitions --- (Updated Structure)
# Uses WorkflowStage and TaskDefinition structure
WORKFLOWS: Dict[str, List[WorkflowStage]] = {
    "concurrent_intent_koa": [
        {
            "stage": 1,
            "tasks": [
                {
                    "task_id": "extract_intent",
                    "prompt": "intent_only",
                    "input_from": [],
                },
                {
                    "task_id": "extract_koa",
                    "prompt": "koa_only",
                    "input_from": [],
                },
            ],
        },
        # Example: Add a hypothetical stage 2 that merges results
        # {
        #     "stage": 2,
        #     "tasks": [
        #         {
        #             "task_id": "merge_intent_koa",
        #             "prompt": "merge_results_prompt", # Assume this prompt exists
        #             "input_from": ["extract_intent", "extract_koa"]
        #         }
        #     ]
        # }
    ],
    "single_all_in_one": [
        {
            "stage": 1,
            "tasks": [
                {
                    "task_id": "process_all_in_one",  # Assign a task ID
                    "prompt": "all_in_one",
                    "input_from": [],
                }
            ],
        }
    ],
    "single_intent_only": [
        {
            "stage": 1,
            "tasks": [
                {
                    "task_id": "extract_intent",  # Reuse task ID if appropriate
                    "prompt": "intent_only",
                    "input_from": [],
                }
            ],
        }
    ],
    "single_koa_only": [  # Added for completeness
        {
            "stage": 1,
            "tasks": [
                {
                    "task_id": "extract_koa",  # Reuse task ID
                    "prompt": "koa_only",
                    "input_from": [],
                }
            ],
        }
    ],
    "inference_only": [  # Assume no specific output model for now
        {
            "stage": 1,
            "tasks": [
                {
                    "task_id": "run_inference",
                    "prompt": "inference",
                    "input_from": [],
                }
            ],
        }
    ],
    # Add other common workflows here using the new structure
}


# --- Common Default Values ---
DEFAULT_JUDGE_MODEL = "google/gemini-2.5-pro-preview-03-25"
DEFAULT_NUM_SAMPLES = 10
DEFAULT_SEED = None
DEFAULT_BASE_DATA_DIR = Path(
    "./data"
)  # Base data dir for constructing specific output dirs

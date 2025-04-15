from transformers import AutoProcessor

def get_processor(model_id: str = "google/gemma-3-27b-it") -> AutoProcessor:
    return AutoProcessor.from_pretrained(model_id)

def get_token_count(processor: AutoProcessor, s: str) -> int:
    messages = [{"role": "user", "content": [{"type": "text", "text": s}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    )
    return inputs["input_ids"].shape[-1]

def get_characters_per_token(processor: AutoProcessor, s: str) -> float:
    return len(s) / get_token_count(processor, s)

def get_words_per_token(processor: AutoProcessor, s: str) -> float:
    return len(s.split()) / get_token_count(processor, s)
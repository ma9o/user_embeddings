import json
import os

import httpx

openrouter_client = None


def initialize_openrouter_client():
    global openrouter_client
    if openrouter_client is None:
        openrouter_client = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
            },
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=200),
        )
    return openrouter_client


async def get_text_completion(model_name: str, data: str) -> str:
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": data},
                ],
            }
        ],
        "provider": {"sort": "throughput"},
    }

    response = await openrouter_client.post(
        "/chat/completions", data=json.dumps(payload)
    )

    response.raise_for_status()
    completion_data = response.json()

    if completion_data.get("choices") and len(completion_data["choices"]) > 0:
        message = completion_data["choices"][0].get("message")
        if message and message.get("content"):
            return message["content"]

    raise ValueError("Could not extract completion content from the response.")

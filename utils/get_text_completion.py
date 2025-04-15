import os
import openai

def get_openrouter_client() -> openai.OpenAI:
    return openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

def get_text_completion(openrouter_client: openai.OpenAI, model_name: str, data: str) -> str:
    completion = openrouter_client.chat.completions.create(
      model=model_name,
      messages=[
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": data
            },
          ]
        }
      ]
    )

    return completion.choices[0].message.content
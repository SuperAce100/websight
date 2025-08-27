from openai import OpenAI
from pydantic import BaseModel
from typing import Any
import os
import dotenv


dotenv.load_dotenv()

text_model = "openai/gpt-4.1-mini"
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


def llm_call(
    prompt: str,
    system_prompt: str | None = None,
    response_format: type[BaseModel] | None = None,
    model: str = text_model,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt} if system_prompt else None,
        {"role": "user", "content": prompt},
    ]
    messages = [m for m in messages if m is not None]
    kwargs: dict[str, Any] = {"model": model, "messages": messages}

    if response_format is not None:
        schema = response_format.model_json_schema()
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_format.__name__,
                "strict": True,
                "schema": schema,
            },
        }
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        return content

    return client.chat.completions.create(**kwargs).choices[0].message.content or ""


def llm_call_image(
    image_base64: str,
    text: str,
    system_prompt: str | None = None,
    model: str = text_model,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt} if system_prompt else None,
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                        if "data:image/png;base64," not in image_base64
                        else image_base64
                    },
                },
                {"type": "text", "text": text},
            ],
        },
    ]
    messages = [m for m in messages if m is not None]
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""

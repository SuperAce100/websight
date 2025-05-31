from openai import OpenAI
from pydantic import BaseModel
from typing import Any
import os
import dotenv
import json

dotenv.load_dotenv()


text_model = "openai/gpt-4.1-mini"
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


def llm_call(
    prompt: str,
    system_prompt: str | None = None,
    response_format: BaseModel | None = None,
    model: str = text_model,
) -> str | BaseModel:
    """
    Make a LLM call with proper structured output for Llama 4 Maverick
    """
    messages = [
        {"role": "system", "content": system_prompt} if system_prompt else None,
        {"role": "user", "content": prompt},
    ]
    messages = [msg for msg in messages if msg is not None]

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

        if not response.choices or not response.choices[0].message.content:
            raise ValueError("No valid response content received from the API")

        response_text = response.choices[0].message.content.strip()
        try:
            return response_format.model_validate_json(response_text)
        except Exception as e:
            print(f"❌ JSON parsing failed: {str(e)}")
            print(f"Full response: {response_text}")
            raise ValueError(f"Failed to parse JSON response: {e}")

    return client.chat.completions.create(**kwargs).choices[0].message.content


def llm_call_messages(
    messages: list[dict[str, str]],
    response_format: BaseModel = None,
    model: str = text_model,
) -> str | BaseModel:
    """
    Make a LLM call with a list of messages - NO FALLBACKS
    """
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

        if not response.choices or not response.choices[0].message.content:
            raise ValueError("No valid response content received from the API")

        response_text = response.choices[0].message.content.strip()

        try:
            return response_format.model_validate_json(response_text)
        except Exception as e:
            print(f"❌ JSON parsing failed: {str(e)}")
            print(f"Full response: {response_text}")
            raise ValueError(f"Failed to parse JSON response: {e}")

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def llm_call_image(
    image_base64: str,
    text: str,
    system_prompt: str | None = None,
    model: str = text_model,
) -> str:
    """
    Make a LLM call with an image
    """
    messages = [
        {"role": "system", "content": system_prompt} if system_prompt else None,
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
                {"type": "text", "text": text},
            ],
        },
    ]
    messages = [msg for msg in messages if msg is not None]
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def test_structured_output():
    """Test structured output with a simple model."""

    class TestResponse(BaseModel):
        message: str
        number: int

    try:
        result = llm_call(
            prompt="Give me a test message and the number 42",
            response_format=TestResponse,
            model="openai/gpt-4.1-mini",
        )
        print(f"✅ Test successful: {result}")
        return True
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    test_structured_output()

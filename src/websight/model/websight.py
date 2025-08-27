from __future__ import annotations

from typing import Callable
from rich.console import Console
from transformers import pipeline

from websight.model.prompts import common_browser_system_prompt
from websight.model.actions import Action, parse_action


_websight_pipe = None


def _get_websight_pipe():
    global _websight_pipe
    if _websight_pipe is None:
        _websight_pipe = pipeline("image-text-to-text", model="tanvirb/websight-7B")
    return _websight_pipe


def _build_messages(
    prompt: str, history: list[tuple[str, str]], image_base64: str
) -> list[dict]:
    messages = [
        *[
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Thought: {reasoning}\nAction: {action}"}
                ],
            }
            for reasoning, action in history
        ],
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": common_browser_system_prompt.format(
                        language="English", instruction=prompt
                    ),
                }
            ],
        },
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
            ],
        },
    ]
    return messages


def websight_call(
    prompt: str,
    image_base64: str,
    history: list[tuple[str, str]] = [],
    console: Console | None = None,
    max_new_tokens: int = 1000,
    pipe_factory: Callable[[], Callable[..., list]] | None = None,
) -> Action:
    """
    Call the Websight model to generate an action.

    Args:
        prompt: The prompt to generate an action for.
        image_base64: The base64 encoded image to generate an action for.
        history: The history of actions and reasoning.
    """
    console = console or Console()
    messages = _build_messages(prompt, history, image_base64)
    pipe = (pipe_factory or _get_websight_pipe)()
    response = pipe(text=messages, max_new_tokens=max_new_tokens)  # type: ignore[call-arg]
    response_text = response[0]["generated_text"][-1]["content"]  # type: ignore[index]
    try:
        response_text = "temp " + str(response_text)
        reasoning = response_text.split("Thought: ")[1].split("\nAction: ")[0].strip()
        action_str = response_text.split("Action: ")[1].strip()
    except Exception as e:
        console.print(
            f"[red]Error parsing websight response: {e}.[/red]\n[red]Response:[/red] {response_text}"
        )
        return Action(action="error", args={}, reasoning=str(response_text))

    console.print(f"[blue]websight Action:[/blue] {action_str}")
    console.print(f"[blue]websight Reasoning:[/blue] {reasoning}")
    return parse_action(action_str, reasoning)

# Use a pipeline as a high-level helper
import re
from pydantic import BaseModel
from rich.console import Console
from communicators.prompts import common_browser_system_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoProcessor
from peft import PeftModel, PeftConfig

ui_tars_pipe = pipeline("image-text-to-text", model="ByteDance-Seed/UI-TARS-1.5-7B")

websight_pipe = pipeline("image-text-to-text", model="tanvirb/websight-7B")


class Action(BaseModel):
    action: str
    args: dict[str, str]
    reasoning: str

def websight_call(
    messages: list[dict[str, str]], max_new_tokens: int = 1000
) -> str:
    response = websight_pipe(text=messages, max_new_tokens=max_new_tokens)
    return response[0]["generated_text"]

def ui_tars_call(
    messages: list[dict[str, str]], max_new_tokens: int = 1000
) -> str:
    response = ui_tars_pipe(text=messages, max_new_tokens=max_new_tokens)
    return response[-1]["generated_text"][-1]["content"]



def vlm_call(
    prompt: str, history: list[tuple[str, str]], image_base64: str, model: str = "ui_tars", console: Console = Console()
) -> Action:
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
    if model == "ui_tars":
        response_text = ui_tars_call(messages)
    elif model == "websight":
        response_text = websight_call(messages)

    try:
        response_text = "temp " + response_text
        reasoning = response_text.split("Thought: ")[1].split("\nAction: ")[0].strip()
        action = response_text.split("Action: ")[1].strip()
    except Exception as e:
        console.print(
            f"[red]Error parsing UI-TARS response: {e}.[/red]\n[red]Response:[/red] {response_text}"
        )
        return Action(action="error", args={}, reasoning=response_text)

    console.print(f"[blue]UI TARS Action:[/blue] {action}")
    console.print(f"[blue]UI TARS Reasoning:[/blue] {reasoning}")
    return parse_action(action, reasoning)


def parse_action(action: str, reasoning: str) -> Action:
    """Parse UI-TARS action into our Action format."""
    if action.startswith("click"):
        if "point='" in action:
            coords = action.split("point='")[1].split("'")[0]
        else:
            coords = action.split("start_box='")[1].split("'")[0]
        x, y = map(int, coords.strip("()").split(","))
        return Action(
            action="click", args={"x": str(x), "y": str(y)}, reasoning=reasoning
        )
    elif action.startswith("left_double"):
        coords = action.split("start_box='")[1].split("'")[0]
        x, y = map(int, coords.strip("()").split(","))
        return Action(
            action="left_double", args={"x": str(x), "y": str(y)}, reasoning=reasoning
        )
    elif action.startswith("right_single"):
        coords = action.split("start_box='")[1].split("'")[0]
        x, y = map(int, coords.strip("()").split(","))
        return Action(
            action="right_single", args={"x": str(x), "y": str(y)}, reasoning=reasoning
        )
    elif action.startswith("drag"):
        start_coords = action.split("start_box='")[1].split("'")[0]
        end_coords = action.split("end_box='")[1].split("'")[0]
        start_x, start_y = map(int, start_coords.strip("()").split(","))
        end_x, end_y = map(int, end_coords.strip("()").split(","))
        return Action(
            action="drag",
            args={
                "start_x": str(start_x),
                "start_y": str(start_y),
                "end_x": str(end_x),
                "end_y": str(end_y),
            },
            reasoning=reasoning,
        )
    elif action.startswith("hotkey"):
        key = action.split("key='")[1].split("'")[0]
        return Action(action="hotkey", args={"key": key}, reasoning=reasoning)
    elif action.startswith("type"):
        content = action.split("content='")[1].split("'")[0]
        return Action(action="type", args={"content": content}, reasoning=reasoning)
    elif action.startswith("scroll"):
        if "point='" in action:
            coords = action.split("point='")[1].split("'")[0]
            x, y = map(int, coords.strip("()").split(","))
        else:
            coords = action.split("start_box='")[1].split("'")[0]
            x, y = map(int, coords.strip("()").split(","))
        direction = action.split("direction='")[1].split("'")[0]
        return Action(
            action="scroll",
            args={"x": str(x), "y": str(y), "direction": direction},
            reasoning=reasoning,
        )
    elif action.startswith("wait"):
        return Action(action="wait", args={}, reasoning=reasoning)
    elif action.startswith("finished"):
        content = action.split("content='")[1].split("'")[0]
        return Action(action="finished", args={"content": content}, reasoning=reasoning)
    elif action.startswith("goto_url"):
        url = action.split("url='")[1].split("'")[0]
        return Action(action="goto_url", args={"url": url}, reasoning=reasoning)
    else:
        raise ValueError(f"Invalid action: {action}")

# Use a pipeline as a high-level helper
from transformers import pipeline
from pydantic import BaseModel

pipe = pipeline("image-text-to-text", model="ByteDance-Seed/UI-TARS-1.5-7B")


class Action(BaseModel):
    action: str
    args: dict[str, str]
    reasoning: str


def ui_tars_call(prompt: str, image_base64: str) -> Action:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
            ],
        }
    ]
    response = pipe(text=messages, max_new_tokens=1000)
    response_text = response[-1]["generated_text"][-1]["content"]
    action = response_text.split("Action: ")[1]
    reasoning = response_text.split("Thought: ")[1]
    return parse_action(action, reasoning)


def parse_action(action: str, reasoning: str) -> Action:
    """Parse UI-TARS action into our Action format."""
    if action.startswith("click"):
        coords = action.split("start_box='")[1].split("'")[0]
        x, y = map(int, coords.strip("()").split(","))
        return Action(action="click", args={"x": str(x), "y": str(y)}, reasoning=reasoning)
    elif action.startswith("left_double"):
        coords = action.split("start_box='")[1].split("'")[0]
        x, y = map(int, coords.strip("()").split(","))
        return Action(action="left_double", args={"x": str(x), "y": str(y)}, reasoning=reasoning)
    elif action.startswith("right_single"):
        coords = action.split("start_box='")[1].split("'")[0]
        x, y = map(int, coords.strip("()").split(","))
        return Action(action="right_single", args={"x": str(x), "y": str(y)}, reasoning=reasoning)
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
            action="scroll", args={"x": str(x), "y": str(y), "direction": direction}
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

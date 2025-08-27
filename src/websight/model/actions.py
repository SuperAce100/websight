from __future__ import annotations

from pydantic import BaseModel


class Action(BaseModel):
    action: str
    args: dict[str, str]
    reasoning: str


def parse_action(action_text: str, reasoning: str) -> Action:
    action = action_text.strip()

    def _extract_coords(source: str) -> tuple[int, int]:
        if "point='" in source:
            coords = source.split("point='")[1].split("'")[0]
        else:
            coords = source.split("start_box='")[1].split("'")[0]
        x_str, y_str = coords.strip("() ").replace(" ", "").split(",")
        return int(x_str), int(y_str)

    if action.startswith("click"):
        x, y = _extract_coords(action)
        return Action(
            action="click", args={"x": str(x), "y": str(y)}, reasoning=reasoning
        )

    if action.startswith("left_double"):
        x, y = _extract_coords(action)
        return Action(
            action="left_double", args={"x": str(x), "y": str(y)}, reasoning=reasoning
        )

    if action.startswith("right_single"):
        x, y = _extract_coords(action)
        return Action(
            action="right_single", args={"x": str(x), "y": str(y)}, reasoning=reasoning
        )

    if action.startswith("drag"):
        start_coords = action.split("start_box='")[1].split("'")[0]
        end_coords = action.split("end_box='")[1].split("'")[0]
        start_x, start_y = map(
            int, start_coords.strip("() ").replace(" ", "").split(",")
        )
        end_x, end_y = map(int, end_coords.strip("() ").replace(" ", "").split(","))
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

    if action.startswith("hotkey"):
        key = action.split("key='")[1].split("'")[0]
        return Action(action="hotkey", args={"key": key}, reasoning=reasoning)

    if action.startswith("type"):
        content = action.split("content='")[1].split("'")[0]
        return Action(action="type", args={"content": content}, reasoning=reasoning)

    if action.startswith("scroll"):
        if "point='" in action or "start_box='" in action:
            x, y = _extract_coords(action)
        else:
            x, y = 500, 500
        direction = action.split("direction='")[1].split("'")[0]
        return Action(
            action="scroll",
            args={"x": str(x), "y": str(y), "direction": direction},
            reasoning=reasoning,
        )

    if action.startswith("wait"):
        return Action(action="wait", args={}, reasoning=reasoning)

    if action.startswith("finished"):
        content = action.split("content='")[1].split("'")[0]
        return Action(action="finished", args={"content": content}, reasoning=reasoning)

    if action.startswith("goto_url"):
        url = action.split("url='")[1].split("'")[0]
        return Action(action="goto_url", args={"url": url}, reasoning=reasoning)

    raise ValueError(f"Invalid action: {action}")

# Websight

Vision first browser agents based on Websight-7B, a custom 7B parameter model.

## Installation

```bash
pip install websight
# or
uv add websight
```

## Quickstart

Call the model directly on an image:

```python
from websight import websight_call

action = websight_call(
    prompt="Click the Login button",
    history=[],  # prior (reasoning, action) pairs if you have them
    image_base64="data:image/png;base64,<...>",
)
print(action.action)  # e.g., "click"
print(action.args)    # e.g., {"x": "175", "y": "514"}
```

## Reference

- websight.websight_call

```python
def websight_call(
    prompt: str,
    history: list[tuple[str, str]],
    image_base64: str,
    console: rich.console.Console | None = None,
    max_new_tokens: int = 1000,
) -> Action
```

Calls the Websight VLM with a screenshot and instruction, returning a structured `Action`.

- websight.Action

```python
class Action(BaseModel):
    action: str                # e.g. "click", "drag", "type", "scroll", ...
    args: dict[str, str]       # e.g. {"x": "175", "y": "514"}
    reasoning: str             # model rationale
```

- websight.Agent

```python
from websight.agent import Agent

agent = Agent(show_browser=False)
result = agent.run("Open https://example.com and search for 'websight'", max_iterations=10)
```

Basic Agent loop using Playwright: takes a screenshot, calls `websight_call`, parses and executes the predicted action, and repeats until it sees `finished(...)`.

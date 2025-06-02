import argparse
from browser import Browser, BrowserState
from communicators.prompts import (
    planner_prompt,
    planner_system_prompt,
    next_action_prompt,
    next_action_system_prompt,
)
from communicators.uitars import ui_tars_call, Action
from communicators.llms import llm_call, llm_call_image
from rich.console import Console
import re


PLANNING_MODEL = "openai/gpt-4.1-mini"
NEXT_ACTION_MODEL = "openai/gpt-4.1-mini"


class Agent:
    def __init__(self):
        self.browser = Browser()
        self.console = Console()

    def execute_action(
        self, next_action: str, history: list[tuple[str, str]]
    ) -> Action | None:
        current_state = self.browser.get_state()

        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        url_match = re.search(url_pattern, next_action)

        if url_match:
            url = url_match.group(0)
            self.browser.goto_url(url)
            return Action(
                action="goto_url",
                args={"url": url},
                reasoning=f"Navigated to {url}",
            )

        action = ui_tars_call(
            next_action, history, current_state.page_screenshot_base64
        )
        

        try:
            if action.action == "click":
                self.browser.click(int(action.args["x"]), int(action.args["y"]))
            elif action.action == "left_double":
                self.browser.left_double(int(action.args["x"]), int(action.args["y"]))
            elif action.action == "right_single":
                self.browser.right_single(int(action.args["x"]), int(action.args["y"]))
            elif action.action == "drag":
                self.browser.drag(
                    int(action.args["start_x"]),
                    int(action.args["start_y"]),
                    int(action.args["end_x"]),
                    int(action.args["end_y"]),
                )
            elif action.action == "hotkey":
                self.browser.hotkey(action.args["key"])
            elif action.action == "type":
                self.browser.type(action.args["content"])
            elif action.action == "scroll":
                self.browser.scroll(
                    int(action.args["x"]),
                    int(action.args["y"]),
                    action.args["direction"],
                )
            elif action.action == "wait":
                self.browser.wait()
            elif action.action == "finished":
                return action.args["content"]
            elif action.action == "goto_url":
                self.browser.goto_url(action.args["url"])
            else:
                raise ValueError(f"Invalid action: {action.action}")
        except Exception as e:
            self.console.print(f"[red]Error:[/red] {e}")
            return None
        return action

    def make_plan(self, task: str) -> list[str]:
        response = llm_call(
            planner_prompt.format(task=task),
            system_prompt=planner_system_prompt,
            model=PLANNING_MODEL,
        )

        steps = []
        for line in response.split("\n"):
            if "<step>" in line and "</step>" in line:
                step = line.split("<step>")[1].split("</step>")[0].strip()
                steps.append(step)
        return steps

    def choose_next_action(
        self, plan: list[str], state: BrowserState, history: list[tuple[str, str]]
    ) -> tuple[str, str]:
        response = llm_call_image(
            state.page_screenshot_base64,
            next_action_prompt.format(plan=plan, history=history),
            system_prompt=next_action_system_prompt,
            model=NEXT_ACTION_MODEL,
        )

        reasoning = response.split("<reasoning>")[1].split("</reasoning>")[0].strip()
        action = response.split("<action>")[1].split("</action>")[0].strip()

        return reasoning, action

    def run(self, task: str, max_iterations: int = 25):
        plan = self.make_plan(task)
        self.console.print(f"[green]Plan:[/green]")
        for i, step in enumerate(plan, 1):
            self.console.print(f"{i}. {step}")
        history = []
        for _ in range(max_iterations):
            state = self.browser.get_state()
            reasoning, action = self.choose_next_action(plan, state, history)
            self.console.print(f"[green]Reasoning:[/green] {reasoning}")
            self.console.print(f"[green]Action:[/green] {action}")
            history.append((reasoning, action))
            if "finished" in action.lower():
                self.console.print(
                    f"[bold green]Task completed successfully[/bold green]"
                )
                break

            self.execute_action(action, history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--max-iters", type=int, default=25)
    args = parser.parse_args()
    task = args.task
    console = Console()
    console.print(f"[green]Task:[/green] {task}")
    agent = Agent()
    result = agent.run(task, args.max_iters)
    if "Error" not in result:
        console.print(f"[green]Result:[/green] {result}")
    else:
        console.print(f"[red]Error:[/red] {result}.")


if __name__ == "__main__":
    main()

import argparse
from typing import List
from pydantic import BaseModel
from browser import Browser, BrowserState
from communicators.uitars import ui_tars_call, Action
from communicators.llms import llm_call
from rich.console import Console
import json


class Agent:
    def __init__(self):
        self.browser = Browser()
        self.console = Console()
        self.plan: list[str] = []

    def execute_action(self, next_action: str) -> Action | None:
        current_state = self.browser.get_state()
        action = ui_tars_call(next_action, current_state.screenshot_base64)
        self.console.print(f"[green]Action:[/green] {action.action}")
        self.console.print(f"[green]Reasoning:[/green] {action.reasoning}")

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

    def make_plan(
        self, task: str, state: BrowserState, history: list[str]
    ) -> list[str]:
        pass

    def choose_next_action(self):
        pass

    def run(self, task: str, max_iterations: int = 25):
        plan = self.make_plan(task)
        history = []
        for _ in range(max_iterations):
            next_action = self.choose_next_action(
                plan, self.browser.get_state(), history
            )
            if next_action is None:
                break
            self.execute_action(next_action)


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

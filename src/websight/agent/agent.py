import re
from datetime import datetime
from rich.console import Console

from websight.agent.browser import Browser
from websight.model.actions import Action
from websight.model.websight import websight_call
from websight.model.llm import llm_call, llm_call_image


PLANNING_MODEL = "openai/gpt-4.1-mini"
NEXT_ACTION_MODEL = "openai/gpt-4.1-mini"


class Agent:
    def __init__(self, show_browser: bool = False):
        self.browser = Browser(show_browser=show_browser)
        self.console = Console()

    def execute_action(
        self, next_action: str, history: list[tuple[str, str]]
    ) -> Action | str | None:
        current_state = self.browser.get_state()

        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        url_match = re.search(url_pattern, next_action)
        if url_match:
            url = url_match.group(0)
            self.browser.goto_url(url)
            return Action(
                action="goto_url", args={"url": url}, reasoning=f"Navigated to {url}"
            )

        action = websight_call(
            next_action,
            history,
            current_state.page_screenshot_base64,
            console=self.console,
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

    def run(self, task: str, max_iterations: int = 25):
        # minimal plan using LLM; prompts can be extended later
        system_plan = (
            f"You are a planner. Today's date is {datetime.now().strftime('%Y-%m-%d')}"
        )
        plan_text = llm_call(
            f"Create a minimal plan for: {task}",
            system_prompt=system_plan,
            model=PLANNING_MODEL,
        )
        self.console.print("[green]Plan:[/green]")
        for i, step in enumerate([s for s in plan_text.split("\n") if s.strip()], 1):
            self.console.print(f"{i}. {step}")

        history: list[tuple[str, str]] = []
        for _ in range(max_iterations):
            state = self.browser.get_state()
            prompt = f"Plan: {plan_text}\nHistory: {history}\nInstruction: {task}\nRespond with <reasoning>...</reasoning><action>...</action>"
            system_next = (
                f"Today is {datetime.now().strftime('%Y-%m-%d')}, URL: {state.page_url}"
            )
            response = llm_call_image(
                state.page_screenshot_base64,
                prompt,
                system_prompt=system_next,
                model=NEXT_ACTION_MODEL,
            )
            try:
                reasoning = (
                    response.split("<reasoning>")[1].split("</reasoning>")[0].strip()
                )
                action = response.split("<action>")[1].split("</action>")[0].strip()
            except Exception:
                reasoning, action = "", response.strip()

            self.console.print(f"[green]Reasoning:[/green] {reasoning}")
            self.console.print(f"[green]Action:[/green] {action}")
            history.append((reasoning, action))
            if "finished" in action.lower():
                self.console.print(
                    "[bold green]Task completed successfully[/bold green]"
                )
                return action
            self.execute_action(action, history)

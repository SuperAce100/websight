import argparse
from typing import List
from pydantic import BaseModel
from browser import Browser
from communicators.uitars import ui_tars_call
from communicators.llms import llm_call
from rich.console import Console
import json

from prompts import common_browser_system_prompt, planner_prompt


class Action(BaseModel):
    action: str
    args: dict[str, str]


class Agent:
    def __init__(self):
        self.browser = Browser()
        self.console = Console()

    def _parse_action(self, action: str) -> Action:
        """Parse UI-TARS action into our Action format."""
        if action.startswith("click"):
            coords = action.split("start_box='")[1].split("'")[0]
            x, y = map(int, coords.strip("()").split(","))
            return Action(action="click", args={"x": str(x), "y": str(y)})
        elif action.startswith("left_double"):
            coords = action.split("start_box='")[1].split("'")[0]
            x, y = map(int, coords.strip("()").split(","))
            return Action(action="left_double", args={"x": str(x), "y": str(y)})
        elif action.startswith("right_single"):
            coords = action.split("start_box='")[1].split("'")[0]
            x, y = map(int, coords.strip("()").split(","))
            return Action(action="right_single", args={"x": str(x), "y": str(y)})
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
            )
        elif action.startswith("hotkey"):
            key = action.split("key='")[1].split("'")[0]
            return Action(action="hotkey", args={"key": key})
        elif action.startswith("type"):
            content = action.split("content='")[1].split("'")[0]
            return Action(action="type", args={"content": content})
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
            )
        elif action.startswith("wait"):
            return Action(action="wait", args={})
        elif action.startswith("finished"):
            content = action.split("content='")[1].split("'")[0]
            return Action(action="finished", args={"content": content})
        elif action.startswith("goto_url"):
            url = action.split("url='")[1].split("'")[0]
            return Action(action="goto_url", args={"url": url})
        else:
            raise ValueError(f"Invalid action: {action}")

    def _execute_action(self, action: Action):
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
            return False
        return True

    def run(self, task: str, max_iterations: int = 25):
        iteration = 0

        procedural_summaries = []
        for url in set(ep["url"] for ep in self.memory.memory["episodic"] if ep["url"]):
            procedural_summary = self.memory.get_procedural_summary(url)
            if procedural_summary != "No successful approaches recorded yet.":
                procedural_summaries.append(f"Site: {url}\n{procedural_summary}")

        plan = llm_call(
            prompt=planner_prompt.format(task=task)
            + "\n\nPrevious Successful Approaches:\n"
            + "\n\n".join(procedural_summaries),
            model="openai/gpt-4.1-mini",
        )

        start_url = None
        for line in plan.split("\n"):
            if line.startswith("START_URL:"):
                start_url = line.split("START_URL:")[1].strip()
                self.console.print(f"[green]Starting URL:[/green] {start_url}")
                break

        site_summaries = []
        recent_episodes = []

        if start_url:
            site_summary = self.memory.get_site_summary(start_url)
            if site_summary != "No experience with this site yet.":
                site_summaries.append(f"Site: {start_url}\n{site_summary}")

            episodes = self.memory.get_recent_episodes(start_url, limit=3)
            if episodes:
                recent_episodes.extend(episodes)

            if site_summary != "No experience with this site yet.":
                self.console.print(f"[blue]Site Experience:[/blue] {site_summary}")

            procedural_summary = self.memory.get_procedural_summary(start_url)
            if procedural_summary != "No successful approaches recorded yet.":
                self.console.print(
                    f"[blue]Successful Approaches:[/blue] {procedural_summary}"
                )

            self.browser.goto_url(start_url)

        memory_context = ""
        if site_summaries:
            memory_context += "\n\nSite Patterns and Issues:\n" + "\n\n".join(
                site_summaries
            )
        if recent_episodes:
            memory_context += "\n\nRecent Episodes:\n" + json.dumps(
                recent_episodes, indent=2
            )

        all_messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": common_browser_system_prompt.format(
                            language="English", instruction=task
                        )
                        + memory_context,
                    },
                ],
            },
        ]

        all_messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": plan}],
            }
        )

        last_action_success = True
        all_actions = []
        last_action = ""
        while iteration < max_iterations:
            iteration += 1
            state = self.browser.get_state()

            for message in all_messages:
                if message["role"] == "user":
                    message["content"] = [
                        item for item in message["content"] if item["type"] != "image"
                    ]

            all_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": state.page_screenshot_base64,
                        }
                    ],
                }
            )
            action, response = ui_tars_call(all_messages)
            all_messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}],
                }
            )

            self.console.print(f"[green]Response:[/green] {response}")

            action = self._parse_action(action)
            if action.action == "finished":
                success_evaluation = (
                    llm_call(
                        prompt=f"Evaluate if the following task was completed successfully. Task: {task}\nResult: {action.args['content']}\nRespond with just 'SUCCESS' or 'FAILURE'",
                        model="openai/gpt-4.1-mini",
                    )
                    .strip()
                    .upper()
                )

                success = success_evaluation == "SUCCESS"

                insights = self.memory._generate_insights(
                    task=task, result=action.args["content"], success=success
                )

                self.memory.add_episode(
                    task=task,
                    success=success,
                    trajectory=all_actions,
                    url=start_url or "",
                    insights=insights,
                )

                chinese_result = action.args["content"]
                self.console.print(
                    f"[green]Chinese result:[/green] {chinese_result}", style="dim"
                )

                result = llm_call(
                    f"Translate the following result into English: {chinese_result}"
                )
                return result

            last_action_success = self._execute_action(action)

            # if action == last_action:
            #     success_evaluation = "FAILURE"
            #     insights = self.memory._generate_insights(
            #         task=task, result="Error: Repeated action", success=False
            #     )

            #     self.memory.add_episode(
            #         task=task,
            #         success=False,
            #         trajectory=all_actions,
            #         url=start_url or "",
            #         insights=insights,
            #     )

            #     return "Error: repeated action"

            if last_action_success:
                all_actions.append(action.dict())
            last_action = action

        return "Error: max iterations reached"


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

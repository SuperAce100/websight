import argparse
from websight.agent import Agent
from rich.console import Console


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--max-iters", type=int, default=25)
    parser.add_argument("--show-browser", action="store_true")
    args = parser.parse_args()
    task = args.task
    console = Console()
    console.print(f"[green]Task:[/green] {task}")
    agent = Agent(show_browser=args.show_browser)
    result = agent.run(task, args.max_iters)
    if isinstance(result, str) and "Error" not in result:
        console.print(f"[green]Result:[/green] {result}")
    else:
        console.print(f"[red]Error:[/red] {result}.")


if __name__ == "__main__":
    main()

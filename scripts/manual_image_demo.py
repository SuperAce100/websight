import argparse
import base64
from pathlib import Path
from rich.console import Console

from websight import websight_call


def to_data_url(path: Path) -> str:
    data = path.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


def main():
    parser = argparse.ArgumentParser(description="Manual Websight call on an image")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument(
        "--prompt",
        default="Click the target area",
        help="Instruction for the model",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256, help="Generation length"
    )
    args = parser.parse_args()

    console = Console()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_b64 = to_data_url(image_path)
    console.print(f"[green]Running websight on:[/green] {image_path}")
    action = websight_call(
        prompt=args.prompt,
        history=[],
        image_base64=image_b64,
        console=console,
        max_new_tokens=args.max_new_tokens,
    )
    console.print(f"[bold]Parsed Action:[/bold] {action.action}")
    console.print(f"[bold]Args:[/bold] {action.args}")
    console.print(f"[dim]Reasoning:[/dim] {action.reasoning}")


if __name__ == "__main__":
    main()

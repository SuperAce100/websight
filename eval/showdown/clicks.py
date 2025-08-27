import os
import time
import argparse
import base64
import json
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
from datasets import Dataset
from rich.console import Console
from rich.progress import Progress

from model.websight import websight_call
from eval.showdown.utils import (
    check_prediction_in_bbox,
    print_colored_result,
    analyze_results,
    visualize_prediction,
    EvaluationItem,
    EvaluationResult,
)

console = Console()


def load_showdown_dataset(split: str = "dev") -> Dataset:
    try:
        console.print("[cyan]Loading showdown-clicks dataset from local JSON...[/cyan]")
        json_path = Path("data/showdown_clicks/showdown_clicks.json")
        if not json_path.exists():
            raise FileNotFoundError(f"Dataset JSON file not found at {json_path}")

        with open(json_path) as f:
            data = json.load(f)

        dataset = Dataset.from_list(data)
        console.print(f"[green]Successfully loaded {len(dataset)} examples[/green]")
        return dataset
    except Exception as e:
        console.print(f"[red]Failed to load dataset: {e}[/red]")
        raise


def validate_dataset(dataset: Dataset) -> None:
    required_fields = [
        "id",
        "image_path",
        "instruction",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
    ]
    missing_fields = [
        field for field in required_fields if field not in dataset.features
    ]

    if missing_fields:
        raise ValueError(f"Dataset missing required fields: {missing_fields}")

    example = dataset[0]
    try:
        EvaluationItem(
            id=example["id"],
            recording_id=example["recording_id"],
            instruction=example["instruction"],
            image=example["image_path"],
            x1=example["x1"],
            y1=example["y1"],
            x2=example["x2"],
            y2=example["y2"],
            width=example["width"],
            height=example["height"],
        )
    except Exception as e:
        raise ValueError(f"Invalid dataset format: {e}")


def get_image_base64(image_path: str) -> Optional[str]:
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        return f"data:image/png;base64,{base64.b64encode(image_data).decode()}"
    except Exception as e:
        console.print(f"[red]Failed to convert image to base64: {e}[/red]")
        return None


def load_existing_results(results_dir: Path) -> Dict[str, Dict]:
    results_file = results_dir / "results.npy"
    if results_file.exists():
        try:
            results = np.load(results_file, allow_pickle=True)
            return {r["id"]: r for r in results}
        except Exception as e:
            console.print(f"[yellow]Could not load existing results: {e}[/yellow]")
    return {}


def save_results(results: List[Dict], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    np.save(results_dir / "results.npy", results)


def evaluate_websight_on_showdown(
    model_name: str = "websight",
    run_id: Optional[str] = None,
    visualize: bool = True,
    max_examples: Optional[int] = None,
    output_dir: str = "data/showdown_clicks",
) -> None:
    output_dir = Path(output_dir)

    dataset = load_showdown_dataset()
    validate_dataset(dataset)

    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
        console.print(f"[yellow]Evaluating on {max_examples} examples[/yellow]")

    results_dir = (
        output_dir / run_id / model_name if run_id else output_dir / model_name
    )
    existing_results = load_existing_results(results_dir)
    console.print(f"[cyan]Found {len(existing_results)} existing results[/cyan]")

    results: List[Dict] = list(existing_results.values())
    remaining_examples = [ex for ex in dataset if ex["id"] not in existing_results]

    if not remaining_examples:
        console.print("[green]All examples already evaluated![/green]")
    else:
        console.print(
            f"[cyan]Evaluating {len(remaining_examples)} remaining examples[/cyan]"
        )

        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Evaluating...", total=len(remaining_examples)
            )

            for example in remaining_examples:
                image_path = example["image_path"]
                image_base64 = get_image_base64(image_path)
                if not image_base64:
                    console.print(
                        f"[red]Failed to process image for example {example['id']}[/red]"
                    )
                    continue

                gt_x1, gt_y1 = example["x1"], example["y1"]
                gt_x2, gt_y2 = example["x2"], example["y2"]

                start_time = time.time()
                try:
                    action = websight_call(
                        prompt=example["instruction"],
                        history=[],
                        image_base64=image_base64,
                    )
                    latency = time.time() - start_time

                    pred_x = (
                        int(action.args.get("x", 0))
                        if action.action == "click"
                        else None
                    )
                    pred_y = (
                        int(action.args.get("y", 0))
                        if action.action == "click"
                        else None
                    )

                    is_in_bbox = check_prediction_in_bbox(
                        pred_x, pred_y, gt_x1, gt_y1, gt_x2, gt_y2
                    )

                    print_colored_result(
                        example["id"],
                        example["instruction"],
                        pred_x,
                        pred_y,
                        latency,
                        is_in_bbox,
                    )

                    vis_path = None
                    if visualize:
                        vis_path = visualize_prediction(
                            image_path=image_path,
                            pred_x=pred_x,
                            pred_y=pred_y,
                            item_id=example["id"],
                            recording_id=example["recording_id"],
                            instruction=example["instruction"],
                            model_name=model_name,
                            run_id=run_id,
                            gt_x1=gt_x1,
                            gt_y1=gt_y1,
                            gt_x2=gt_x2,
                            gt_y2=gt_y2,
                            is_in_bbox=is_in_bbox,
                        )

                    result = EvaluationResult(
                        id=example["id"],
                        recording_id=example["recording_id"],
                        instruction=example["instruction"],
                        image_path=image_path,
                        gt_x1=gt_x1,
                        gt_y1=gt_y1,
                        gt_x2=gt_x2,
                        gt_y2=gt_y2,
                        pred_x=pred_x,
                        pred_y=pred_y,
                        is_in_bbox=is_in_bbox,
                        latency_seconds=latency,
                        visualization_path=vis_path,
                        raw_response=action.reasoning,
                    )
                    results.append(result.model_dump())

                    save_results(results, results_dir)

                except Exception as e:
                    console.print(
                        f"[red]Error processing example {example['id']}: {e}[/red]"
                    )
                    continue

                progress.update(task, advance=1)

    if not results:
        console.print("[red]No results to analyze - all examples failed[/red]")
        return None

    metrics = analyze_results(results, run_id)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate websight-7b on showdown-clicks benchmark"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID for organizing results (default: timestamp)",
    )
    parser.add_argument(
        "--no-visualize", action="store_true", help="Disable visualization generation"
    )
    parser.add_argument(
        "--max-examples", type=int, help="Maximum number of examples to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/showdown_clicks",
        help="Directory to save results (default: data/showdown_clicks)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.run_id:
        args.run_id = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate_websight_on_showdown(
        run_id=args.run_id,
        visualize=not args.no_visualize,
        max_examples=args.max_examples,
        output_dir=args.output_dir,
    )

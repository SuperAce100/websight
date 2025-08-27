import base64
import os
import urllib.parse
from typing import Any, Dict, List, Optional

import numpy as np
from colorama import Fore, Style, init
from PIL import Image, ImageDraw, ImageFont
from scipy import stats
from pydantic import BaseModel

init(autoreset=True)


class EvaluationMetrics(BaseModel):
    total_processed: int
    total_correct: int
    accuracy: float
    ci: float
    accuracy_ci_low: Optional[float] = None
    accuracy_ci_high: Optional[float] = None


class EvaluationItem(BaseModel):
    id: str
    recording_id: str
    instruction: str
    image: str
    x1: Optional[int] = None
    y1: Optional[int] = None
    x2: Optional[int] = None
    y2: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None


class EvaluationResult(BaseModel):
    id: str
    recording_id: str
    instruction: str
    image_path: str
    gt_x1: Optional[int] = None
    gt_y1: Optional[int] = None
    gt_x2: Optional[int] = None
    gt_y2: Optional[int] = None
    pred_x: Optional[int] = None
    pred_y: Optional[int] = None
    is_in_bbox: Optional[bool] = None
    latency_seconds: float
    visualization_path: Optional[str] = None
    raw_response: Optional[str] = None


def is_point_in_bbox(
    x: Optional[int],
    y: Optional[int],
    x1: Optional[int],
    y1: Optional[int],
    x2: Optional[int],
    y2: Optional[int],
) -> bool:
    if x is None or y is None or x1 is None or y1 is None or x2 is None or y2 is None:
        return False
    return x1 <= x <= x2 and y1 <= y <= y2


def visualize_prediction(
    image_path: str,
    pred_x: Optional[int],
    pred_y: Optional[int],
    item_id: str,
    recording_id: str,
    instruction: str,
    model_name: str,
    run_id: Optional[str],
    gt_x1: Optional[int],
    gt_y1: Optional[int],
    gt_x2: Optional[int],
    gt_y2: Optional[int],
    is_in_bbox: Optional[bool] = None,
) -> Optional[str]:
    try:
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        if run_id:
            vis_dir = os.path.join(
                base_dir, "results", run_id, model_name, "visualizations"
            )
        else:
            vis_dir = os.path.join(base_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        print(f"Visualization directory: {vis_dir}")

        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("Arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()

        image_filename = os.path.basename(image_path)

        if all(v is not None for v in [gt_x1, gt_y1, gt_x2, gt_y2]):
            draw.rectangle([(gt_x1, gt_y1), (gt_x2, gt_y2)], outline="blue", width=2)  # type: ignore
            if gt_y1 is not None:
                draw.text((gt_x1, gt_y1 - 20), "Bounding Box", fill="blue", font=font)  # type: ignore
            else:
                draw.text((gt_x1, 0), "Bounding Box", fill="blue", font=font)  # type: ignore

        if pred_x is not None and pred_y is not None:
            outline_color = "orange" if is_in_bbox else "red"
            draw.ellipse(
                [(pred_x - 15, pred_y - 15), (pred_x + 15, pred_y + 15)],
                outline=outline_color,
                width=3,
            )
            draw.text(
                (pred_x + 20, pred_y + 30), "Prediction", fill=outline_color, font=font
            )

        draw.text(
            (10, img.height - 30),
            f"Item ID: {item_id} | Recording ID: {recording_id} | Instruction: {instruction}",
            fill="white",
            font=font,
        )

        output_filename = f"rec{recording_id}_item{item_id}_{image_filename}_{urllib.parse.quote_plus(instruction)}.png"
        output_dir = os.path.join(vis_dir, "correct" if is_in_bbox else "incorrect")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        img.save(output_path)

        return output_path
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return None


def encode_image_to_base64(image_path: str) -> Optional[str]:
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        encoded_data = base64.b64encode(image_data).decode("utf-8")
        mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
        return f"data:{mime_type};base64,{encoded_data}"
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None


def check_prediction_in_bbox(
    pred_x: Optional[int],
    pred_y: Optional[int],
    gt_x1: Optional[int],
    gt_y1: Optional[int],
    gt_x2: Optional[int],
    gt_y2: Optional[int],
) -> bool:
    return is_point_in_bbox(pred_x, pred_y, gt_x1, gt_y1, gt_x2, gt_y2)


def print_colored_result(
    item_id: str,
    instruction: str,
    pred_x: Optional[int],
    pred_y: Optional[int],
    latency: float,
    is_in_bbox: Optional[bool] = None,
) -> None:
    color = Fore.GREEN if is_in_bbox else Fore.RED
    print(
        f"{color}ID: {item_id} | Instruction: {instruction} | "
        f"Prediction: {pred_x} {pred_y} | "
        f"Correct: {is_in_bbox} | Time: {latency:.2f}s{Style.RESET_ALL}"
    )


def analyze_results(
    results: List[Dict[str, Any]], run_id: Optional[str] = None
) -> EvaluationMetrics:
    if not results:
        raise ValueError("No results to summarize")

    total_processed = len(results)
    total_in_bbox = sum(1 for result in results if result.get("is_in_bbox", False))

    bbox_results = np.array(
        [
            1 if result.get("is_in_bbox", False) else 0
            for result in results
            if "is_in_bbox" in result
        ]
    )

    accuracy = (total_in_bbox / total_processed) * 100 if total_processed > 0 else 0.0

    accuracy_ci = None
    ci = 0.95

    def calculate_accuracy(data):
        return np.mean(data) * 100

    if len(bbox_results) > 0:
        try:
            accuracy_bootstrap = stats.bootstrap(
                (bbox_results,),
                calculate_accuracy,
                confidence_level=ci,
                method="percentile",
            )
            accuracy_ci = accuracy_bootstrap.confidence_interval
        except Exception as e:
            print(f"Error calculating bounding box accuracy confidence interval: {e}")

    print("\nResults Summary:")
    print(f"Total Processed: {total_processed}")
    print(f"Total Correct: {total_in_bbox}")
    print(f"Accuracy: {accuracy:.2f}%")
    if accuracy_ci:
        print(f"95% CI: [{accuracy_ci.low:.2f}%, {accuracy_ci.high:.2f}%]")

    metrics = EvaluationMetrics(
        total_processed=total_processed,
        total_correct=total_in_bbox,
        accuracy=accuracy,
        ci=ci,
        accuracy_ci_low=accuracy_ci.low if accuracy_ci else None,
        accuracy_ci_high=accuracy_ci.high if accuracy_ci else None,
    )

    return metrics

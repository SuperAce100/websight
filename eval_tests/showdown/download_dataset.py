#!/usr/bin/env python3
"""
Download the full showdown-clicks dataset from Hugging Face.
"""

import argparse
import base64
import io
import requests
from pathlib import Path
from datasets import load_dataset
from rich.console import Console
from rich.progress import Progress
import json
from PIL import Image
from typing import Optional

console = Console()

def download_image(image_path: str, output_dir: Path, example_id: str) -> str:
    """
    Download image from Hugging Face and convert to PNG.
    
    Args:
        image_path: Relative path to image in dataset
        output_dir: Directory to save image
        example_id: ID of the example for filename
        
    Returns:
        str: Path to saved image
    """
    try:
        # Construct full URL
        base_url = "https://huggingface.co/datasets/generalagents/showdown-clicks/resolve/main/showdown-clicks-dev"
        url = f"{base_url}/{image_path}"
        
        # Download image
        response = requests.get(url)
        response.raise_for_status()
        
        # Convert to PNG
        img = Image.open(io.BytesIO(response.content))
        output_path = output_dir / f"{example_id}.png"
        img.save(output_path, "PNG")
            
        return str(output_path)
    except Exception as e:
        console.print(f"[red]Failed to download image for example {example_id}: {e}[/red]")
        return None

def download_dataset(output_dir: str = "data/showdown_clicks", max_examples: Optional[int] = None) -> None:
    """
    Download the full showdown-clicks dataset.
    
    Args:
        output_dir: Directory to save the dataset
        max_examples: Optional limit on number of examples to download
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create images directory
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    console.print("[cyan]Loading showdown-clicks dataset...[/cyan]")
    dataset = load_dataset("generalagents/showdown-clicks", split="dev")
    
    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
        console.print(f"[yellow]Downloading {max_examples} examples[/yellow]")
    else:
        console.print(f"[green]Downloading all {len(dataset)} examples[/green]")
    
    # Process dataset and download images
    processed_data = []
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing examples...", total=len(dataset))
        
        for example in dataset:
            # Download image
            image_path = download_image(example["image"], images_dir, example["id"])
            
            # Create processed example
            processed_example = {
                "id": example["id"],
                "recording_id": example["id"].split("_")[0],
                "instruction": example["instruction"],
                "image_path": image_path,
                "x1": example["x1"],
                "y1": example["y1"],
                "x2": example["x2"],
                "y2": example["y2"],
                "width": example["width"],
                "height": example["height"]
            }
            processed_data.append(processed_example)
            
            progress.update(task, advance=1)
    
    # Save the processed dataset as JSON
    output_file = output_dir / "showdown_clicks.json"
    console.print(f"[cyan]Saving dataset to {output_file}...[/cyan]")
    
    with open(output_file, "w") as f:
        json.dump(processed_data, f, indent=2)
    
    console.print("[green]Dataset saved successfully![/green]")
    console.print(f"\nDataset stats:")
    console.print(f"Total examples: {len(processed_data)}")
    console.print(f"Features: {list(processed_data[0].keys())}")
    console.print(f"Images saved to: {images_dir}")
    
    # Print first example for reference
    console.print("\nFirst example:")
    example = processed_data[0]
    for key, value in example.items():
        console.print(f"{key}: {value}")

def parse_args():
    parser = argparse.ArgumentParser(description="Download showdown-clicks dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/showdown_clicks",
        help="Directory to save dataset (default: data/showdown_clicks)"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Maximum number of examples to download"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    download_dataset(args.output_dir, args.max_examples) 
#!/usr/bin/env python3
"""
Prepare Wave-UI-25K (single-element format) for LLaMA-Factory SFT / LoRA fine-tuning.
Outputs
-------
data/
  images/                   # PNG screenshots
  waveui_train.jsonl
  waveui_val.jsonl
  waveui_test.jsonl
train_waveui_lora.yaml      # ready for llamafactory-cli
"""

import argparse, base64, io, json, math, pathlib, random, re, uuid, yaml
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, disable_caching

disable_caching()


# ───────── CLI ───────── #
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data")
    p.add_argument("--img_root", default="images")
    p.add_argument("--model", default="ByteDance-Seed/UI-TARS-1.5-7B")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.03)
    p.add_argument("--test_frac", type=float, default=0.03)
    return p.parse_args()


# ───────── helpers ───────── #
ACTION_RE = re.compile(r"Action\s*:\s*(.*)", flags=re.I | re.S)


def strip_thought(text: str | None) -> str:
    if not text:
        return ""
    m = ACTION_RE.search(text)
    return (m.group(1) if m else text).strip()


def scale_bbox(bbox, resolution):
    """bbox = [x1,y1,x2,y2] absolute pixels, resolution=[W,H]. → 0–999 ints."""
    W, H = resolution
    x1, y1, x2, y2 = bbox
    return (
        round(x1 * 999 / W),
        round(y1 * 999 / H),
        round(x2 * 999 / W),
        round(y2 * 999 / H),
    )


def make_chat(img_path, question, answer):
    return {
        "id": str(uuid.uuid4()),
        "conversations": [
            {"from": "human", "value": f"<img>{img_path}</img> {question}"},
            {"from": "gpt", "value": answer},
        ],
    }


def save_image(pil_or_b64, img_dir: pathlib.Path) -> str:
    """Handles PIL.Image or base-64 string → saves PNG, returns relative path."""
    if isinstance(pil_or_b64, Image.Image):
        img = pil_or_b64
    else:  # assume base-64
        img = Image.open(io.BytesIO(base64.b64decode(pil_or_b64)))
    fname = f"{uuid.uuid4().hex}.png"
    img.save(img_dir / fname, format="PNG")
    return f"{img_dir.name}/{fname}"


# ───────── main ───────── #
def main():
    args = get_args()
    random.seed(args.seed)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / args.img_root
    img_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Wave-UI-25K …")
    ds = load_dataset(
        "agentsea/wave-ui-25k", split="train"
    )  # HF dataset schema  [oai_citation:0‡Hugging Face](https://huggingface.co/datasets/agentsea/wave-ui-25k?utm_source=chatgpt.com)

    rows = []
    print("Converting records …")
    for s in tqdm(ds, total=len(ds)):
        img_rel = save_image(s["image"], img_dir)

        # scale bbox
        x1, y1, x2, y2 = scale_bbox(s["bbox"], s["resolution"])

        # build NL question
        parts = [f"Click the {s.get('name') or s.get('type', 'element')}"]
        if s.get("purpose"):
            parts.append(f"({s['purpose']})")
        if s.get("OCR"):
            parts.append(f'whose text reads "{s["OCR"]}"')
        question = " ".join(parts) + "."

        # answer: action-only box
        answer = (
            strip_thought(s.get("ui_tars_trace"))
            or f"<box>({x1},{y1}),({x2},{y2})</box>"
        )
        rows.append(make_chat(img_rel, question, answer))

    print(f"Total rows: {len(rows):,}")

    # shuffle & split
    random.shuffle(rows)
    test_sz = math.ceil(len(rows) * args.test_frac)
    val_sz = math.ceil(len(rows) * args.val_frac)
    splits = {
        "test": rows[:test_sz],
        "val": rows[test_sz : test_sz + val_sz],
        "train": rows[test_sz + val_sz :],
    }

    for tag, subset in splits.items():
        path = out_dir / f"waveui_{tag}.jsonl"
        with open(path, "w") as f:
            for r in subset:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{tag.capitalize():5} → {len(subset):,} rows  ({path})")

    print("\n✅ Dataset ready.")
    print(f"Train with:\n  llamafactory-cli train train_waveui_lora.yaml")
    print(
        "Evaluate with:\n  llamafactory-cli evaluate "
        "--model output/ui_tars_waveui_lora "
        f"--dataset {out_dir / 'waveui_test.jsonl'} --format sharegpt"
    )


if __name__ == "__main__":
    main()

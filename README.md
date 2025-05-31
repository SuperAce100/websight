# Websight

## Prepare for finetune


```bash
cd LLaMA-Factory
uv pip install -e .
cd ..
uv run -m model.prepare_data
```

Put the following in `data/dataset_info.json`

```json
{
  "waveui_clicks": {
    "file_name": "waveui_train.jsonl",
    "eval_file_name": "waveui_val.jsonl",
    "template": "qwen2_vl",
    "formatting": "sharegpt",
    "columns": { "messages": "conversations", "images": null }
  }
}
```

Train with:

```bash
llamafactory-cli train train_waveui_lora.yaml
```

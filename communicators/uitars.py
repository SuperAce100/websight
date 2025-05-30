# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("image-text-to-text", model="ByteDance-Seed/UI-TARS-1.5-7B")


def ui_tars_call(messages):
    response = pipe(text=messages, max_new_tokens=1000)
    response_text = response[-1]["generated_text"][-1]["content"]
    action = response_text.split("Action: ")[1]
    return action, response_text

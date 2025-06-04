from transformers import pipeline
from communicators.prompts import common_browser_system_prompt
import base64

pipe = pipeline("image-text-to-text", model="tanvirb/websight-7B")

img_path = "data/screenshots/screenshot_1748735884.270176.png"

with open(img_path, "rb") as image_file:
    image_b64 = base64.b64encode(image_file.read()).decode("utf-8")

messages = [
    {"role": "system", "content": [{"type": "text", "text": common_browser_system_prompt.format(language="English", instruction="Click on the orange case.")}]},
    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}]},
]

print(pipe(text=messages, max_new_tokens=1000))
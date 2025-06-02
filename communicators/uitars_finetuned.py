# Finetuned UI-TARS model implementation
import re
import torch
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from pydantic import BaseModel
from rich.console import Console
from communicators.prompts import common_browser_system_prompt
import base64
import io
from PIL import Image

console = Console()

# Global model variables
model = None
tokenizer = None
processor = None

def load_finetuned_model():
    """Load the finetuned UI-TARS model with LoRA adapter."""
    global model, tokenizer, processor
    
    if model is not None:
        return  # Already loaded
    
    base_model_path = "ByteDance-Seed/UI-TARS-1.5-7B"
    adapter_path = "."  # Current directory where ui-tars_finetuned.safetensors is located
    
    console.print("[yellow]Loading finetuned UI-TARS model...[/yellow]")
    
    # Load tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    
    # Load base model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        torch_dtype=torch.bfloat16
    )
    
    model.eval()
    console.print("[green]Finetuned UI-TARS model loaded successfully![/green]")

def base64_to_image(image_base64: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    # Remove data URL prefix if present
    if "data:image" in image_base64:
        image_base64 = image_base64.split(",")[1]
    
    # Decode base64 to image
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    return image

def generate_response(messages: list, image: Image.Image, max_new_tokens: int = 1000) -> str:
    """Generate response using the finetuned model."""
    global model, tokenizer, processor
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Process inputs
    inputs = processor(
        text=[text], 
        images=[image],
        padding=True, 
        return_tensors="pt"
    )
    
    # Move inputs to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Use greedy decoding for consistent results
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response.strip()


class Action(BaseModel):
    action: str
    args: dict[str, str]
    reasoning: str


def ui_tars_finetuned_call(
    prompt: str, history: list[tuple[str, str]], image_base64: str
) -> Action:
    """
    Main function to call the finetuned UI-TARS model.
    
    Args:
        prompt: The instruction/task to perform
        history: List of previous (reasoning, action) tuples
        image_base64: Base64 encoded screenshot image
        
    Returns:
        Action object with action type, arguments, and reasoning
    """
    # Load model if not already loaded
    load_finetuned_model()
    
    # Convert base64 image to PIL Image
    image = base64_to_image(image_base64)
    
    # Build messages in the format expected by the model
    messages = [
        {
            "role": "system",
            "content": common_browser_system_prompt.format(
                language="English", instruction=prompt
            ),
        }
    ]
    
    # Add history
    for reasoning, action in history:
        messages.append({
            "role": "assistant",
            "content": f"Thought: {reasoning}\nAction: {action}"
        })
    
    # Add current user message with image
    messages.append({
        "role": "user", 
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    })
    
    # Generate response
    response_text = generate_response(messages, image, max_new_tokens=1000)

    try:
        response_text = "temp " + response_text
        reasoning = response_text.split("Thought: ")[1].split("\nAction: ")[0].strip()
        action = response_text.split("Action: ")[1].strip()
    except Exception as e:
        console.print(
            f"[red]Error parsing UI-TARS response: {e}.[/red]\n[red]Response:[/red] {response_text}"
        )
        return Action(action="error", args={}, reasoning=response_text)

    console.print(f"[blue]UI TARS Finetuned Action:[/blue] {action}")
    console.print(f"[blue]UI TARS Finetuned Reasoning:[/blue] {reasoning}")
    return parse_action(action, reasoning)


def parse_action(action: str, reasoning: str) -> Action:
    """Parse UI-TARS action into our Action format."""
    if action.startswith("click"):
        if "point='" in action:
            coords = action.split("point='")[1].split("'")[0]
        else:
            coords = action.split("start_box='")[1].split("'")[0]
        x, y = map(int, coords.strip("()").split(","))
        return Action(
            action="click", args={"x": str(x), "y": str(y)}, reasoning=reasoning
        )
    elif action.startswith("left_double"):
        coords = action.split("start_box='")[1].split("'")[0]
        x, y = map(int, coords.strip("()").split(","))
        return Action(
            action="left_double", args={"x": str(x), "y": str(y)}, reasoning=reasoning
        )
    elif action.startswith("right_single"):
        coords = action.split("start_box='")[1].split("'")[0]
        x, y = map(int, coords.strip("()").split(","))
        return Action(
            action="right_single", args={"x": str(x), "y": str(y)}, reasoning=reasoning
        )
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
            reasoning=reasoning,
        )
    elif action.startswith("hotkey"):
        key = action.split("key='")[1].split("'")[0]
        return Action(action="hotkey", args={"key": key}, reasoning=reasoning)
    elif action.startswith("type"):
        content = action.split("content='")[1].split("'")[0]
        return Action(action="type", args={"content": content}, reasoning=reasoning)
    elif action.startswith("scroll"):
        if "point='" in action:
            coords = action.split("point='")[1].split("'")[0]
            x, y = map(int, coords.strip("()").split(","))
        else:
            coords = action.split("start_box='")[1].split("'")[0]
            x, y = map(int, coords.strip("()").split(","))
        direction = action.split("direction='")[1].split("'")[0]
        return Action(
            action="scroll",
            args={"x": str(x), "y": str(y), "direction": direction},
            reasoning=reasoning,
        )
    elif action.startswith("wait"):
        return Action(action="wait", args={}, reasoning=reasoning)
    elif action.startswith("finished"):
        content = action.split("content='")[1].split("'")[0]
        return Action(action="finished", args={"content": content}, reasoning=reasoning)
    elif action.startswith("goto_url"):
        url = action.split("url='")[1].split("'")[0]
        return Action(action="goto_url", args={"url": url}, reasoning=reasoning)
    else:
        raise ValueError(f"Invalid action: {action}")


# Alternative interface for direct use (same as original)
def ui_tars_call(prompt: str, history: list[tuple[str, str]], image_base64: str) -> Action:
    """Alias for backward compatibility."""
    return ui_tars_finetuned_call(prompt, history, image_base64) 

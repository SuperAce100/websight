#!/usr/bin/env python3
"""
Inference script for the trained UI-TARS model with LoRA adapter.
Usage: python inference.py --image_path "path/to/image.png" --query "Click the button"
"""

import argparse
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

class UITarsInference:
    def __init__(self, base_model_path="ByteDance-Seed/UI-TARS-1.5-7B", 
                 adapter_path="output/ui_tars_waveui_lora"):
        """
        Initialize the UI-TARS model with LoRA adapter for inference.
        
        Args:
            base_model_path: Path to the base model
            adapter_path: Path to the trained LoRA adapter
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        print("Loading tokenizer and processor...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, 
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        
        print("Loading base model...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
            torch_dtype=torch.bfloat16
        )
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def predict(self, image_path, query_text, max_new_tokens=512):
        """
        Predict the UI action based on image and text query.
        
        Args:
            image_path: Path to the screenshot image
            query_text: Text description of the action to perform
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated response (typically bounding box coordinates)
        """
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        
        # Create conversation format
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query_text}
                ]
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            text=[text], 
            images=[image],
            padding=True, 
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Use greedy decoding for consistent results
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()

def main():
    parser = argparse.ArgumentParser(description="Run inference with UI-TARS model")
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to the screenshot image")
    parser.add_argument("--query", type=str, required=True,
                       help="Text query describing the UI action")
    parser.add_argument("--base_model", type=str, 
                       default="ByteDance-Seed/UI-TARS-1.5-7B",
                       help="Base model path")
    parser.add_argument("--adapter_path", type=str,
                       default="output/ui_tars_waveui_lora",
                       help="LoRA adapter path")
    parser.add_argument("--max_tokens", type=int, default=512,
                       help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    # Initialize model
    ui_model = UITarsInference(
        base_model_path=args.base_model,
        adapter_path=args.adapter_path
    )
    
    # Run inference
    print(f"\nImage: {args.image_path}")
    print(f"Query: {args.query}")
    print("Generating response...")
    
    response = ui_model.predict(
        image_path=args.image_path,
        query_text=args.query,
        max_new_tokens=args.max_tokens
    )
    
    print(f"\nResponse: {response}")

# Example usage as a module
def quick_inference(image_path, query, adapter_path="output/ui_tars_waveui_lora"):
    """Quick inference function for programmatic use"""
    model = UITarsInference(adapter_path=adapter_path)
    return model.predict(image_path, query)

if __name__ == "__main__":
    main() 
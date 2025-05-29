#!/usr/bin/env python3
"""
Quick test of vision functionality
"""

import base64
import os
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

def quick_vision_test():
    """Simple vision test."""
    
    print("🧪 Quick Vision Test")
    print("=" * 30)
    
    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ No API key found")
        return False
    
    # Simple 1x1 red pixel test image
    test_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    
    try:
        print("👁️ Testing vision with simple image...")
        
        response = client.chat.completions.create(
            model="meta-llama/llama-4-maverick:free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color do you see?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{test_image}"}}
                    ]
                }
            ],
            max_tokens=50
        )
        
        result = response.choices[0].message.content
        print(f"✅ Vision works! Response: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vision test failed: {str(e)}")
        return False


if __name__ == "__main__":
    quick_vision_test() 
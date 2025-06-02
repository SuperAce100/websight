#!/usr/bin/env python3
"""
Simple test to check if Llama 4 Maverick vision works
"""

import base64
import os
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

def test_vision():
    """Test basic vision functionality with a simple image."""
    
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found")
        return False
    
    print("🧪 Testing Llama 4 Maverick Vision...")
    
    # Create a simple test image (1x1 red pixel)
    # This is a base64 encoded 1x1 red PNG image
    test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    
    try:
        # Test 1: Simple text message first
        print("📝 Test 1: Simple text message...")
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash-preview-05-20",
            messages=[
                {"role": "user", "content": "Hello! Can you see images?"}
            ],
            max_tokens=100
        )
        
        print(f"✅ Text response: {response.choices[0].message.content}")
        
        # Test 2: Image message
        print("\n👁️ Test 2: Image message...")
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash-preview-05-20",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What color is this pixel?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{test_image_b64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=50
        )
        
        print(f"✅ Vision response: {response.choices[0].message.content}")
        print("🎉 Vision test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Vision test failed: {str(e)}")
        return False


if __name__ == "__main__":
    test_vision() 
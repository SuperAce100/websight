#!/usr/bin/env python3
"""
Simple test to verify Llama 4 Maverick vision capabilities
"""

import base64
import os
from browser import Browser
from communicators.llms import llm_call_messages
from pydantic import BaseModel
from typing import Optional, List


class SimpleAction(BaseModel):
    action: str  # "click", "type", "scroll", "navigate", "wait", "complete"
    description: str  # What the action will do
    x: Optional[int] = None  # X coordinate for click/scroll
    y: Optional[int] = None  # Y coordinate for click/scroll  
    text: Optional[str] = None  # Text to type or URL to navigate


def test_llama_vision():
    """Test if Llama 4 Maverick can see and analyze screenshots."""
    
    print("🧪 Testing Llama 4 Maverick Vision Capabilities")
    print("=" * 50)
    
    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not found in environment")
        return False
    
    # Create browser and take screenshot
    print("📸 Opening browser and taking screenshot...")
    browser = Browser()
    
    try:
        # Go to a simple page
        browser.goto_url("https://www.google.com")
        
        # Take screenshot
        os.makedirs("data/screenshots", exist_ok=True)
        screenshot_path = "data/screenshots/vision_test.png"
        browser.take_screenshot(screenshot_path)
        
        # Read screenshot as base64
        with open(screenshot_path, "rb") as img_file:
            screenshot_b64 = base64.b64encode(img_file.read()).decode()
        
        print(f"📊 Screenshot saved: {screenshot_path}")
        print(f"🔢 Image size: {len(screenshot_b64)} characters")
        
        # Test 1: Simple description (no structured output)
        print("\n🔍 Test 1: Simple image description...")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "What do you see in this screenshot? Describe the main elements briefly."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}
                    }
                ]
            }
        ]
        
        try:
            response = llm_call_messages(
                messages=messages,
                model="meta-llama/llama-4-maverick:free"
            )
            print(f"✅ Simple description: {response[:200]}...")
            
        except Exception as e:
            print(f"❌ Simple description failed: {str(e)}")
            return False
        
        # Test 2: Structured output
        print("\n🔍 Test 2: Structured action suggestion...")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Look at this Google homepage screenshot. I want to search for something. What should I do? Give me ONE action - either click on the search box or type in it. If clicking, provide x and y coordinates."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}
                    }
                ]
            }
        ]
        
        try:
            action = llm_call_messages(
                messages=messages,
                response_format=SimpleAction,
                model="meta-llama/llama-4-maverick:free"
            )
            print(f"✅ Structured action: {action}")
            
        except Exception as e:
            print(f"❌ Structured action failed: {str(e)}")
            print("🔄 Trying without structured output...")
            
            # Fallback: try without structured output
            try:
                response = llm_call_messages(
                    messages=messages,
                    model="meta-llama/llama-4-maverick:free"
                )
                print(f"✅ Fallback response: {response[:200]}...")
            except Exception as e2:
                print(f"❌ Fallback also failed: {str(e2)}")
                return False
        
        print("\n🎉 Vision tests completed!")
        return True
        
    finally:
        browser.close()


def test_simple_agent():
    """Test a very simple agent that just takes one screenshot and suggests one action."""
    
    print("\n🤖 Testing Simple Agent")
    print("=" * 30)
    
    browser = Browser()
    
    try:
        # Navigate to Google
        browser.goto_url("https://www.google.com")
        
        # Take screenshot
        os.makedirs("data/screenshots", exist_ok=True)
        screenshot_path = "data/screenshots/agent_test.png"
        browser.take_screenshot(screenshot_path)
        
        with open(screenshot_path, "rb") as img_file:
            screenshot_b64 = base64.b64encode(img_file.read()).decode()
        
        # Ask for action
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "I want to search for 'Llama 4 Maverick'. What should I do first? Look at this Google homepage and tell me what to click or type. Be specific about coordinates if clicking."
                    },
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}
                    }
                ]
            }
        ]
        
        # Try structured output first
        try:
            action = llm_call_messages(
                messages=messages,
                response_format=SimpleAction,
                model="meta-llama/llama-4-maverick:free"
            )
            print(f"🎯 Structured action: {action}")
            
            # Try to execute the action
            if action.action == "click" and action.x and action.y:
                print(f"🖱️ Clicking at ({action.x}, {action.y})")
                browser.click(action.x, action.y)
                
            elif action.action == "type" and action.text:
                print(f"⌨️ Typing: {action.text}")
                browser.type(action.text)
                
        except Exception as e:
            print(f"❌ Structured action failed: {str(e)}")
            print("🔄 Getting text response instead...")
            
            response = llm_call_messages(
                messages=messages,
                model="meta-llama/llama-4-maverick:free"
            )
            print(f"💬 AI Response: {response[:300]}...")
        
        # Take another screenshot to see result
        screenshot_path_2 = "data/screenshots/agent_test_after.png"
        browser.take_screenshot(screenshot_path_2)
        print(f"📸 After action screenshot: {screenshot_path_2}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple agent test failed: {str(e)}")
        return False
        
    finally:
        browser.close()


if __name__ == "__main__":
    try:
        # Test vision capabilities
        vision_works = test_llama_vision()
        
        if vision_works:
            # Test simple agent
            test_simple_agent()
        else:
            print("❌ Vision tests failed, skipping agent test")
            
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
    except Exception as e:
        print(f"💥 Unexpected error: {str(e)}") 
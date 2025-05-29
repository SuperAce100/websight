#!/usr/bin/env python3
"""
Simplified vision-based web agent focusing on image processing
"""

import base64
import os
from browser import Browser
from communicators.llms import llm_call_messages
from pydantic import BaseModel
from typing import Optional


class WebAction(BaseModel):
    action: str  # "click", "type", "scroll", "navigate", "wait", "complete"
    reasoning: str  # Why this action was chosen
    x: Optional[int] = None  # X coordinate for click/scroll
    y: Optional[int] = None  # Y coordinate for click/scroll
    text: Optional[str] = None  # Text to type or URL to navigate


def simple_web_agent_test():
    """Test a simple web agent that takes screenshots and decides actions."""
    
    print("🤖 Simple Vision Web Agent Test")
    print("=" * 40)
    
    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not found")
        return False
    
    browser = Browser()
    
    try:
        # Navigate to a simple page
        print("🌐 Opening Google...")
        browser.goto_url("https://www.google.com")
        
        # Take screenshot
        os.makedirs("data/screenshots", exist_ok=True)
        screenshot_path = "data/screenshots/google_homepage.png"
        browser.take_screenshot(screenshot_path)
        
        print(f"📸 Screenshot saved: {screenshot_path}")
        
        # Read screenshot
        with open(screenshot_path, "rb") as img_file:
            screenshot_b64 = base64.b64encode(img_file.read()).decode()
        
        print(f"🔢 Image size: {len(screenshot_b64)} characters")
        
        # Ask the vision model what to do
        print("👁️ Asking Llama 4 Maverick what to do...")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "I want to search for 'python tutorial'. Look at this Google homepage screenshot and tell me what I should do. Give me ONE specific action with exact coordinates if clicking."
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
                response_format=WebAction,
                model="meta-llama/llama-4-maverick:free"
            )
            
            print(f"🎯 AI Decision: {action}")
            
            # Execute the action
            if action.action == "click" and action.x and action.y:
                print(f"🖱️ Clicking at ({action.x}, {action.y}) - {action.reasoning}")
                browser.click(action.x, action.y)
                
            elif action.action == "type" and action.text:
                print(f"⌨️ Typing: {action.text} - {action.reasoning}")
                browser.type(action.text)
                
            else:
                print(f"📝 Action: {action.action} - {action.reasoning}")
                
        except Exception as e:
            print(f"❌ Structured action failed: {str(e)}")
            print("🔄 Getting text response instead...")
            
            try:
                response = llm_call_messages(
                    messages=messages,
                    model="meta-llama/llama-4-maverick:free"
                )
                print(f"💬 AI Response: {response[:300]}...")
                
                # Simple fallback: if response mentions clicking search box, click center of screen
                if "search" in response.lower() and "click" in response.lower():
                    print("🖱️ Fallback: Clicking center of search area (640, 360)")
                    browser.click(640, 360)
                    
            except Exception as e2:
                print(f"❌ Text response also failed: {str(e2)}")
                return False
        
        # Take another screenshot to see the result
        screenshot_path_2 = "data/screenshots/google_after_action.png"
        browser.take_screenshot(screenshot_path_2)
        print(f"📸 After action screenshot: {screenshot_path_2}")
        
        print("✅ Simple agent test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Simple agent test failed: {str(e)}")
        return False
        
    finally:
        browser.close()


if __name__ == "__main__":
    try:
        simple_web_agent_test()
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"💥 Unexpected error: {str(e)}") 
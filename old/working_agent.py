#!/usr/bin/env python3
"""
Working vision-based web agent - simplified and functional
"""

import base64
import os
import sys
import time
from openai import OpenAI
from browser import Browser

import dotenv

dotenv.load_dotenv()


def working_agent():
    """A simple working agent that can see and interact with web pages."""
    
    print("🤖 Working Vision Web Agent")
    print("=" * 35)
    
    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ No OpenRouter API key found!")
        print("   Create a .env file with: OPENROUTER_API_KEY=your_key_here")
        return False
    
    print("🔑 API key found")
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    
    # Initialize browser
    print("🌐 Starting browser...")
    browser = Browser()
    
    try:
        # Navigate to Google
        print("📍 Going to Google...")
        browser.goto_url("https://www.google.com")
        
        # Take screenshot
        os.makedirs("data/screenshots", exist_ok=True)
        screenshot_path = "data/screenshots/google_page.png"
        browser.take_screenshot(screenshot_path)
        print(f"📸 Screenshot saved: {screenshot_path}")
        
        # Read screenshot as base64
        with open(screenshot_path, "rb") as img_file:
            screenshot_b64 = base64.b64encode(img_file.read()).decode()
        
        # Ask AI what it sees
        print("👁️ Asking AI to describe what it sees...")
        
        response = client.chat.completions.create(
            model="meta-llama/llama-4-maverick:free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe what you see in this screenshot. What can someone do on this page?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}
                        }
                    ]
                }
            ],
            max_tokens=200
        )
        
        description = response.choices[0].message.content
        print(f"🔍 AI Description: {description}")
        
        # Ask AI for a specific action
        print("\n🎯 Asking AI for an action...")
        
        response = client.chat.completions.create(
            model="meta-llama/llama-4-maverick:free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "I want to search for 'python programming'. Looking at this Google page, tell me exactly what I should do. Give me coordinates if I should click somewhere, or tell me if I should type something."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}
                        }
                    ]
                }
            ],
            max_tokens=150
        )
        
        action_advice = response.choices[0].message.content
        print(f"💡 AI Action Advice: {action_advice}")
        
        # Simple action execution based on the advice
        if "search" in action_advice.lower():
            print("\n🖱️ AI suggests searching - clicking on center of page to find search box")
            browser.click(640, 360)  # Click center where search box usually is
            
            time.sleep(2)
            
            print("⌨️ Typing search query...")
            browser.type("python programming")
            
            time.sleep(1)
            
            print("🔍 Pressing Enter to search...")
            browser.hotkey("enter")
            
            time.sleep(3)
            
            # Take another screenshot to see results
            result_path = "data/screenshots/search_results.png"
            browser.take_screenshot(result_path)
            print(f"📸 Search results screenshot: {result_path}")
            
        print("\n✅ Agent completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Agent failed: {str(e)}")
        return False
        
    finally:
        print("🔒 Closing browser...")
        browser.close()


if __name__ == "__main__":
    try:
        working_agent()
    except KeyboardInterrupt:
        print("\n⏹️ Agent stopped by user")
    except Exception as e:
        print(f"💥 Unexpected error: {str(e)}") 
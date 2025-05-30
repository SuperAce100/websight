#!/usr/bin/env python3
"""
WebSight Demo - Interactive Vision-Based Web Agent

This demo script shows how to use the WebAgent to perform web automation tasks
using natural language instructions and vision-based decision making.
"""

import os
from old.agent import WebAgent


def interactive_demo():
    """Interactive demo where user can input custom tasks."""
    
    print("🌟 Welcome to WebSight - Vision-Based Web Agent Demo!")
    print("=" * 60)
    print()
    print("💡 This agent can perform web automation tasks using natural language.")
    print("   It uses Llama 4 Maverick to understand screenshots and make decisions.")
    print()
    print("📝 Example tasks you can try:")
    print("   • 'Go to google.com and search for python tutorials'")
    print("   • 'Navigate to github.com and find trending Python repositories'")
    print("   • 'Visit apple.com and find the latest iPhone pricing'")
    print("   • 'Go to wikipedia.org and search for artificial intelligence'")
    print()
    
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY environment variable not set!")
        print("   Please create a .env file with your OpenRouter API key.")
        return
    
    # Initialize agent
    agent = WebAgent()
    
    try:
        while True:
            print("🎯 Enter your task (or 'quit' to exit):")
            user_task = input("> ").strip()
            
            if user_task.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_task:
                print("⚠️ Please enter a task description.")
                continue
            
            print("\n" + "="*60)
            print(f"🚀 Executing task: {user_task}")
            print("="*60)
            
            try:
                result = agent.execute_task(user_task)
                print("\n" + "="*60)
                print(f"✅ Task completed: {result}")
                print("="*60)
                
            except Exception as e:
                print(f"❌ Task failed: {str(e)}")
            
            print("\nTask execution finished. Ready for next task!\n")
    
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user.")
    
    finally:
        agent.close()


def preset_demo():
    """Demo with preset tasks to showcase capabilities."""
    
    print("🌟 WebSight Preset Demo")
    print("="*40)
    
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY environment variable not set!")
        return
    
    # Preset tasks to demonstrate capabilities
    tasks = [
        {
            "name": "Google Search",
            "description": "Go to google.com and search for 'web automation with python'",
            "explanation": "Demonstrates navigation and search functionality"
        },
        {
            "name": "GitHub Exploration", 
            "description": "Navigate to github.com and browse trending repositories",
            "explanation": "Shows ability to navigate and explore dynamic content"
        },
        {
            "name": "Product Research",
            "description": "Visit apple.com and find information about the latest iPhone",
            "explanation": "Demonstrates product page navigation and information extraction"
        }
    ]
    
    agent = WebAgent()
    
    try:
        for i, task in enumerate(tasks, 1):
            print(f"\n📋 Demo {i}/{len(tasks)}: {task['name']}")
            print(f"📝 Task: {task['description']}")
            print(f"💡 Purpose: {task['explanation']}")
            
            response = input("\n▶️ Execute this task? (y/n/q): ").strip().lower()
            
            if response == 'q':
                print("Demo stopped by user.")
                break
            elif response != 'y':
                print("Skipping task...")
                continue
            
            print("\n" + "="*60)
            try:
                result = agent.execute_task(task['description'])
                print(f"\n✅ Demo {i} completed: {result}")
            except Exception as e:
                print(f"❌ Demo {i} failed: {str(e)}")
            print("="*60)
    
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user.")
    
    finally:
        agent.close()


def main():
    """Main demo entry point."""
    
    print("🤖 WebSight Vision-Based Web Agent")
    print("Choose demo mode:")
    print("1. Interactive mode (enter custom tasks)")
    print("2. Preset demo (predefined tasks)")
    print("3. Quit")
    
    while True:
        choice = input("\nSelect option (1/2/3): ").strip()
        
        if choice == '1':
            interactive_demo()
            break
        elif choice == '2':
            preset_demo()
            break
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("⚠️ Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main() 
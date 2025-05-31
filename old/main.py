#!/usr/bin/env python3
"""
WebSight - Vision-Based Web Agent

Quick start example showing how to use the WebAgent for automated web tasks.
"""

from old.agent import WebAgent

def main():
    """Quick start example."""
    
    # Initialize the vision-based web agent
    agent = WebAgent()
    
    try:
        # Example task - modify this to test different scenarios
        task = "Give me the score of the most recent NBA game."
        
        print(f"🚀 Executing task: {task}")
        result = agent.execute_task(task)
        
        print(f"\n✅ Task completed: {result}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    finally:
        # Always clean up browser resources
        agent.close()

if __name__ == "__main__":
    main()
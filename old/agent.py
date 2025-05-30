import base64
import time
from typing import Optional, List, Literal
from pydantic import BaseModel

from browser import Browser
from task import Task
from communicators.llms import llm_call, llm_call_messages


class PlanStep(BaseModel):
    step_number: int
    description: str
    action_type: Literal["navigate", "click", "type", "scroll", "wait", "screenshot"]
    details: str


class TaskPlan(BaseModel):
    goal: str
    steps: List[PlanStep]
    estimated_duration: str


class ActionDecision(BaseModel):
    action: Literal["click", "type", "scroll", "navigate", "hotkey", "wait", "complete"]
    coordinates: Optional[List[int]] = None  # [x, y] for click/scroll actions
    content: Optional[str] = None  # Text to type or URL to navigate
    reasoning: str
    confidence: float  # 0.0 to 1.0
    next_step_description: str
    is_task_complete: bool = False


class ProgressUpdate(BaseModel):
    current_step: int
    total_steps: int
    completed_actions: List[str]
    remaining_actions: List[str]
    current_state_description: str
    is_task_complete: bool
    next_planned_action: str


class WebAgent:
    """Vision-based web automation agent using Llama 4 Maverick."""
    
    def __init__(self):
        self.browser = Browser()
        self.current_task: Optional[Task] = None
        
    def execute_task(self, user_instruction: str) -> str:
        """Main entry point for task execution."""
        
        try:
            # Step 1: Generate initial plan
            print(f"🎯 Planning task: {user_instruction}")
            plan = self._generate_plan(user_instruction)
            print(f"📋 Generated plan with {len(plan.steps)} steps")
            
            # Step 2: Initialize task
            self.current_task = Task(
                task_description=user_instruction,
                task_plan=plan.model_dump_json(indent=2)
            )
            
            # Step 3: Start browser and take initial screenshot
            print("🌐 Opening browser...")
            self.browser.goto_url("about:blank")  # Start with blank page
            self._take_screenshot("Initial browser state")
            
            # Step 4: Execute plan step by step
            return self._execute_plan_iteratively(plan)
            
        except Exception as e:
            error_msg = f"Task execution failed: {str(e)}"
            print(f"❌ {error_msg}")
            if self.current_task:
                self.current_task.update_current_state(error_msg)
                self.current_task.save_to_file()
            return error_msg
    
    def _generate_plan(self, instruction: str) -> TaskPlan:
        """Generate detailed execution plan using Llama 4 Maverick."""
        
        prompt = f"""
        Create a detailed plan to accomplish this web automation task: {instruction}
        
        Consider these available browser actions:
        - navigate: Go to a specific URL
        - click: Click at specific screen coordinates
        - type: Type text into input fields
        - scroll: Scroll the page (up/down/left/right)
        - wait: Wait for page loads or elements
        - screenshot: Take screenshots for analysis
        - hotkey: Use keyboard shortcuts (ctrl+c, ctrl+v, etc.)
        
        Break down the task into specific, actionable steps that can be executed sequentially.
        Each step should be clear and measurable.
        """
        
        return llm_call(
            prompt=prompt,
            system_prompt="You are a web automation expert. Create detailed, step-by-step plans for browser automation tasks. Focus on concrete, executable actions.",
            response_format=TaskPlan,
            model="meta-llama/llama-4-maverick:free"
        )
    
    def _execute_plan_iteratively(self, plan: TaskPlan) -> str:
        """Execute the plan step by step with vision feedback."""
        
        for step_num, step in enumerate(plan.steps, 1):
            print(f"\n🔄 Step {step_num}/{len(plan.steps)}: {step.description}")
            
            # Take screenshot and analyze current state
            screenshot_path = self._take_screenshot(f"Before step {step_num}")
            
            # Get action decision from vision model
            print("👁️ Analyzing screenshot and deciding action...")
            action = self._decide_next_action(step, screenshot_path)
            
            print(f"🎬 Action: {action.action}")
            if action.reasoning:
                print(f"💭 Reasoning: {action.reasoning}")
            
            # Execute the action
            self._execute_action(action)
            
            # Update task progress
            self.current_task.add_step_taken(f"Step {step_num}: {action.reasoning}")
            self.current_task.update_current_state(action.next_step_description)
            
            # Check if task is complete
            if action.action == "complete" or action.is_task_complete:
                return self._finalize_task("✅ Task completed successfully")
            
            # Brief wait between actions
            time.sleep(2)
        
        return self._finalize_task("✅ Plan execution completed")
    
    def _decide_next_action(self, current_step: PlanStep, screenshot_path: str) -> ActionDecision:
        """Use vision model to decide next action based on screenshot."""
        
        # Read screenshot as base64
        with open(screenshot_path, "rb") as img_file:
            screenshot_b64 = base64.b64encode(img_file.read()).decode()
        
        prompt = f"""
        You are viewing a web page screenshot. Your current goal is: {current_step.description}
        
        Overall task context:
        - Main task: {self.current_task.data.task}
        - Current step: {current_step.description}
        - Steps completed so far: {self.current_task.data.steps_taken}
        - Current state: {self.current_task.data.current_state}
        
        Available actions you can choose from:
        - click: Click at specific coordinates [x, y] - use for buttons, links, form fields
        - type: Type text content - use after clicking in input fields
        - scroll: Scroll page in direction - provide coordinates [x, y] and direction in content
        - navigate: Go to a URL - provide full URL in content field
        - hotkey: Press key combinations - provide key combo like "ctrl c" in content
        - wait: Wait for page to load or settle
        - complete: Mark task as finished when goal is achieved
        
        Analyze the screenshot carefully and decide what action to take next.
        Be specific with coordinates for clickable elements.
        Provide clear reasoning for your decision.
        """
        
        # Create messages for vision model
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}
                    }
                ]
            }
        ]
        
        return llm_call_messages(
            messages=messages,
            response_format=ActionDecision,
            model="meta-llama/llama-4-maverick:free"
        )
    
    def _execute_action(self, action: ActionDecision):
        """Execute the decided action using browser automation."""
        
        try:
            if action.action == "click" and action.coordinates:
                x, y = action.coordinates[0], action.coordinates[1]
                print(f"🖱️ Clicking at ({x}, {y})")
                self.browser.click(x, y)
                
            elif action.action == "type" and action.content:
                print(f"⌨️ Typing: {action.content}")
                self.browser.type(action.content)
                
            elif action.action == "scroll":
                x, y = action.coordinates or [500, 500]
                direction = action.content or "down"
                print(f"📜 Scrolling {direction} at ({x}, {y})")
                self.browser.scroll(x, y, direction)
                
            elif action.action == "navigate" and action.content:
                print(f"🔗 Navigating to: {action.content}")
                self.browser.goto_url(action.content)
                
            elif action.action == "hotkey" and action.content:
                print(f"⌨️ Hotkey: {action.content}")
                self.browser.hotkey(action.content)
                
            elif action.action == "wait":
                print("⏳ Waiting...")
                self.browser.wait()
                
            elif action.action == "complete":
                print("✅ Task marked as complete")
                pass  # Task completion handled by caller
                
        except Exception as e:
            print(f"❌ Action execution failed: {str(e)}")
            raise
    
    def _take_screenshot(self, description: str) -> str:
        """Take screenshot and update task state."""
        self.current_task.take_screenshot_with_browser(self.browser, description)
        return self.current_task.data.current_state_screenshot_path
    
    def _finalize_task(self, result: str) -> str:
        """Finalize task and cleanup."""
        print(f"🏁 {result}")
        self.current_task.update_current_state(result)
        task_file = self.current_task.save_to_file()
        print(f"💾 Task saved to: {task_file}")
        return result
    
    def close(self):
        """Cleanup resources."""
        if self.browser:
            print("🔒 Closing browser...")
            self.browser.close()


def main():
    """Example usage of the WebAgent."""
    
    # Initialize agent
    agent = WebAgent()
    
    try:
        # Example tasks
        examples = [
            "Go to google.com and search for 'python web automation'",
            "Navigate to github.com and browse trending repositories", 
            "Visit apple.com and find information about the latest iPhone"
        ]
        
        # You can modify this to use any task
        task = examples[0]  # Change index to try different examples
        
        print(f"🚀 Starting WebAgent with task: {task}")
        result = agent.execute_task(task)
        print(f"\n📊 Final result: {result}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Task interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
    finally:
        agent.close()


if __name__ == "__main__":
    main() 
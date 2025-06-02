import base64
import json
import os
import random
import time
from pydantic import BaseModel


class TaskData(BaseModel):
    task: str
    task_plan: str
    steps_taken: str
    steps_remaining: str
    current_state: str
    current_state_screenshot_path: str


class Task:
    def __init__(self, task_description: str = "", task_plan: str = "", steps_remaining: str = ""):
        """Initialize a new task with optional description and plan."""
        self.data = TaskData(
            task=task_description,
            task_plan=task_plan,
            steps_taken="No steps taken yet.",
            steps_remaining=steps_remaining,
            current_state="Have not started yet.",
            current_state_screenshot_path=""
        )
        self.task_id = self._generate_task_id()
        self.created_at = time.time()
        
    def _generate_task_id(self) -> str:
        """Generate a unique task ID."""
        return f"task_{int(time.time())}_{random.randint(1000, 9999)}"
    
    def update_task(self, task_description: str):
        """Update the task description."""
        self.data.task = task_description
    
    def update_plan(self, task_plan: str):
        """Update the task plan."""
        self.data.task_plan = task_plan
    
    def add_step_taken(self, step: str):
        """Add a step to the steps taken."""
        if self.data.steps_taken:
            self.data.steps_taken += f"\n{step}"
        else:
            self.data.steps_taken = step
    
    def update_steps_remaining(self, steps: str):
        """Update the remaining steps."""
        self.data.steps_remaining = steps
    
    def update_current_state(self, state: str):
        """Update the current state description."""
        self.data.current_state = state
    
    def set_screenshot_path(self, path: str):
        """Set the path to the current state screenshot."""
        self.data.current_state_screenshot_path = path
    
    def take_screenshot_with_browser(self, browser, description: str = ""):
        """Take a screenshot using a browser instance and update the task state."""
        base_path = "data/screenshots"
        os.makedirs(base_path, exist_ok=True)
        
        timestamp = int(time.time())
        screenshot_path = f"{base_path}/task_{self.task_id}_{timestamp}.png"
        
        browser.take_screenshot(screenshot_path)
        self.set_screenshot_path(screenshot_path)
        
        if description:
            self.update_current_state(description)
    
    def get_progress_summary(self) -> dict:
        """Get a summary of task progress."""
        return {
            "task_id": self.task_id,
            "task": self.data.task,
            "plan": self.data.task_plan,
            "steps_completed": len(self.data.steps_taken.split('\n')) if self.data.steps_taken else 0,
            "current_state": self.data.current_state,
            "has_screenshot": bool(self.data.current_state_screenshot_path),
            "created_at": self.created_at
        }
    
    def save_to_file(self, filepath: str = None):
        """Save task data to a JSON file."""
        if filepath is None:
            base_path = "data/tasks"
            os.makedirs(base_path, exist_ok=True)
            filepath = f"{base_path}/task_{self.task_id}.json"
        
        task_export = {
            "task_id": self.task_id,
            "created_at": self.created_at,
            "data": self.data.dict()
        }
        
        with open(filepath, 'w') as f:
            json.dump(task_export, f, indent=2)
        
        return filepath
    
    @classmethod
    def load_from_file(cls, filepath: str):
        """Load task data from a JSON file."""
        with open(filepath, 'r') as f:
            task_export = json.load(f)
        
        task = cls()
        task.task_id = task_export["task_id"]
        task.created_at = task_export["created_at"]
        task.data = TaskData(**task_export["data"])
        
        return task
    
    def get_data(self) -> TaskData:
        """Get the current task data."""
        return self.data
    
    def __str__(self) -> str:
        """String representation of the task."""
        return f"Task({self.task_id}): {self.data.task[:50]}{'...' if len(self.data.task) > 50 else ''}"


def main():
    """Example usage of the Task class."""
    # Create a new task
    task = Task(
        task_description="Navigate to Apple iPhone page and take screenshots",
        task_plan="1. Open browser\n2. Navigate to apple.com/iphone\n3. Take initial screenshot\n4. Scroll down\n5. Take final screenshot"
    )
    
    print(f"Created task: {task}")
    print(f"Progress summary: {task.get_progress_summary()}")
    
    # Simulate adding steps
    task.add_step_taken("Opened browser successfully")
    task.add_step_taken("Navigated to apple.com/iphone")
    task.update_current_state("Currently viewing iPhone main page")
    
    # Save task
    saved_path = task.save_to_file()
    print(f"Task saved to: {saved_path}")
    
    # Load task back
    loaded_task = Task.load_from_file(saved_path)
    print(f"Loaded task: {loaded_task}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Demo Evaluation Script for Websight

This script runs a small subset of evaluation tasks to test the system.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv
from rich.console import Console

# Load environment variables
load_dotenv()

console = Console()
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

EVALUATION_MODEL = "openai/gpt-4.1-nano"

def evaluate_single_response(question: str, expected_answer: str, actual_output: str) -> dict:
    """Evaluate a single response using the lightweight model"""
    
    evaluation_prompt = f"""
    You are evaluating whether a web automation agent's response correctly answers a given task.

    TASK: {question}
    EXPECTED ANSWER: {expected_answer}
    ACTUAL OUTPUT: {actual_output}

    Your job is to determine if the actual output correctly addresses the task, even if it's not identical to the expected answer.

    Consider these criteria:
    1. Does the actual output contain the key information requested in the task?
    2. Is the information factually correct and relevant?
    3. Does it address the core question, even if phrased differently?
    4. For real-time data (prices, ratings, etc.), is the format and type of information correct?

    The responses don't need to be identical - they should share the same general idea and answer the same question, but specifics can differ (especially for real-time data).

    Respond with ONLY a JSON object in this exact format:
    {{
        "is_correct": true/false,
        "confidence": 0.0-1.0,
        "reasoning": "Brief explanation of your decision"
    }}
    """

    try:
        response = client.chat.completions.create(
            model=EVALUATION_MODEL,
            messages=[
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        evaluation_text = response.choices[0].message.content.strip()
        evaluation_data = json.loads(evaluation_text)
        
        return {
            "is_correct": evaluation_data.get("is_correct", False),
            "confidence": evaluation_data.get("confidence", 0.0),
            "reasoning": evaluation_data.get("reasoning", "")
        }
        
    except Exception as e:
        console.print(f"[red]Evaluation error: {e}[/red]")
        return {
            "is_correct": False,
            "confidence": 0.0,
            "reasoning": f"Evaluation error: {str(e)}"
        }

def run_websight_task(question: str, web_url: str, max_iters: int = 10) -> dict:
    """Run a single websight task"""
    
    # Construct the full task
    full_task = f"{question} Please visit {web_url}"
    
    console.print(f"[blue]Running task:[/blue] {question[:80]}...")
    console.print(f"[dim]URL: {web_url}[/dim]")
    
    start_time = time.time()
    
    try:
        # Execute websight.py
        cmd = [
            sys.executable, "websight.py", 
            "--task", full_task,
            "--max-iters", str(max_iters)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            # timeout=120  # 2 minute timeout for demo
        )
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            return {
                "success": True,
                "output": result.stdout.strip(),
                "error": "",
                "execution_time": execution_time
            }
        else:
            return {
                "success": False,
                "output": "",
                "error": result.stderr.strip(),
                "execution_time": execution_time
            }
            
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        return {
            "success": False,
            "output": "",
            "error": "Task execution timed out",
            "execution_time": execution_time
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        return {
            "success": False,
            "output": "",
            "error": f"Execution error: {str(e)}",
            "execution_time": execution_time
        }

def demo_evaluation():
    """Run a demo evaluation on a few tasks"""
    
    console.print("[bold green]Websight Demo Evaluation[/bold green]")
    console.print("=" * 50)
    
    # Check prerequisites
    if not os.getenv("OPENROUTER_API_KEY"):
        console.print("[red]Error: OPENROUTER_API_KEY environment variable not set[/red]")
        return
        
    if not Path("websight.py").exists():
        console.print("[red]Error: websight.py not found[/red]")
        return
        
    if not Path("webvoyavger.json").exists():
        console.print("[red]Error: webvoyavger.json not found[/red]")
        return
    
    # Load a few test cases
    with open("webvoyavger.json", 'r') as f:
        all_tasks = json.load(f)
    
    # Select first 3 tasks for demo
    demo_tasks = all_tasks[:3]
    
    console.print(f"[blue]Running demo with {len(demo_tasks)} tasks[/blue]\n")
    
    results = []
    
    for i, task in enumerate(demo_tasks, 1):
        console.print(f"[bold]Task {i}/{len(demo_tasks)}: {task['id']}[/bold]")
        
        # Extract task data
        question = task.get("ques", "")
        web_url = task.get("web", "")
        expected_answer = task.get("answer", {}).get("ans", "")
        
        # Run websight task
        execution_result = run_websight_task(question, web_url)
        
        if execution_result["success"]:
            console.print("[green]✅ Execution successful[/green]")
            console.print(f"[dim]Execution time: {execution_result['execution_time']:.2f}s[/dim]")
            
            # Evaluate response
            evaluation = evaluate_single_response(
                question, 
                expected_answer, 
                execution_result["output"]
            )
            
            status = "✅ CORRECT" if evaluation["is_correct"] else "❌ INCORRECT"
            console.print(f"{status} (Confidence: {evaluation['confidence']:.2f})")
            console.print(f"[dim]Reasoning: {evaluation['reasoning']}[/dim]")
            
            results.append({
                "task_id": task['id'],
                "success": True,
                "is_correct": evaluation["is_correct"],
                "confidence": evaluation["confidence"],
                "reasoning": evaluation["reasoning"],
                "execution_time": execution_result["execution_time"],
                "actual_output": execution_result["output"][:200] + "..." if len(execution_result["output"]) > 200 else execution_result["output"]
            })
            
        else:
            console.print("[red]❌ Execution failed[/red]")
            console.print(f"[red]Error: {execution_result['error']}[/red]")
            
            results.append({
                "task_id": task['id'],
                "success": False,
                "is_correct": False,
                "confidence": 0.0,
                "reasoning": "Execution failed",
                "execution_time": execution_result["execution_time"],
                "error": execution_result["error"]
            })
        
        console.print("")  # Empty line for readability
    
    # Summary
    console.print("[bold green]Demo Results Summary[/bold green]")
    console.print("=" * 30)
    
    successful_executions = sum(1 for r in results if r["success"])
    correct_answers = sum(1 for r in results if r["is_correct"])
    total_tasks = len(results)
    
    console.print(f"Total tasks: {total_tasks}")
    console.print(f"Successful executions: {successful_executions}/{total_tasks}")
    console.print(f"Correct answers: {correct_answers}/{total_tasks}")
    console.print(f"Success rate: {successful_executions/total_tasks:.2%}")
    console.print(f"Accuracy: {correct_answers/total_tasks:.2%}")
    
    if successful_executions > 0:
        avg_time = sum(r["execution_time"] for r in results if r["success"]) / successful_executions
        console.print(f"Average execution time: {avg_time:.2f}s")
    
    # Save demo results
    demo_results_file = f"demo_results_{int(time.time())}.json"
    with open(demo_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    console.print(f"[green]Demo results saved to {demo_results_file}[/green]")

if __name__ == "__main__":
    demo_evaluation() 
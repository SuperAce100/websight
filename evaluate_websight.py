#!/usr/bin/env python3
"""
Websight Evaluation Script

This script executes websight.py with tasks from webvoyavger.json and evaluates 
the responses using a lightweight model to determine correctness.
"""

import json
import subprocess
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from openai import OpenAI
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, TaskID

# Load environment variables
load_dotenv()

# Initialize clients
console = Console()
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

EVALUATION_MODEL = "openai/gpt-4.1-nano"

@dataclass
class TaskResult:
    """Container for task execution results"""
    task_id: str
    web_name: str
    question: str
    web_url: str
    expected_answer: str
    actual_output: str
    execution_time: float
    success: bool
    error_message: str = ""
    evaluation_score: float = 0.0
    evaluation_reasoning: str = ""
    is_correct: bool = False

class WebsightEvaluator:
    """Main evaluator class for websight performance"""
    
    def __init__(self, dataset_path: str = "webvoyavger.json", max_iterations: int = 25):
        self.dataset_path = Path(dataset_path)
        self.max_iterations = max_iterations
        self.results: List[TaskResult] = []
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('evaluation.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load the webvoyager dataset"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"Loaded {len(data)} tasks from {self.dataset_path}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            return []

    def execute_websight_task(self, task_data: Dict[str, Any]) -> TaskResult:
        """Execute a single websight task"""
        task_id = task_data.get("id", "unknown")
        web_name = task_data.get("web_name", "unknown")
        question = task_data.get("ques", "")
        web_url = task_data.get("web", "")
        expected_answer = task_data.get("answer", {}).get("ans", "")
        
        # Construct the task string: "question plus the web"
        full_task = f"{question} Please visit {web_url}"
        
        console.print(f"[blue]Executing task {task_id}:[/blue] {question[:80]}...")
        
        start_time = time.time()
        
        try:
            # Execute websight.py with the task
            cmd = [
                sys.executable, "websight.py", 
                "--task", full_task,
                "--max-iters", str(self.max_iterations)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per task
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                actual_output = result.stdout.strip()
                success = True
                error_message = ""
            else:
                actual_output = result.stderr.strip()
                success = False
                error_message = f"Return code: {result.returncode}, Error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            actual_output = ""
            success = False
            error_message = "Task execution timed out"
            
        except Exception as e:
            execution_time = time.time() - start_time
            actual_output = ""
            success = False
            error_message = f"Execution error: {str(e)}"
            
        return TaskResult(
            task_id=task_id,
            web_name=web_name,
            question=question,
            web_url=web_url,
            expected_answer=expected_answer,
            actual_output=actual_output,
            execution_time=execution_time,
            success=success,
            error_message=error_message
        )

    def evaluate_response(self, result: TaskResult) -> TaskResult:
        """Evaluate if the actual output aligns with the expected answer"""
        
        if not result.success:
            result.evaluation_score = 0.0
            result.evaluation_reasoning = "Task execution failed"
            result.is_correct = False
            return result
            
        evaluation_prompt = f"""
        You are evaluating whether a web automation agent's response correctly answers a given task.

        TASK: {result.question}
        EXPECTED ANSWER: {result.expected_answer}
        ACTUAL OUTPUT: {result.actual_output}

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
            
            result.is_correct = evaluation_data.get("is_correct", False)
            result.evaluation_score = evaluation_data.get("confidence", 0.0)
            result.evaluation_reasoning = evaluation_data.get("reasoning", "")
            
        except Exception as e:
            self.logger.error(f"Evaluation failed for task {result.task_id}: {e}")
            result.evaluation_score = 0.0
            result.evaluation_reasoning = f"Evaluation error: {str(e)}"
            result.is_correct = False
            
        return result

    def run_evaluation(self, limit: int = None) -> Dict[str, Any]:
        """Run the complete evaluation pipeline"""
        
        console.print("[bold green]Starting Websight Evaluation[/bold green]")
        
        # Load dataset
        dataset = self.load_dataset()
        if not dataset:
            console.print("[red]Failed to load dataset[/red]")
            return {}
            
        # Limit number of tasks if specified
        if limit:
            dataset = dataset[:limit]
            console.print(f"[yellow]Limited evaluation to {limit} tasks[/yellow]")
            
        total_tasks = len(dataset)
        console.print(f"[blue]Evaluating {total_tasks} tasks[/blue]")
        
        # Execute tasks with progress bar
        with Progress() as progress:
            task_progress = progress.add_task("Executing tasks...", total=total_tasks)
            
            for i, task_data in enumerate(dataset):
                console.print(f"\n[bold]Task {i+1}/{total_tasks}[/bold]")
                
                # Execute websight task
                result = self.execute_websight_task(task_data)
                
                # Evaluate response
                result = self.evaluate_response(result)
                
                # Store result
                self.results.append(result)
                
                # Log result
                status = "✅ CORRECT" if result.is_correct else "❌ INCORRECT"
                console.print(f"{status} - Score: {result.evaluation_score:.2f}")
                console.print(f"[dim]Reasoning: {result.evaluation_reasoning}[/dim]")
                
                progress.update(task_progress, advance=1)
                
        # Calculate statistics
        stats = self.calculate_statistics()
        
        # Save results
        self.save_results()
        
        return stats

    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate evaluation statistics"""
        
        if not self.results:
            return {}
            
        total_tasks = len(self.results)
        successful_executions = sum(1 for r in self.results if r.success)
        correct_answers = sum(1 for r in self.results if r.is_correct)
        
        avg_execution_time = sum(r.execution_time for r in self.results) / total_tasks
        avg_evaluation_score = sum(r.evaluation_score for r in self.results) / total_tasks
        
        accuracy = correct_answers / total_tasks if total_tasks > 0 else 0
        success_rate = successful_executions / total_tasks if total_tasks > 0 else 0
        
        # Group by web_name for detailed analysis
        by_website = {}
        for result in self.results:
            website = result.web_name
            if website not in by_website:
                by_website[website] = {"total": 0, "correct": 0, "success": 0}
            by_website[website]["total"] += 1
            if result.is_correct:
                by_website[website]["correct"] += 1
            if result.success:
                by_website[website]["success"] += 1
                
        # Calculate per-website accuracy
        for website, stats in by_website.items():
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            stats["success_rate"] = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        
        stats = {
            "total_tasks": total_tasks,
            "successful_executions": successful_executions,
            "correct_answers": correct_answers,
            "accuracy": accuracy,
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "average_evaluation_score": avg_evaluation_score,
            "by_website": by_website
        }
        
        return stats

    def save_results(self):
        """Save detailed results to JSON files"""
        
        timestamp = int(time.time())
        
        # Save detailed results
        detailed_results = []
        for result in self.results:
            detailed_results.append({
                "task_id": result.task_id,
                "web_name": result.web_name,
                "question": result.question,
                "web_url": result.web_url,
                "expected_answer": result.expected_answer,
                "actual_output": result.actual_output,
                "execution_time": result.execution_time,
                "success": result.success,
                "error_message": result.error_message,
                "evaluation_score": result.evaluation_score,
                "evaluation_reasoning": result.evaluation_reasoning,
                "is_correct": result.is_correct
            })
            
        detailed_file = f"evaluation_results_detailed_{timestamp}.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
            
        console.print(f"[green]Detailed results saved to {detailed_file}[/green]")
        
        # Save summary statistics
        stats = self.calculate_statistics()
        summary_file = f"evaluation_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
            
        console.print(f"[green]Summary saved to {summary_file}[/green]")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Websight performance")
    parser.add_argument("--dataset", default="webvoyavger.json", help="Path to dataset file")
    parser.add_argument("--limit", type=int, help="Limit number of tasks to evaluate")
    parser.add_argument("--max-iters", type=int, default=25, help="Max iterations per task")
    
    args = parser.parse_args()
    
    # Check for required environment variables
    if not os.getenv("OPENROUTER_API_KEY"):
        console.print("[red]Error: OPENROUTER_API_KEY environment variable not set[/red]")
        sys.exit(1)
        
    # Check if websight.py exists
    if not Path("websight.py").exists():
        console.print("[red]Error: websight.py not found in current directory[/red]")
        sys.exit(1)
        
    # Check if dataset exists
    if not Path(args.dataset).exists():
        console.print(f"[red]Error: Dataset file {args.dataset} not found[/red]")
        sys.exit(1)
        
    # Run evaluation
    evaluator = WebsightEvaluator(args.dataset, args.max_iters)
    stats = evaluator.run_evaluation(args.limit)
    
    # Print final statistics
    if stats:
        console.print("\n[bold green]Evaluation Complete![/bold green]")
        console.print(f"[blue]Total Tasks:[/blue] {stats['total_tasks']}")
        console.print(f"[blue]Successful Executions:[/blue] {stats['successful_executions']}")
        console.print(f"[blue]Correct Answers:[/blue] {stats['correct_answers']}")
        console.print(f"[blue]Overall Accuracy:[/blue] {stats['accuracy']:.2%}")
        console.print(f"[blue]Success Rate:[/blue] {stats['success_rate']:.2%}")
        console.print(f"[blue]Average Execution Time:[/blue] {stats['average_execution_time']:.2f}s")
        console.print(f"[blue]Average Evaluation Score:[/blue] {stats['average_evaluation_score']:.2f}")
        
        # Show top performing websites
        console.print("\n[bold]Performance by Website:[/bold]")
        sorted_websites = sorted(
            stats['by_website'].items(), 
            key=lambda x: x[1]['accuracy'], 
            reverse=True
        )
        for website, website_stats in sorted_websites[:10]:
            console.print(f"  {website}: {website_stats['accuracy']:.2%} ({website_stats['correct']}/{website_stats['total']})")

if __name__ == "__main__":
    main() 
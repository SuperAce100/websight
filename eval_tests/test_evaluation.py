#!/usr/bin/env python3
"""
Quick test of the evaluation system without running websight.py
"""

import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console

load_dotenv()
console = Console()

def test_evaluation_api():
    """Test that the evaluation API works"""
    
    console.print("[bold blue]Testing Evaluation System[/bold blue]")
    
    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        console.print("[red]❌ OPENROUTER_API_KEY not set[/red]")
        return False
    
    # Initialize client
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    
    # Test evaluation
    test_question = "Find the price of a hotel room in Paris"
    test_expected = "Hotel Room: $150/night"
    test_actual = "Found hotel room for 150 USD per night"
    
    evaluation_prompt = f"""
    You are evaluating whether a web automation agent's response correctly answers a given task.

    TASK: {test_question}
    EXPECTED ANSWER: {test_expected}
    ACTUAL OUTPUT: {test_actual}

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
        console.print("[yellow]Making API call...[/yellow]")
        
        response = client.chat.completions.create(
            model="openai/gpt-4.1-nano",
            messages=[
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        evaluation_text = response.choices[0].message.content.strip()
        console.print(f"[green]Raw response:[/green] {evaluation_text}")
        
        evaluation_data = json.loads(evaluation_text)
        
        console.print("[green]✅ API test successful![/green]")
        console.print(f"Is Correct: {evaluation_data['is_correct']}")
        console.print(f"Confidence: {evaluation_data['confidence']}")
        console.print(f"Reasoning: {evaluation_data['reasoning']}")
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ API test failed: {e}[/red]")
        return False

def test_dataset_loading():
    """Test that the dataset can be loaded"""
    
    console.print("\n[bold blue]Testing Dataset Loading[/bold blue]")
    
    try:
        with open("webvoyavger.json", 'r') as f:
            data = json.load(f)
        
        console.print(f"[green]✅ Dataset loaded successfully![/green]")
        console.print(f"Total tasks: {len(data)}")
        console.print(f"First task ID: {data[0]['id']}")
        console.print(f"First task question: {data[0]['ques'][:80]}...")
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Dataset loading failed: {e}[/red]")
        return False

def main():
    """Run all tests"""
    
    console.print("[bold green]Websight Evaluation System Test[/bold green]")
    console.print("=" * 50)
    
    api_ok = test_evaluation_api()
    dataset_ok = test_dataset_loading()
    
    console.print(f"\n[bold]Test Results:[/bold]")
    console.print(f"API Test: {'✅ PASS' if api_ok else '❌ FAIL'}")
    console.print(f"Dataset Test: {'✅ PASS' if dataset_ok else '❌ FAIL'}")
    
    if api_ok and dataset_ok:
        console.print("\n[bold green]🎉 All tests passed! System is ready.[/bold green]")
        console.print("\nNext steps:")
        console.print("1. Run: python demo_evaluation.py")
        console.print("2. Or run: python evaluate_websight.py --limit 5")
    else:
        console.print("\n[bold red]⚠️ Some tests failed. Check the issues above.[/bold red]")

if __name__ == "__main__":
    main() 
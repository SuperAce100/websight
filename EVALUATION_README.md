# Websight Evaluation System

This directory contains a comprehensive evaluation system for the Websight web automation agent. The system executes tasks from the WebVoyager dataset and evaluates responses using a lightweight model.

## Overview

The evaluation system consists of two main components:

1. **Task Execution**: Runs `websight.py` with tasks combining questions and web URLs
2. **Response Evaluation**: Uses `openai/gpt-4.1-nano` to evaluate if outputs align with expected answers

## Files

- `evaluate_websight.py` - Full evaluation script for all tasks
- `demo_evaluation.py` - Demo script for testing with a few tasks
- `webvoyavger.json` - Dataset containing web automation tasks
- `EVALUATION_README.md` - This documentation

## Prerequisites

### Environment Setup

1. **API Key**: Set your OpenRouter API key in environment variables:
   ```bash
   export OPENROUTER_API_KEY="your_key_here"
   ```
   Or create a `.env` file:
   ```
   OPENROUTER_API_KEY=your_key_here
   ```

2. **Dependencies**: Ensure you have the required packages:
   ```bash
   pip install openai python-dotenv rich
   ```

3. **Websight**: Make sure `websight.py` is in the current directory and functional

### Dataset Structure

The `webvoyavger.json` dataset contains tasks with this structure:
```json
{
  "web_name": "Booking",
  "id": "Booking--43", 
  "ques": "Search for properties in Los Angeles...",
  "web": "https://www.booking.com/",
  "answer": {
    "notice": "real-time, check task requirements...",
    "ans": "Breakfast Included, Wonderful: 9+, Fitness center ..."
  }
}
```

## Usage

### Demo Evaluation (Recommended First)

Run a quick test with 3 tasks:

```bash
python demo_evaluation.py
```

This will:
- Execute the first 3 tasks from the dataset
- Show real-time progress and results
- Save results to `demo_results_<timestamp>.json`

### Full Evaluation

Run evaluation on all tasks:

```bash
python evaluate_websight.py
```

**Options:**
- `--limit N` - Evaluate only first N tasks
- `--max-iters N` - Set max iterations per task (default: 25)
- `--dataset PATH` - Use different dataset file

**Examples:**
```bash
# Evaluate first 10 tasks only
python evaluate_websight.py --limit 10

# Use shorter iterations
python evaluate_websight.py --max-iters 15

# Custom dataset
python evaluate_websight.py --dataset my_tasks.json
```

## Evaluation Methodology

### Task Construction

Tasks are constructed by combining the question with the web URL:
```
"{question} Please visit {web_url}"
```

### Evaluation Criteria

The lightweight model (`openai/gpt-4.1-nano`) evaluates responses based on:

1. **Content Accuracy**: Does the output contain requested information?
2. **Relevance**: Is the information relevant to the task?
3. **Core Question**: Does it address the main question?
4. **Format Correctness**: For real-time data, is the format appropriate?

### Scoring

- **Binary Correctness**: `true`/`false` for each task
- **Confidence Score**: 0.0-1.0 indicating evaluation confidence
- **Reasoning**: Text explanation of the decision

### Flexibility

The evaluation is designed to be flexible:
- Responses don't need to be identical to expected answers
- Real-time data differences are acceptable
- Different phrasing is allowed if core information matches

## Output Files

### Demo Results
- `demo_results_<timestamp>.json` - Results from demo evaluation

### Full Evaluation Results
- `evaluation_results_detailed_<timestamp>.json` - Detailed results for each task
- `evaluation_summary_<timestamp>.json` - Aggregated statistics
- `evaluation.log` - Execution logs

### Result Structure

**Detailed Results:**
```json
{
  "task_id": "Booking--43",
  "web_name": "Booking", 
  "question": "Search for properties...",
  "web_url": "https://www.booking.com/",
  "expected_answer": "Breakfast Included...",
  "actual_output": "Found the following filters...",
  "execution_time": 45.2,
  "success": true,
  "evaluation_score": 0.85,
  "evaluation_reasoning": "Output contains relevant filters...",
  "is_correct": true
}
```

**Summary Statistics:**
```json
{
  "total_tasks": 100,
  "successful_executions": 95,
  "correct_answers": 78,
  "accuracy": 0.78,
  "success_rate": 0.95,
  "average_execution_time": 42.3,
  "average_evaluation_score": 0.72,
  "by_website": {
    "Booking": {"total": 10, "correct": 8, "accuracy": 0.8},
    "Google": {"total": 15, "correct": 12, "accuracy": 0.8}
  }
}
```

## Performance Metrics

The system tracks multiple metrics:

- **Accuracy**: Percentage of correct answers
- **Success Rate**: Percentage of successful task executions
- **Execution Time**: Average time per task
- **Website Performance**: Accuracy breakdown by website
- **Confidence Scores**: Average evaluation confidence

## Troubleshooting

### Common Issues

1. **API Key Error**: 
   ```
   Error: OPENROUTER_API_KEY environment variable not set
   ```
   **Solution**: Set your OpenRouter API key in environment variables

2. **Websight Not Found**:
   ```
   Error: websight.py not found in current directory
   ```
   **Solution**: Run from directory containing `websight.py`

3. **Dataset Missing**:
   ```
   Error: Dataset file webvoyavger.json not found
   ```
   **Solution**: Ensure dataset file is in current directory

4. **Task Timeout**:
   ```
   Task execution timed out
   ```
   **Solution**: Some tasks may take longer; this is normal

5. **Browser Issues**:
   - Ensure browser dependencies are installed
   - Check if display is available (for GUI browsers)
   - Consider running in headless mode

### Debugging

- Check `evaluation.log` for detailed execution logs
- Review failed tasks in detailed results files
- Use demo evaluation first to test setup

## Customization

### Custom Evaluation Prompts

Modify the evaluation prompt in the scripts to change evaluation criteria:

```python
evaluation_prompt = f"""
Your custom evaluation instructions here...
"""
```

### Different Models

Change the evaluation model:

```python
EVALUATION_MODEL = "openai/gpt-4.1-mini"  # More capable but slower
EVALUATION_MODEL = "openai/gpt-3.5-turbo"  # Alternative option
```

### Timeout Settings

Adjust timeouts for different environments:

```python
timeout=300  # 5 minutes for full evaluation
timeout=120  # 2 minutes for demo
```

## Analysis Tips

1. **Website-Specific Performance**: Check `by_website` statistics to identify which sites work best
2. **Execution vs Accuracy**: Compare success rate and accuracy to identify evaluation vs execution issues
3. **Time Analysis**: Monitor execution times to optimize performance
4. **Error Patterns**: Review error messages to identify common failure modes

## Future Enhancements

Potential improvements:
- Parallel task execution
- More sophisticated evaluation metrics
- Integration with different evaluation models
- Real-time monitoring dashboard
- Automated retry mechanisms
- Performance regression testing

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review log files for detailed error information
3. Test with demo evaluation first
4. Verify all prerequisites are met 
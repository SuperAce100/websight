# WebSight

A vision-based web automation agent powered by Llama 4 Maverick and Playwright.

## Features
- Automate web tasks using natural language instructions
- Vision-based decision making (analyzes screenshots)
- Uses Playwright for browser automation
- Integrates with OpenRouter Llama 4 Maverick for LLM and vision

## Requirements
- Python 3.12+
- [OpenRouter API key](https://openrouter.ai/keys)

## Installation
1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd websight
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables:
   - Copy `env_template.txt` to `.env` and add your OpenRouter API key:
     ```bash
     cp env_template.txt .env
     # Edit .env and set OPENROUTER_API_KEY
     ```

## Usage
Run the main agent with a sample task:
```bash
python main.py
```

Or try the interactive demo:
```bash
python tests/demo.py
```

## Example Task
- "Go to google.com and search for 'Llama 4 Maverick'"

## Project Structure
- `main.py` - Quick start example
- `agent.py` - Main WebAgent logic
- `browser.py` - Playwright browser wrapper
- `llms.py` - LLM and vision API integration
- `task.py` - Task management
- `tests/` - Demos and test scripts

## License
MIT

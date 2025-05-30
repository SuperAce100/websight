- Take a task from the user of what to do for a browser task
- Query openrouter's gemini 2.5 flash using llms.py to generate a plan, set the current state to nothing done, and sets to actions done to nothing done
- Have a next task function which takes the plan, state, and actions completed, and then queries openrouter's gpt 4.1 nano of what specific SINGULAR task (like click, scroll, navigate to page, etc) take these tasks from what can be done in @browser.py and use the structured output calls from communicators/llms.py
- Have a process screenshot function that screenshots the page and determines what action to take based on what can be done in @browser.py and how to interact with the current page using llama 4 maverick openrouter use communicators/api_client.py
- Attempt to make this update, if it works, call openrouter gpt 4.1 nano to update current state and actions done
- Keep going until task is completed, keep track of this somehow whether it is a recurring call to see what it is and call a final call when completed to summarize task results

DO NOT MAKE NEW FILES FOR THE SUPPORTER FUNCTIONS, JUST READ AND USE BROWSER, LLMS, and API_CLIENT, MAKE A NEW FILE CALLED websight.py that does everything, TRIPLE CHECK ALL LOGIC WORKS AND CHECK THE INTERNET TO ENSURE THE DOCS FOR API CALLS ARE CURRENT AND FUNCTIONAL
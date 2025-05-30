import browser
from communicators import llms, api_client
from pydantic import BaseModel
import os # Added for screenshot path

# --- Pydantic Models for Structured LLM Responses ---
class NextTaskResponse(BaseModel):
    action: str
    # Parameters for the action, e.g., url for navigate, coordinates for click
    parameters: dict | None = None

class StateUpdateResponse(BaseModel):
    new_state: str

class ImageAnalysisResponse(BaseModel):
    action_to_take: str # e.g., "click", "type", "scroll"
    element_details: dict | None = None # e.g., {"text": "Login button", "coordinates": {"x": 100, "y": 200}} or {"form_field_label": "Username", "value_to_type": "testuser"}


# Initialize browser
browser_instance = browser.Browser()

# Initialize state
current_state = "nothing done"
actions_done = []
plan = None
MAX_ACTIONS = 10 # Safety break for the main loop
SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def get_plan_from_user_task(user_task: str):
    """
    Generates a plan based on the user task using Gemini 2.5 Flash.
    Updates current_state and resets actions_done.
    """
    global plan, current_state, actions_done
    system_prompt = "You are a planning assistant. Create a step-by-step plan to accomplish the given browser task. The plan should be a numbered list of high-level actions."
    prompt = f"Create a plan to accomplish the following browser task: {user_task}"
    try:
        generated_plan = llms.llm_call(prompt=prompt, system_prompt=system_prompt, model="google/gemini-2.5-flash-preview-05-20") # Corrected model name
        plan = generated_plan
        current_state = "Plan generated."
        actions_done = []
        print(f"Plan: {plan}")
    except Exception as e:
        print(f"Error getting plan: {e}")
        plan = "Error in planning." # Fallback plan
        current_state = "Error in planning."
    return plan

def get_next_task(current_plan: str, state: str, completed_actions: list) -> NextTaskResponse | None:
    """
    Determines the next specific singular task using GPT 4.1 Nano (or a similar small model).
    Uses structured output.
    """
    system_prompt = (
        "You are a task decomposer. Based on the overall plan, current state, and actions already completed, "
        "determine the very next single, specific browser action to perform. "
        "Choose an action from the following: "
        "navigate (parameters: {'url': 'string'}), "
        "click (parameters: {'x': int, 'y': int}), " # Assuming click by coordinates for now
        "type (parameters: {'content': 'string'}), " # Assuming type at current focus
        "scroll (parameters: {'direction': 'up'|'down'|'left'|'right', 'x': int, 'y': int}), "
        "wait (no parameters, waits for a few seconds)."
        "If no further actions are clear or the plan seems complete, respond with action 'finish'."
    )
    prompt = (
        f"Current Plan: '{current_plan}'\n"
        f"Current State: '{state}'\n"
        f"Actions Completed: {completed_actions}\n"
        f"What is the next single browser task to perform based on the available actions?"
    )
    try:
        # Using gpt-4-turbo as a stand-in for gpt-4.1-nano as it's available on OpenRouter
        # And ensuring we use a model capable of structured output well.
        # "openai/gpt-4-turbo" is a valid model that should support JSON schema.
        next_task_response = llms.llm_call(
            prompt=prompt,
            system_prompt=system_prompt,
            response_format=NextTaskResponse,
            model="openai/gpt-4.1-nano" # Updated model
        )
        if isinstance(next_task_response, NextTaskResponse):
            print(f"Next task: {next_task_response.action} with params {next_task_response.parameters}")
            if next_task_response.action == "finish":
                return None # Indicates completion
            return next_task_response
        else:
            print(f"Received unexpected response type from LLM for next task: {type(next_task_response)}")
            return None
    except Exception as e:
        print(f"Error getting next task: {e}")
        return None


def process_screenshot_and_decide_action(current_plan: str, current_task_description: str) -> ImageAnalysisResponse | None:
    """
    Takes a screenshot, analyzes it using Llama 4 Maverick (or Molmo), and decides the browser interaction.
    Uses structured output.
    """
    screenshot_filename = os.path.join(SCREENSHOT_DIR, f"current_page_{len(actions_done)}.png")
    try:
        # browser_instance.take_screenshot returns base64, no need to open file
        screenshot_base64 = browser_instance.take_screenshot(path=screenshot_filename)
        print(f"Screenshot saved to {screenshot_filename}")

        query = (
            f"The overall plan is: '{current_plan}'. The current specific task is: '{current_task_description}'. "
            "Based on this screenshot, what precise browser action should be taken? "
            "For clicks, provide coordinates or identify elements by text/role. "
            "For typing, identify the form field and specify the text. "
            "For scrolling, specify direction and a general area if relevant."
            "Available actions are: click, type, scroll."
        )
        # Using "allenai/molmo-7b-d:free" as specified in api_client.py, which is good for image analysis
        action_decision = api_client.analyze_image(
            image_base64=screenshot_base64,
            query=query,
            response_format=ImageAnalysisResponse,
            model="allenai/molmo-7b-d:free"
        )
        if isinstance(action_decision, ImageAnalysisResponse):
            print(f"Action from screenshot analysis: {action_decision.action_to_take} with details {action_decision.element_details}")
            return action_decision
        else:
            print(f"Received unexpected response type from image analysis: {type(action_decision)}")
            return None
    except Exception as e:
        print(f"Error processing screenshot: {e}")
        return None


def execute_browser_action(action_details: NextTaskResponse | ImageAnalysisResponse) -> bool:
    """
    Executes the determined browser action using the browser_instance.
    Handles both NextTaskResponse and ImageAnalysisResponse.
    """
    success = False
    action_type = None
    params = None

    if isinstance(action_details, NextTaskResponse):
        action_type = action_details.action
        params = action_details.parameters if action_details.parameters else {}
    elif isinstance(action_details, ImageAnalysisResponse):
        action_type = action_details.action_to_take
        params = action_details.element_details if action_details.element_details else {}
    else:
        print(f"Unknown action_details type: {type(action_details)}")
        return False

    print(f"Executing browser action: {action_type} with params: {params}")

    try:
        if action_type == "navigate":
            url = params.get("url")
            if url:
                browser_instance.goto_url(url)
                success = True
            else:
                print("Navigate action missing URL.")
        elif action_type == "click":
            # If process_screenshot provides coordinates, use them.
            # Otherwise, this part might need more sophisticated element finding based on text/role
            # For now, using coordinates if present, else logging.
            x = params.get("x")
            y = params.get("y")
            element_text = params.get("text") # From ImageAnalysisResponse potentially

            if x is not None and y is not None:
                browser_instance.click(x=int(x), y=int(y))
                success = True
            elif element_text:
                # This is a placeholder. Real implementation would need to find element by text.
                print(f"Placeholder: Would need to implement finding and clicking element with text '{element_text}'")
                # For demo, let's assume a click at a default coordinate if text is given but no x,y
                # browser_instance.click(x=100, y=100) # Example
                # success = True
                print("Click by text not fully implemented yet, skipping.")
            else:
                print("Click action missing coordinates or identifiable element text.")

        elif action_type == "type":
            content = params.get("content") # From NextTaskResponse
            value_to_type = params.get("value_to_type") # From ImageAnalysisResponse

            text_to_type = content if content is not None else value_to_type
            
            if text_to_type is not None:
                # Assuming typing into the currently focused element or a specified one
                # Form field identification from ImageAnalysisResponse needs to be handled.
                # For now, just types.
                browser_instance.type(text_to_type)
                success = True
            else:
                print("Type action missing content.")
        elif action_type == "scroll":
            direction = params.get("direction")
            # browser.py scroll takes x, y, direction. We might need to adapt.
            # For simplicity, let's assume a generic scroll if no coordinates given, or adapt based on what LLM provides
            scroll_x = params.get("x", 500) # Default scroll position
            scroll_y = params.get("y", 500) # Default scroll position
            if direction:
                browser_instance.scroll(x=int(scroll_x), y=int(scroll_y), direction=direction)
                success = True
            else:
                print("Scroll action missing direction.")
        elif action_type == "wait":
            browser_instance.wait() # Default 5 seconds
            success = True
        elif action_type == "finish":
            print("Received 'finish' action. No browser action to execute.")
            success = True # This is a successful "no-op"
        else:
            print(f"Unknown or unsupported action type: {action_type}")

        if success:
            print(f"Action '{action_type}' executed successfully.")
        else:
            print(f"Action '{action_type}' failed or was not fully specified.")

    except Exception as e:
        print(f"Error executing browser action {action_type}: {e}")
        success = False
    return success


def update_state_and_actions(executed_action_type: str, success: bool):
    """
    Updates the current state and actions done using GPT 4.1 Nano (or similar).
    Uses structured output for the new state.
    """
    global current_state, actions_done
    if success:
        actions_done.append(executed_action_type)
        system_prompt = "You are a state tracking assistant. Based on the last action's success and the history, provide a concise update to the current state of the overall task."
        prompt = (
            f"The action '{executed_action_type}' was completed successfully. "
            f"Previous actions were: {actions_done[:-1]}. "
            f"Previous state was: '{current_state}'. "
            f"What is the new, concise current state of the overall task?"
        )
        try:
            # Using gpt-4-turbo for reliable structured output.
            response = llms.llm_call(
                prompt=prompt,
                system_prompt=system_prompt,
                response_format=StateUpdateResponse,
                model="openai/gpt-4.1-nano" # Updated model
            )
            if isinstance(response, StateUpdateResponse):
                current_state = response.new_state
            else:
                current_state = f"State update failed after successful {executed_action_type}. Fallback state."
        except Exception as e:
            print(f"Error updating state with LLM: {e}")
            current_state = f"State after {executed_action_type} (LLM update failed)"
    else:
        # Potentially a more sophisticated error handling or state update for failures.
        current_state = f"State after failed attempt of action: {executed_action_type}. Previous state was: {current_state}"

    print(f"Updated State: {current_state}")
    print(f"Actions Done: {actions_done}")


def is_task_completed(current_plan: str, state: str, completed_actions: list) -> bool:
    """
    Checks if the overall task is completed.
    This could be based on the plan, number of steps, or an LLM call.
    """
    if not plan or plan == "Error in planning.": # If planning failed, can't complete.
        return False
        
    if "finish" in completed_actions: # If 'finish' action was explicitly decided.
        print("Task marked as completed due to 'finish' action.")
        return True

    # Simple heuristic: if plan has N steps and we've done N or N+k actions.
    # This is a placeholder for more robust completion checking.
    # For now, let's use a simpler check based on actions_done length or next_task returning None.
    if len(actions_done) >= MAX_ACTIONS:
        print(f"Task considered completed due to reaching max actions ({MAX_ACTIONS}).")
        return True
        
    # A more sophisticated check could involve asking an LLM:
    # system_prompt = "Based on the plan, current state, and actions completed, is the overall task finished?"
    # prompt = f"Plan: {current_plan}\nState: {state}\nCompleted Actions: {completed_actions}\nIs the task complete? Respond with only 'yes' or 'no'."
    # response = llms.llm_call(prompt, system_prompt, model="openai/gpt-4-turbo") # Using a capable model
    # return response.lower().strip() == "yes"

    return False # Default to not completed


def summarize_task_results():
    """
    Summarizes the task results using a final LLM call.
    """
    global plan, current_state, actions_done
    if not plan or plan == "Error in planning.":
        summary = "Task could not be summarized because planning failed."
        print(f"Final Summary: {summary}")
        return summary

    system_prompt = "You are a summarization assistant. Provide a concise summary of the browser task execution."
    prompt = (
        f"Summarize the results of the browser task execution based on the following:\n"
        f"Initial Plan: '{plan}'\n"
        f"Final State: '{current_state}'\n"
        f"Actions Performed: {actions_done}"
    )
    try:
        summary = llms.llm_call(prompt=prompt, system_prompt=system_prompt, model="openai/gpt-4.1-nano") # Using a capable model for summarization
    except Exception as e:
        print(f"Error generating summary: {e}")
        summary = f"Summary generation failed. Plan: {plan}, Final State: {current_state}, Actions: {actions_done}"

    print(f"Final Summary: {summary}")
    return summary


def main_loop(user_task: str):
    """
    Main loop to execute the browser task.
    """
    global plan, current_state, actions_done

    # 1. Get Plan
    get_plan_from_user_task(user_task)
    if not plan or plan == "Error in planning.":
        print("Halting execution due to planning failure.")
        summarize_task_results() # Give a summary of the failure
        return

    actions_this_session = 0
    while not is_task_completed(plan, current_state, actions_done):
        if actions_this_session >= MAX_ACTIONS: # Safety break
             print(f"Exiting loop: Reached max actions ({MAX_ACTIONS}) for this session.")
             break
        
        # 2. Determine Next Task (Singular, from Plan)
        next_task_directive = get_next_task(plan, current_state, actions_done)

        if next_task_directive is None or next_task_directive.action == "finish":
            print("No more tasks to perform or 'finish' action received. Ending loop.")
            actions_done.append("finish") # Mark finish as an action
            break

        # 3. Process Screenshot for Current Task Context (Optional Refinement)
        # This step helps ground the 'next_task_directive' with visual context.
        # The LLM in process_screenshot_and_decide_action can use the general directive
        # (e.g., "click login button") and the screenshot to find *where* to click.
        
        # We pass the description of the task from `next_task_directive`
        current_task_description_for_vision = f"{next_task_directive.action}"
        if next_task_directive.parameters:
            current_task_description_for_vision += f" with parameters {next_task_directive.parameters}"

        action_to_execute = None
        # Decide whether to use screenshot analysis or direct execution
        # For now, let's try screenshot analysis if the action isn't simple like 'navigate' or 'wait'
        if next_task_directive.action not in ["navigate", "wait", "finish"]:
            print(f"Attempting screenshot analysis for task: {current_task_description_for_vision}")
            visual_action_details = process_screenshot_and_decide_action(
                current_plan=plan,
                current_task_description=current_task_description_for_vision
            )
            if visual_action_details:
                action_to_execute = visual_action_details # Use refined action from vision model
            else:
                print("Screenshot analysis did not yield a specific action, falling back to directive.")
                action_to_execute = next_task_directive # Fallback to the planned task if vision fails
        else:
            action_to_execute = next_task_directive # Directly use non-visual tasks

        if not action_to_execute:
            print("Could not determine an action to execute. Ending current iteration.")
            # This might indicate a need to re-plan or that the task is stuck.
            # For now, we'll just break this iteration, the main loop condition will be checked.
            break


        # 4. Execute Action
        action_type_for_state_update = "unknown_action"
        if isinstance(action_to_execute, NextTaskResponse):
            action_type_for_state_update = action_to_execute.action
        elif isinstance(action_to_execute, ImageAnalysisResponse):
            action_type_for_state_update = action_to_execute.action_to_take
            
        success = execute_browser_action(action_to_execute)

        # 5. Update State
        update_state_and_actions(action_type_for_state_update, success)

        if not success:
            print(f"Failed to execute: {action_type_for_state_update}. Consider re-planning or adapting.")
            # For now, we break on failure. More sophisticated logic could retry, or ask for help.
            break
        
        actions_this_session += 1
        # Small delay to make browser actions observable and avoid overwhelming services
        browser_instance.active_page.wait_for_timeout(1000)


    # 6. Summarize
    summarize_task_results()


if __name__ == "__main__":
    try:
        # browser_instance is already initialized globally
        print("Websight agent started.")
        # browser_instance.goto_url("about:blank") # Start with a blank page

        user_input = input("Please enter the browser task: ")
        if user_input:
            main_loop(user_input)
    except Exception as e:
        print(f"An error occurred in the main execution: {e}")
    finally:
        if browser_instance:
            browser_instance.close()
        print("Websight agent finished.") 

from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.firefox import GeckoDriverManager
from datetime import datetime
import os
import sys
import time
import pyautogui
import base64
from dotenv import load_dotenv
from api_client import analyze_image
from typing import Optional
from pydantic import BaseModel
from PIL import Image
import io

class ScreenElement(BaseModel):
    element_type: str
    text: str
    location: str
    is_clickable: bool

class ScreenAnalysis(BaseModel):
    elements: list[ScreenElement]
    suggested_actions: list[str]

class BrowserController:
    def __init__(self):
        """Initialize the Firefox browser controller."""
        try:
            # Set up Firefox options
            firefox_options = Options()
            firefox_options.set_preference("browser.download.folderList", 2)
            firefox_options.set_preference("browser.download.manager.showWhenStarting", False)
            firefox_options.add_argument("--start-maximized")
            
            # Set up Firefox driver
            service = Service(GeckoDriverManager().install())
            
            self.driver = webdriver.Firefox(
                service=service,
                options=firefox_options
            )
            
            # Maximize window while keeping browser UI
            self.driver.maximize_window()
            
            self.wait = WebDriverWait(self.driver, 10)  # 10 seconds timeout
            print("Firefox browser initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing Firefox driver: {str(e)}")
            print("Please make sure:")
            print("1. Mozilla Firefox is installed on your system")
            print("2. You're running the latest version of Firefox")
            sys.exit(1)

    def analyze_screenshot_with_molmo(self, image_path: str, user_query: str, response_format: Optional[BaseModel] = None) -> str | BaseModel:
        """
        Send screenshot to Molmo for analysis using OpenAI client.
        """
        try:
            # Read and encode the image
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Use the API client to analyze the image
            return analyze_image(
                image_base64=encoded_image,
                query=user_query,
                response_format=response_format
            )
            
        except Exception as e:
            print(f"Error analyzing screenshot: {str(e)}")
            return [] if response_format else ""

    def analyze_current_page(self, user_query: str) -> ScreenAnalysis:
        """
        Take a screenshot and analyze it with Molmo.
        """
        # Take screenshot
        screenshot_path = self.take_screenshot()
        
        # Analyze with Molmo using structured output
        analysis = self.analyze_screenshot_with_molmo(
            screenshot_path, 
            user_query,
            response_format=ScreenAnalysis
        )
        
        # Print results
        if isinstance(analysis, ScreenAnalysis):
            print("\nDetected Elements:")
            for idx, element in enumerate(analysis.elements, 1):
                print(f"{idx}. {element.element_type}: {element.text} ({element.location})")
            
            print("\nSuggested Actions:")
            for idx, action in enumerate(analysis.suggested_actions, 1):
                print(f"{idx}. {action}")
        
        return analysis

    def navigate_to(self, url):
        """Navigate to a specific URL."""
        self.driver.get(url)
        # Wait a bit for the page to load completely
        time.sleep(2)
        
    def take_screenshot(self, filename=None):
        """
        Take a screenshot of the browser window using Selenium and downscale to max 720p.
        If no filename is provided, it will use timestamp.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            
        # Create screenshots directory if it doesn't exist
        os.makedirs("screenshots", exist_ok=True)
        filepath = os.path.join("screenshots", filename)
        
        try:
            # Take screenshot using Selenium
            screenshot = self.driver.get_screenshot_as_png()
            
            # Convert to PIL Image for processing
            image = Image.open(io.BytesIO(screenshot))
            width, height = image.size
            
            # Downscale if larger than 720p (1280x720)
            if width > 1280 or height > 720:
                # Calculate new dimensions maintaining aspect ratio
                ratio = min(1280/width, 720/height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            image.save(filepath)
            print(f"Browser window screenshot saved as: {filepath}")
            
        except Exception as e:
            print(f"Error taking screenshot: {str(e)}")
        return filepath
        
    def click_element(self, selector, by=By.CSS_SELECTOR):
        """
        Click an element on the page using the provided selector.
        Default selector type is CSS_SELECTOR.
        """
        element = self.wait.until(
            EC.element_to_be_clickable((by, selector))
        )
        element.click()
        
    def click_at_coordinates(self, x, y):
        """Click at specific coordinates on the page."""
        actions = ActionChains(self.driver)
        actions.move_by_offset(x, y).click().perform()
        
    def close(self):
        """Close the browser."""
        self.driver.quit()

# Example usage
if __name__ == "__main__":
    try:
        # Create controller instance
        controller = BrowserController()
        
        # Example user query
        user_query = "identify all clickable elements and suggest possible search topics"
        
        # Navigate to a website
        controller.navigate_to("https://www.google.com")
        
        # Analyze the page with structured output
        analysis = controller.analyze_current_page(user_query)
        
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        # Always close the browser
        if 'controller' in locals():
            controller.close() 
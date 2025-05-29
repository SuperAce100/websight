from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime
import os
import sys

class ChromeController:
    def __init__(self):
        """Initialize the Chrome browser controller."""
        try:
            # Set up Chrome options
            chrome_options = Options()
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            # Set up Chrome driver with explicit version
            service = Service(ChromeDriverManager().install())
            
            self.driver = webdriver.Chrome(
                service=service,
                options=chrome_options
            )
            self.wait = WebDriverWait(self.driver, 10)  # 10 seconds timeout
        except Exception as e:
            print(f"Error initializing Chrome driver: {str(e)}")
            print("Please make sure:")
            print("1. Google Chrome is installed on your system")
            print("2. You're running the latest version of Chrome")
            print("3. Your system's antivirus isn't blocking the ChromeDriver")
            sys.exit(1)
        
    def navigate_to(self, url):
        """Navigate to a specific URL."""
        self.driver.get(url)
        
    def take_screenshot(self, filename=None):
        """
        Take a screenshot of the current page.
        If no filename is provided, it will use timestamp.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            
        # Create screenshots directory if it doesn't exist
        os.makedirs("screenshots", exist_ok=True)
        filepath = os.path.join("screenshots", filename)
        
        # Take screenshot
        self.driver.save_screenshot(filepath)
        print(f"Screenshot saved as: {filepath}")
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
    # Create controller instance
    controller = ChromeController()
    
    try:
        # Navigate to a website
        controller.navigate_to("https://www.google.com")
        
        # Take a screenshot
        controller.take_screenshot()
        
        # Click search box using CSS selector
        controller.click_element("input[name='q']")
        
        # Take another screenshot
        controller.take_screenshot("after_click.png")
        
    finally:
        # Always close the browser
        controller.close() 
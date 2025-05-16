# Browser Controller

This tool allows you to control Firefox browser programmatically, take screenshots, and simulate clicks using Selenium WebDriver.

## Prerequisites

- Python 3.7 or higher
- Mozilla Firefox browser installed
- pip (Python package installer)

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. The GeckoDriver (Firefox driver) will be automatically downloaded and managed by webdriver-manager.

## Usage

The `BrowserController` class provides several useful methods:

```python
from firefox_controller import BrowserController

# Create a controller instance
controller = BrowserController()

try:
    # Navigate to a website
    controller.navigate_to("https://www.example.com")
    
    # Take a screenshot (automatically saves with timestamp)
    controller.take_screenshot()
    
    # Take a screenshot with custom filename
    controller.take_screenshot("my_screenshot.png")
    
    # Click an element using CSS selector (default)
    controller.click_element("#submit-button")
    
    # Click an element using XPath
    controller.click_element("//button[@type='submit']", by=By.XPATH)
    
    # Click at specific coordinates
    controller.click_at_coordinates(100, 200)
    
finally:
    # Always close the browser when done
    controller.close()
```

Screenshots are saved in the `screenshots` directory.

## Features

- Automatic GeckoDriver (Firefox) management
- Screenshot capture with timestamp or custom filename
- Click elements using CSS selectors or XPath
- Click at specific coordinates
- Waits for elements to be clickable before interacting
- Automatic screenshots directory creation

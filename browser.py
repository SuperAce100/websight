import base64
import json
import os
import random
import time
import asyncio
from pydantic import BaseModel
from playwright.sync_api import sync_playwright
from PIL import Image


class BrowserState(BaseModel):
    page_url: str
    page_screenshot_base64: str


class Browser:
    def __init__(self):
        # Check if we're in an async context and handle it
        try:
            loop = asyncio.get_running_loop()
            print("⚠️ Warning: Detected running async loop, using sync API")
        except RuntimeError:
            # No running loop, we're good
            pass

        self.playwright = sync_playwright().start()
        self.driver = self.playwright.chromium.launch(headless=True, timeout=120000)
        self.context = self.driver.new_context()
        self.active_page = self.context.new_page()

    def _wait_for_load_state(self):
        self.active_page.wait_for_load_state("networkidle")
        self.active_page.wait_for_timeout(5000)

    def click(self, x: int, y: int):
        """Click at specific coordinates."""
        self.active_page.mouse.click(x, y)
        self._wait_for_load_state()

    def left_double(self, x: int, y: int):
        """Double click at specific coordinates."""
        self.active_page.mouse.dblclick(x, y)
        self._wait_for_load_state()

    def right_single(self, x: int, y: int):
        """Right click at specific coordinates."""
        self.active_page.mouse.click(x, y, button="right")
        self._wait_for_load_state()

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int):
        """Drag from start to end coordinates."""
        self.active_page.mouse.move(start_x, start_y)
        self.active_page.mouse.down()
        self.active_page.mouse.move(end_x, end_y)
        self.active_page.mouse.up()
        self._wait_for_load_state()

    def hotkey(self, key: str):
        """Press a hotkey combination."""

        keys = key.lower().split()
        # Press all modifier keys first
        for k in keys:
            if k == "ctrl":
                self.active_page.keyboard.down("Control")
            elif k == "shift":
                self.active_page.keyboard.down("Shift")
            elif k == "alt":
                self.active_page.keyboard.down("Alt")
            elif k == "cmd":
                self.active_page.keyboard.down("Meta")

        # Press the main key
        k = keys[-1]
        if k == "enter":
            self.active_page.keyboard.press("Enter")
        elif k == "tab":
            self.active_page.keyboard.press("Tab")
        elif k == "backspace":
            self.active_page.keyboard.press("Backspace")
        elif k == "delete":
            self.active_page.keyboard.press("Delete")
        elif k == "esc":
            self.active_page.keyboard.press("Escape")
        elif k == "space":
            self.active_page.keyboard.press("Space")
        elif k == "up":
            self.active_page.keyboard.press("ArrowUp")
        elif k == "down":
            self.active_page.keyboard.press("ArrowDown")
        elif k == "left":
            self.active_page.keyboard.press("ArrowLeft")
        elif k == "right":
            self.active_page.keyboard.press("ArrowRight")
        else:
            self.active_page.keyboard.press(k)

        # Release all modifier keys
        for k in reversed(keys[:-1]):
            if k == "ctrl":
                self.active_page.keyboard.up("Control")
            elif k == "shift":
                self.active_page.keyboard.up("Shift")
            elif k == "alt":
                self.active_page.keyboard.up("Alt")
            elif k == "cmd":
                self.active_page.keyboard.up("Meta")
        self._wait_for_load_state()

    def type(self, content: str):
        """Type content with support for escape characters."""
        self.active_page.keyboard.type(content)
        self._wait_for_load_state()

    def scroll(self, x: int, y: int, direction: str):
        """Scroll at specific coordinates in given direction."""
        self.active_page.mouse.move(x, y)
        if direction == "down":
            self.active_page.mouse.wheel(0, 1000)
        elif direction == "up":
            self.active_page.mouse.wheel(0, -1000)
        elif direction == "right":
            self.active_page.mouse.wheel(1000, 0)
        elif direction == "left":
            self.active_page.mouse.wheel(-1000, 0)
        self._wait_for_load_state()

    def wait(self):
        """Wait for 5 seconds."""
        self.active_page.wait_for_timeout(5000)

    def take_screenshot(self, path: str):
        """Take a screenshot of the active page."""
        self.active_page.screenshot(path=path)
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def goto_url(self, url: str):
        """Navigate to a URL."""
        self.active_page.goto(url)
        self.active_page.wait_for_load_state("networkidle", timeout=120000)
        self.active_page.wait_for_timeout(6000)

    def get_state(self) -> BrowserState:
        """Get current browser state."""
        self._wait_for_load_state()

        return BrowserState(
            page_url=self.active_page.url,
            page_screenshot_base64=f"data:image/png;base64,{self.take_screenshot(f'data/screenshots/screenshot_{time.time()}.png')}",
        )

    def close(self):
        """Close the browser."""
        self.context.close()
        self.driver.close()
        self.playwright.stop()


def main():
    base_path = "data/screenshots"
    os.makedirs(base_path, exist_ok=True)
    browser = Browser()
    browser.goto_url("https://apple.com/iphone")
    browser.take_screenshot(f"{base_path}/screenshot.png")
    browser.wait()
    browser.scroll(500, 500, "down")
    browser.take_screenshot(f"{base_path}/screenshot.png")
    browser.wait()
    browser.scroll(500, 500, "down")
    browser.take_screenshot(f"{base_path}/screenshot.png")


if __name__ == "__main__":
    main()

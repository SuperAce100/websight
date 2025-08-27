import base64
import os
import time
import asyncio
from pydantic import BaseModel
from playwright.sync_api import sync_playwright


class BrowserState(BaseModel):
    page_url: str
    page_screenshot_base64: str


class Browser:
    def __init__(self, show_browser: bool = False):
        try:
            asyncio.get_running_loop()
            print("⚠️ Warning: Detected running async loop, using sync API")
        except RuntimeError:
            pass

        self.playwright = sync_playwright().start()
        self.driver = self.playwright.chromium.launch(
            headless=not show_browser, timeout=120000
        )
        self.context = self.driver.new_context()
        self.active_page = self.context.new_page()

    def _wait_for_load_state(self):
        self.active_page.wait_for_timeout(5000)

    def click(self, x: int, y: int):
        self.active_page.mouse.click(x, y)
        self._wait_for_load_state()

    def left_double(self, x: int, y: int):
        self.active_page.mouse.dblclick(x, y)
        self._wait_for_load_state()

    def right_single(self, x: int, y: int):
        self.active_page.mouse.click(x, y, button="right")
        self._wait_for_load_state()

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int):
        self.active_page.mouse.move(start_x, start_y)
        self.active_page.mouse.down()
        self.active_page.mouse.move(end_x, end_y)
        self.active_page.mouse.up()
        self._wait_for_load_state()

    def hotkey(self, key: str):
        keys = key.lower().split()
        for k in keys:
            if k == "ctrl":
                self.active_page.keyboard.down("Control")
            elif k == "shift":
                self.active_page.keyboard.down("Shift")
            elif k == "alt":
                self.active_page.keyboard.down("Alt")
            elif k == "cmd":
                self.active_page.keyboard.down("Meta")
        k = keys[-1]
        keymap = {
            "enter": "Enter",
            "tab": "Tab",
            "backspace": "Backspace",
            "delete": "Delete",
            "esc": "Escape",
            "space": "Space",
            "up": "ArrowUp",
            "down": "ArrowDown",
            "left": "ArrowLeft",
            "right": "ArrowRight",
        }
        self.active_page.keyboard.press(keymap.get(k, k))
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
        self.active_page.keyboard.type(content)
        self._wait_for_load_state()

    def scroll(self, x: int, y: int, direction: str):
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
        self.active_page.wait_for_timeout(5000)

    def take_screenshot(self, path: str):
        self.active_page.screenshot(path=path)
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def goto_url(self, url: str):
        self.active_page.goto(url)
        self._wait_for_load_state()

    def get_state(self) -> BrowserState:
        self._wait_for_load_state()
        return BrowserState(
            page_url=self.active_page.url,
            page_screenshot_base64=f"data:image/png;base64,{self.take_screenshot(f'data/screenshots/screenshot_{time.time()}.png')}",
        )

    def close(self):
        self.context.close()
        self.driver.close()
        self.playwright.stop()

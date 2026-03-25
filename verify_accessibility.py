from playwright.sync_api import sync_playwright
import time

def verify_feature():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(record_video_dir="verification/video")
        page = context.new_page()

        page.goto('http://localhost:3000')
        page.wait_for_timeout(1000)

        # Verify Basic Tab
        page.keyboard.press("Tab") # Focus Basic Inference tab
        page.wait_for_timeout(500)

        # Take full page screenshot
        page.screenshot(path='verification.png', full_page=True)

        context.close()
        browser.close()

if __name__ == "__main__":
    verify_feature()

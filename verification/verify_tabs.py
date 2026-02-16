import re
from playwright.sync_api import sync_playwright, expect

def test_tabs():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Go to the local server
        page.goto("http://localhost:8000")

        # Verify title
        expect(page).to_have_title("BitNet WASM - Browser Integration Example")

        # Verify tab list role
        tablist = page.get_by_role("tablist")
        expect(tablist).to_be_visible()

        # Verify tabs exist
        tabs = page.get_by_role("tab")
        expect(tabs).to_have_count(5)

        # Verify basic tab is selected initially
        basic_tab = page.get_by_role("tab", name="Basic Inference")
        expect(basic_tab).to_have_attribute("aria-selected", "true")
        expect(basic_tab).to_have_class(re.compile(r"active"))

        # Verify streaming tab is not selected
        streaming_tab = page.get_by_role("tab", name="Streaming")
        expect(streaming_tab).to_have_attribute("aria-selected", "false")

        # Click streaming tab
        streaming_tab.click()

        # Verify selection changed
        expect(basic_tab).to_have_attribute("aria-selected", "false")
        expect(streaming_tab).to_have_attribute("aria-selected", "true")
        expect(streaming_tab).to_have_class(re.compile(r"active"))

        # Verify panel visibility
        streaming_panel = page.locator("#streaming-tab")
        expect(streaming_panel).to_be_visible()
        expect(streaming_panel).to_have_attribute("role", "tabpanel")

        basic_panel = page.locator("#basic-tab")
        expect(basic_panel).not_to_be_visible()

        # Take screenshot
        page.screenshot(path="verification/tabs_verified.png")
        print("Verification successful! Screenshot saved to verification/tabs_verified.png")

        browser.close()

if __name__ == "__main__":
    test_tabs()

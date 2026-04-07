from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto('file://' + __import__('os').path.abspath('examples/wasm/browser/index.html'))

    # Check tab structure
    tabs_html = page.evaluate('document.querySelector(".tabs").outerHTML')
    print("TABS HTML:\n", tabs_html)

    # Check what role and tab index they have
    tab_details = page.evaluate('''() => {
        return Array.from(document.querySelectorAll('.tab')).map(t => ({
            text: t.textContent,
            role: t.getAttribute('role'),
            tabindex: t.getAttribute('tabindex')
        }));
    }''')
    print("TAB DETAILS:\n", tab_details)

    browser.close()

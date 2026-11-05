#!/usr/bin/env python3
"""Check the exported model view with a real browser, using only bundle assets.

Run with kb-playwright-python validate_model_view_browser.py <export-output>.
The temporary server and Chrome belong to this process and close on exit.
"""

import functools
import http.server
from pathlib import Path
import re
import shutil
import sys
import threading
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright


output = Path(sys.argv[1]).resolve()
handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                            directory=str(output / "site"))
server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
origin = f"http://127.0.0.1:{server.server_port}"

try:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=shutil.which("google-chrome"), headless=True,
            args=["--no-sandbox"])
        try:
            for theme in ("light", "dark"):
                context = browser.new_context(
                    viewport={"width": 1400, "height": 1100},
                    color_scheme=theme, has_touch=True)
                try:
                    external = []
                    failed = []

                    def local_only(route):
                        if route.request.url.startswith(origin + "/"):
                            route.continue_()
                        else:
                            external.append(route.request.url)
                            route.abort()

                    context.route("**/*", local_only)
                    page = context.new_page()
                    page.on("response", lambda response: failed.append(
                        (response.status, response.url)) if response.status >= 400 else None)
                    page.goto(origin + "/index.html", wait_until="networkidle")
                    page.wait_for_function("document.querySelector('code.language-julia .token') !== null")
                    page.screenshot(path=str(output / f"source-{theme}.png"), full_page=True)

                    for name in ("shifted_curve", "centered_curve"):
                        component = page.get_by_role("tab", name=re.compile(r"ExportFixture\." + name + r"$"))
                        assert component.count() == 1  # repeated calls share one source tab
                        component.click()
                        assert component.get_attribute("aria-selected") == "true"
                        assert page.get_by_role("heading", name="Call instances", exact=True).is_visible()
                        panel = page.locator("#" + component.get_attribute("aria-controls"))
                        source = panel.locator("pre").first
                        assert "Core.@doc" not in source.inner_text()
                        assert source.evaluate("el => el.scrollWidth <= el.clientWidth"), name
                        page.wait_for_timeout(250)
                        page.screenshot(path=str(output / f"{name}-{theme}.png"), full_page=True)

                    roles = page.get_by_role("tab", name="Roles and ranks", exact=True)
                    roles.tap()
                    assert roles.get_attribute("aria-selected") == "true"
                    assert page.get_by_role("heading", name="Model quantities", exact=True).is_visible()
                    page.wait_for_timeout(250)  # let the tab underline transition finish
                    page.screenshot(path=str(output / f"roles-{theme}.png"), full_page=True)

                    roles.focus()
                    page.keyboard.press("End")
                    page.keyboard.press("Enter")
                    docs = page.get_by_role("tab", name="Documentation", exact=True)
                    assert docs.get_attribute("aria-selected") == "true"
                    assert page.get_by_role("heading", name="Authored documentation", exact=True).is_visible()
                    assert page.locator("strong", has_text="response scale").is_visible()
                    provenance = page.locator("summary", has_text="Declaration and provenance").first
                    provenance.focus()
                    page.keyboard.press("Enter")
                    assert provenance.evaluate("el => el.parentElement.open")
                    assert provenance.locator("..").get_by_text(
                        "validate_model_view_export.jl", exact=False).is_visible()
                    page.wait_for_timeout(250)
                    page.screenshot(path=str(output / f"provenance-{theme}.png"), full_page=True)
                    page.keyboard.press("Space")
                    assert not provenance.evaluate("el => el.parentElement.open")
                    summary = page.locator("summary", has_text="ExportFixture.center_predictor")
                    summary.focus()
                    page.keyboard.press("Enter")
                    assert summary.evaluate("el => el.parentElement.open")
                    assert page.get_by_text("Center the toy predictor.", exact=False).is_visible()
                    page.screenshot(path=str(output / f"documentation-{theme}.png"), full_page=True)
                    page.keyboard.press("Space")
                    assert not summary.evaluate("el => el.parentElement.open")

                    dimensions = page.evaluate("""() => ({
                        font: getComputedStyle(document.documentElement).fontSize,
                        viewport: document.documentElement.clientWidth,
                        content: document.documentElement.scrollWidth
                    })""")
                    assert dimensions["font"] == "16px", dimensions
                    assert dimensions["content"] <= dimensions["viewport"], dimensions
                    pdf = page.get_by_role("link", name="Download the complete model PDF")
                    response = context.request.get(urljoin(page.url, pdf.get_attribute("href")),
                                                   fail_on_status_code=True)
                    assert response.body().startswith(b"%PDF-")
                    assert not external, external
                    assert not failed, failed
                    print(f"{theme}: bundled syntax, touch tabs, keyboard tabs/docs, PDF and width passed", flush=True)
                finally:
                    context.close()

            # A collaborator can also open the exported HTML directly from
            # disk, without running either Julia or a local web server.
            context = browser.new_context()
            try:
                external = []
                errors = []

                def deny_network(route):
                    external.append(route.request.url)
                    route.abort()

                context.route("http://**/*", deny_network)
                context.route("https://**/*", deny_network)
                page = context.new_page()
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto((output / "site" / "index.html").as_uri(), wait_until="networkidle")
                page.wait_for_function("document.querySelector('code.language-julia .token') !== null")
                page.get_by_role("tab", name="Roles and ranks", exact=True).click()
                assert page.get_by_role("heading", name="Model quantities", exact=True).is_visible()
                page.get_by_role("tab", name="Documentation", exact=True).click()
                provenance = page.locator("summary", has_text="Declaration and provenance").first
                provenance.click()
                assert provenance.evaluate("el => el.parentElement.open")
                assert provenance.locator("..").locator("code.language-julia .token").count() > 0
                pdf = page.get_by_role("link", name="Download the complete model PDF")
                assert pdf.evaluate("el => el.href") == (output / "site" / "files" / "model.pdf").as_uri()
                assert not external, external
                assert not errors, errors
                print("file: bundled syntax, tabs, provenance and relative PDF link passed", flush=True)
            finally:
                context.close()
        finally:
            browser.close()
finally:
    server.shutdown()
    server.server_close()
    thread.join()

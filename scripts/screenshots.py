"""Take README screenshots and a demo GIF from the running Docker demo (`make demo` first).

Needs Playwright with a Chrome/Chromium install:  pip install playwright pillow
Usage: python scripts/screenshots.py [--url http://localhost:8070] [--out docs/screenshots]
"""

import argparse
import io
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

VIEWPORT = {"width": 1440, "height": 900}
HIDE_TOURS = (
    "document.addEventListener('DOMContentLoaded', () => { const s = document.createElement('style');"
    " s.textContent = '.o_tour_pointer { display: none !important; }'; document.head.appendChild(s); });"
)
RPC = """async ([model, method, args, kwargs]) => {
    const r = await fetch('/web/dataset/call_kw', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({jsonrpc: '2.0', method: 'call', params: {model, method, args, kwargs}})});
    return (await r.json()).result;
}"""


def settle(page) -> None:
    """Odoo keeps a live connection open (bus), so wait for the rendered view instead of network idle."""
    page.wait_for_selector(".o_action_manager .o_view_controller", timeout=30000)
    page.wait_for_timeout(1500)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8070")
    parser.add_argument("--db", default="demo")
    parser.add_argument("--out", type=Path, default=Path("docs/screenshots"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames = []

    def shot(page, name: str, frame: bool = True) -> None:
        png = page.screenshot()
        (args.out / f"{name}.png").write_bytes(png)
        if frame:
            frames.append(Image.open(io.BytesIO(png)).convert("RGB"))

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
        page.add_init_script(HIDE_TOURS)
        page.goto(f"{args.url}/web/login?db={args.db}")
        page.fill("input[name=login]", "admin")
        page.fill("input[name=password]", "admin")
        page.click("button[type=submit]")
        page.wait_for_selector(".o_main_navbar", timeout=30000)

        # Start from an unreviewed queue so the walkthrough is the same every time the script runs
        page.goto(f"{args.url}/odoo")
        page.wait_for_selector(".o_main_navbar")
        reviewed = page.evaluate(RPC, ["journal.anomaly.review", "search", [[["state", "!=", "to_review"]]], {}])
        if reviewed:
            page.evaluate(RPC, ["journal.anomaly.review", "action_reset", [reviewed], {}])

        page.goto(f"{args.url}/odoo/action-journal_anomaly.action_journal_anomaly_review")
        settle(page)
        shot(page, "01_review_queue")

        # Open a flagged duplicate: the clearest example of an entry nobody would want to miss
        ids = page.evaluate(
            RPC,
            [
                "journal.anomaly.review",
                "search",
                [[["reasons", "ilike", "Same amount"], ["state", "=", "to_review"]]],
                {"limit": 1, "order": "score desc"},
            ],
        )
        page.goto(f"{args.url}/odoo/action-journal_anomaly.action_journal_anomaly_review/{ids[0]}")
        page.wait_for_selector(".o_form_view")
        page.wait_for_timeout(1500)
        shot(page, "02_flagged_entry")

        page.locator("button[name=action_true_issue]").first.click()
        page.wait_for_timeout(1500)
        shot(page, "03_marked_true_issue")

        page.goto(f"{args.url}/odoo/action-journal_anomaly.action_journal_anomaly_review")
        settle(page)
        page.locator(".o_searchview_facet .o_facet_remove").first.click()  # drop the default "To Review" filter
        settle(page)
        page.locator(".o_searchview_dropdown_toggler").click()
        page.get_by_text("True Issues", exact=True).click()
        page.keyboard.press("Escape")
        page.mouse.move(700, 600)  # no hover effect on the search bar
        page.evaluate("document.activeElement && document.activeElement.blur()")  # no focus ring either
        settle(page)
        shot(page, "04_true_issues")
        browser.close()

    gif = args.out / "demo.gif"
    small = [f.resize((960, 600), Image.Resampling.LANCZOS) for f in frames]
    small[0].save(gif, save_all=True, append_images=small[1:], duration=[2500, 4000, 2500, 3000], loop=0, optimize=True)
    print(f"{len(frames)} frames -> {gif}; screenshots in {args.out}")


if __name__ == "__main__":
    main()

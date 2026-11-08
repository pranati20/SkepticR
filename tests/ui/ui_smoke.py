"""Playwright smoke test of the real UI against a running server.
    python -m uvicorn skeptic.backend.app:app --port 8000   (in another shell)
    python skeptic/tests/ui/ui_smoke.py [http://localhost:8000]
"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
OUT = Path(__file__).parent / "shots"
OUT.mkdir(exist_ok=True)

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1500, "height": 1000})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.goto(URL)

    # prompt-first flow is the default: question -> dataset -> audit -> conclusion from valid findings only
    assert "what do you want to conclude" in pg.inner_text("body").lower(), "conclude tab missing"
    pg.locator("textarea").first.fill("Does the new design increase conversion, and for whom?")
    pg.get_by_role("button", name="Find the valid conclusion").click()
    pg.wait_for_function("document.body.innerText.includes('Left out and not used')", timeout=180000)
    low = pg.inner_text("body").lower()
    assert "valid findings used" in low or "no valid conclusion" in low, "conclusion missing"
    pg.screenshot(path=str(OUT / "conclude.png"), full_page=True)
    assert "find papers & data" not in pg.inner_text("body").lower(), "removed tab is back"

    # court: replay a recorded run
    pg.get_by_role("button", name="Watch it work").click()
    pg.locator("select").first.select_option("market")
    pg.get_by_text("Options", exact=True).click()
    pg.get_by_role("button", name="instant").click()
    pg.get_by_role("button", name="▶ Start").click()
    pg.wait_for_function("document.body.innerText.includes('strategy 1 of')", timeout=90000)
    pg.wait_for_timeout(2500)
    low = pg.inner_text("body").lower()
    assert "cost of illusion" in low and "thrown out" in low, "cost of illusion card missing"
    assert "ideas that were just luck" in low, "luck list missing"
    pg.screenshot(path=str(OUT / "court.png"), full_page=True)
    pg.get_by_role("button", name="Show technical details").click()
    assert "credibility meter" in pg.inner_text("body").lower(), "technical details missing"

    pg.get_by_role("button", name="AI vs Skeptic").click(); pg.wait_for_timeout(600)
    assert "naive ai analyst" in pg.inner_text("body").lower()
    pg.screenshot(path=str(OUT / "split.png"), full_page=True)

    assert pg.get_by_text("Export report (PDF)").count() == 1, "export button missing"
    pg.get_by_role("button", name="Improve the algorithm").click()
    pg.wait_for_function("document.body.innerText.toLowerCase().includes('equations from the paper')", timeout=90000)
    assert pg.locator(".katex").count() > 0, "equations not rendered"
    pg.screenshot(path=str(OUT / "methods.png"), full_page=True)
    pg.get_by_role("button", name="Papers it used").click(); pg.wait_for_timeout(800)
    assert "papers & datasets it has looked at" in pg.inner_text("body").lower(), "papers tab missing"
    pg.screenshot(path=str(OUT / "papers.png"), full_page=True)

    pg.get_by_role("button", name="Check a paper").click(); pg.wait_for_timeout(500)
    pg.screenshot(path=str(OUT / "replication.png"), full_page=True)
    b.close()

if errors:
    print("console errors:", errors)
    sys.exit(1)
print("UI smoke OK")

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
    pg.get_by_role("button", name="instant").click()
    pg.get_by_role("button", name="▶ Replay cached run").click()
    pg.wait_for_function("/graveyard · [1-9]/i.test(document.body.innerText) && document.body.innerText.includes('Ground truth check')", timeout=90000)
    pg.wait_for_timeout(2500)
    txt = pg.inner_text("body")
    low = txt.lower()
    assert "credibility meter" in low and "ideas tried" in low, "meters missing"
    assert "killed ideas" in low, "graveyard missing"
    pg.screenshot(path=str(OUT / "court.png"), full_page=True)
    pg.locator("text=Died").first.click()
    pg.wait_for_timeout(800)
    assert "prosecutor results" in pg.inner_text("body").lower(), "detail drawer did not open"
    pg.screenshot(path=str(OUT / "drawer.png"))
    pg.keyboard.press("Escape")
    pg.get_by_role("button", name="Naive AI vs Skeptic").click(); pg.wait_for_timeout(600)
    assert "naive ai analyst" in pg.inner_text("body").lower(); pg.screenshot(path=str(OUT / "split.png"), full_page=True)
    pg.get_by_role("button", name="Benchmark").click(); pg.wait_for_timeout(2500)
    t = pg.inner_text("body"); assert "skeptic (full court)" in t.lower() and "where skeptic fails" in t.lower()
    pg.screenshot(path=str(OUT / "benchmark.png"), full_page=True)
    pg.get_by_role("button", name="Replication").click(); pg.get_by_role("button", name="Replay example").click(); pg.wait_for_timeout(5000)
    pg.get_by_role("button", name="instant").count()
    pg.screenshot(path=str(OUT / "replication.png"), full_page=True)
    b.close()
real = [e for e in errors if "favicon" not in e]
print("console/page errors:", real if real else "none")
print("UI smoke test PASSED" if not real else "UI smoke test finished with errors")

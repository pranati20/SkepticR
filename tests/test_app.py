"""HTTP layer + Librarian tests (the API had no tests before)."""
import os
import time

os.environ["SKEPTIC_OFFLINE"] = "1"          # tests never touch the network unless they mock it

import numpy as np
import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from skeptic.backend import app as A
from skeptic.backend import literature as LIT
from skeptic.backend.court import stats as S
from skeptic.backend.court.alpha_investing import AlphaInvesting

client = TestClient(A.app)


def test_health():
    assert client.get("/api/health").json()["ok"] is True


@pytest.mark.skipif(not A.DIST.exists(), reason="frontend not built")
@pytest.mark.parametrize("path", ["/%2e%2e%2f%2e%2e%2f%2e%2e%2fetc/passwd", "/..%2f..%2fREADME.md", "/%2e%2e/%2e%2e/README.md",
                                  "/assets/../../README.md", "/..\\..\\README.md"])
def test_spa_path_traversal_serves_index_only(path):
    r = client.get(path)
    assert r.status_code == 200
    assert "<div id=\"root\"" in r.text or "<!doctype html" in r.text.lower()
    assert "Skeptic: an AI researcher" not in r.text        # README must never leak


def test_unknown_scenario_404():
    assert client.post("/api/run", json={"scenario": "nope"}).status_code == 404


def test_bad_seed_emits_error_event_not_silence():
    rid = client.post("/api/run", json={"scenario": "ab_test", "seed": "not-a-number"}).json()["run_id"]
    for _ in range(50):
        d = client.get(f"/api/runs/{rid}/events").json()
        if d["done"]:
            break
        time.sleep(0.1)
    assert d["done"] and any(e["type"] == "error" for e in d["events"])


def test_runs_are_evicted():
    for _ in range(A.MAX_RUNS + 5):
        client.post("/api/run", json={"scenario": "nope2"})            # 404s do not register
    ids = [client.post("/api/run", json={"scenario": "ab_test", "seed": "x"}).json()["run_id"] for _ in range(A.MAX_RUNS + 5)]
    assert len(A.RUNS) <= A.MAX_RUNS and ids[0] not in A.RUNS


@pytest.mark.parametrize("body", [{}, {"csv": None}, {"csv": 5}, {"csv": ""}, {"csv": "\x00\x00"}])
def test_upload_bad_input_is_4xx(body):
    assert client.post("/api/upload", json=body).status_code in (400, 413, 422)


def test_upload_ok():
    r = client.post("/api/upload", json={"csv": "a,b\n1,2\n3,4\n"})
    assert r.status_code == 200 and r.json()["rows"] == 2


def test_nullrerun_keyed_on_seed_and_clamped():
    a = client.post("/api/nullrerun", json={"n_worlds": 50, "seed": 0}).json()
    b = client.post("/api/nullrerun", json={"n_worlds": 10**6, "seed": 1}).json()
    assert b["n_worlds"] == A.MAX_WORLDS
    assert a["planted"][0]["stat"] != b["planted"][0]["stat"]            # different seed -> different data
    assert client.post("/api/nullrerun", json={"seed": "x"}).status_code == 400


def test_cors_not_wildcard():
    r = client.options("/api/run", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") != "*" and "evil" not in r.headers.get("access-control-allow-origin", "")


# ---------------- statistics robustness
@pytest.mark.parametrize("make", [lambda f: np.hstack([f, f]),                       # duplicated factor
                                  lambda f: np.hstack([f, np.ones((len(f), 1))]),    # constant factor (collinear with intercept)
                                  lambda f: np.hstack([f, np.zeros((len(f), 1))])])  # zero factor
def test_alpha_t_survives_degenerate_factors(make):
    rng = np.random.default_rng(0)
    f = rng.standard_normal((300, 1))
    y = rng.standard_normal((4, 300)) + 0.5                       # real intercept -> must not collapse to t ~ 0
    t = S.alpha_t(y, make(f))
    assert np.all(np.isfinite(t)) and np.all(np.abs(t) > 3)


def test_alpha_t_too_few_rows():
    assert np.all(S.alpha_t(np.random.randn(3, 2), np.random.randn(2, 2)) == 0)


def test_alpha_investing_reward_never_exceeds_level_and_floor():
    ai = AlphaInvesting(payout=0.04)
    w = ai.wealth
    ai.update(0.01, True)
    assert ai.wealth - w <= 0.01 + 1e-12
    ai.wealth = 1e-6
    assert ai.propose_level(0.5, min_level=1 / 501) >= 1 / 501


# ---------------- Librarian: live results are passed through, never invented
def test_librarian_live_uses_only_api_results(monkeypatch):
    fake = [{"title": "Short-term reversal in stock returns", "authors": "Doe, Roe", "year": 2012, "doi": "10.1/x", "url": "https://doi.org/10.1/x",
             "venue": "J Fin", "abstract": "We study short-term reversal of weekly stock returns.", "source": "Crossref (live)"}]
    monkeypatch.setattr(LIT, "live_search", lambda q, rows=5: (fake, ["Crossref: 1 results", "arXiv: unavailable (URLError)"]))
    lib = LIT.Librarian(LIT.Scenario(start_year=2015, end_year=2025), online=True)
    found = lib.find("short-term reversal weekly stock returns", "mean_reversion")
    live = [p for p in found if p["source"].endswith("(live)")]
    assert live and live[0]["url"] == "https://doi.org/10.1/x" and live[0]["expected_haircut"] > 0       # post-publication decay flagged
    assert all(p["source"].startswith("curated index") for p in found if p not in live)                    # fallback is labelled
    assert "arXiv: unavailable (URLError)" in lib.status


def test_librarian_offline_has_no_live_papers():
    lib = LIT.Librarian(LIT.Scenario(), online=False)
    assert all(p["source"].startswith("curated index") for p in lib.find("momentum winners", "momentum"))


def test_dotenv_loader(tmp_path, monkeypatch):
    import importlib
    import skeptic.backend as B
    monkeypatch.delenv("SKEPTIC_TEST_KEY", raising=False)
    monkeypatch.setattr(B, "__file__", str(tmp_path / "backend" / "__init__.py"))
    (tmp_path / ".env").write_text("# c\nSKEPTIC_TEST_KEY=abc\nEMPTY=\n", encoding="utf-8")
    B._load_dotenv()
    assert os.environ["SKEPTIC_TEST_KEY"] == "abc" and "EMPTY" not in os.environ
    monkeypatch.delenv("SKEPTIC_TEST_KEY", raising=False)


def test_gemini_provider_request_and_parse(monkeypatch):
    import io
    import json as J
    import urllib.request
    from skeptic.backend import llm as L
    seen = {}

    class R(io.BytesIO):
        __enter__ = lambda s: s
        __exit__ = lambda s, *a: None

    def fake(req, timeout=0):
        seen["url"], seen["key"], seen["body"] = req.full_url, req.headers.get("X-goog-api-key"), J.loads(req.data)
        return R(J.dumps({"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}],
                          "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3}}).encode())
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    monkeypatch.setenv("ARGON_API_KEY", "k")
    monkeypatch.setenv("SKEPTIC_USE_LLM", "1")
    llm = L.LLM(use_cache=False)
    assert isinstance(llm.provider, L.GeminiProvider)
    obj, u = llm.complete_json("judge", "sys", "prompt")
    assert obj == {"ok": True} and (u.tokens_in, u.tokens_out) == (7, 3) and u.model == "gemini-4-argon"
    assert seen["url"].endswith("/models/gemini-4-argon:generateContent") and seen["key"] == "k"

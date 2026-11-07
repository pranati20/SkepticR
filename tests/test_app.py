"""HTTP layer + Librarian tests (the API had no tests before)."""
import os
import time

import tempfile
os.environ["SKEPTIC_LIBRARY_PATH"] = os.path.join(tempfile.mkdtemp(), "papers.json")   # tests never touch the real library
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
    assert found == live                                                                                   # no hardcoded papers mixed in
    assert "arXiv: unavailable (URLError)" in lib.status


def test_librarian_offline_finds_nothing_and_there_is_no_builtin_list():
    lib = LIT.Librarian(LIT.Scenario(), online=False)
    assert lib.find("momentum winners", "momentum") == [] and "offline" in lib.status[0]
    assert not hasattr(LIT, "CORPUS")


def test_openalex_parses_real_shape(monkeypatch):
    import json as J
    monkeypatch.setenv("OPENALEX_API_KEY", "k")
    body = {"results": [{"title": "Momentum &amp; value", "authorships": [{"author": {"display_name": "Ann Lee"}}], "publication_year": 2010,
                         "doi": "https://doi.org/10.1/q", "id": "https://openalex.org/W1", "cited_by_count": 42,
                         "primary_location": {"source": {"display_name": "J Fin"}}, "abstract_inverted_index": {"momentum": [0], "works": [1]}}]}
    monkeypatch.setattr(LIT, "_get", lambda url, t: J.dumps(body).encode())
    monkeypatch.setattr(LIT, "_cached", lambda kind, q, fn: fn())
    r = LIT.openalex("momentum")[0]
    assert r["title"] == "Momentum & value" and r["abstract"] == "momentum works" and r["cited_by"] == 42 and r["url"].startswith("https://doi.org")


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


# ---------------- chat agent
def _fake_sources(monkeypatch):
    from skeptic.backend import chat as C
    paper = {"kind": "paper", "title": "Momentum in stock returns", "by": "Doe", "year": 2012, "venue": "J", "url": "https://doi.org/10.1/p",
             "snippet": "abstract", "source": "Crossref (live)"}
    data = {"kind": "dataset", "title": "org/stock-prices", "by": "org", "year": "2024", "venue": "1 downloads", "url": "https://huggingface.co/datasets/org/stock-prices",
            "snippet": "daily prices", "source": "Hugging Face (live)"}
    monkeypatch.setitem(C.TOOLS, "search_papers", lambda q, rows=5: ([paper], ["Crossref: 1 results"]))
    monkeypatch.setitem(C.TOOLS, "search_datasets", lambda q, rows=5: ([data], ["Hugging Face: 1 results"]))
    return C


def test_chat_offline_returns_only_real_results(monkeypatch):
    C = _fake_sources(monkeypatch)
    r = C.run_chat([{"role": "user", "content": "momentum stock returns"}])
    assert [s["url"] for s in r["sources"]] == ["https://doi.org/10.1/p", "https://huggingface.co/datasets/org/stock-prices"]
    assert "1 paper" in r["answer"] and "No AI model" in r["answer"]


def test_chat_strips_invented_links_and_citations(monkeypatch):
    C = _fake_sources(monkeypatch)

    class Fake:
        is_mock = False
        class provider: name = "fake"
        n = 0
        def register_mock(self, *a): pass
        def complete_json(self, role, system, prompt, schema, mock_hint=None, max_tokens=0):
            self.n += 1
            if self.n == 1:
                return {"action": "search_papers", "query": "momentum"}, None
            return {"action": "answer", "answer": "See [S1] and [S9] or http://made-up.example/paper"}, None
    r = C.run_chat([{"role": "user", "content": "momentum?"}], llm=Fake())
    assert "[S1]" in r["answer"] and "[S9]" not in r["answer"] and "made-up.example" not in r["answer"]


def test_chat_llm_failure_still_gives_results(monkeypatch):
    C = _fake_sources(monkeypatch)

    class Boom:
        is_mock = False
        class provider: name = "x"
        def register_mock(self, *a): pass
        def complete_json(self, *a, **k): raise RuntimeError("401 bad key")
    r = C.run_chat([{"role": "user", "content": "q"}], llm=Boom())
    assert r["error"] and len(r["sources"]) == 2 and "could not be reached" in r["answer"]


def test_chat_endpoint_validation(monkeypatch):
    _fake_sources(monkeypatch)
    assert client.post("/api/chat", json={}).status_code == 400
    assert client.post("/api/chat", json={"messages": [{"role": "assistant", "content": "hi"}]}).status_code == 400
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "momentum"}]})
    assert r.status_code == 200 and len(r.json()["sources"]) == 2


def test_illusion_events_compare_backtest_with_reality():
    from skeptic.backend.runner import run_scenario
    ev = []
    run_scenario("synthetic", {"n_worlds": 100}, ev.append, 0)
    il = [e for e in ev if e["type"] == "illusion"]
    assert il and len(il) <= 8
    for e in il:
        assert e["in_total"] > 0 and len(e["in_curve"]) == len(e["real_curve"]) and e["basis"] and e["cause"]
        assert abs(e["in_curve"][-1] - e["in_total"]) < 1e-6


def test_library_records_and_lists(monkeypatch):
    from skeptic.backend import library
    library.record([{"title": "A real paper", "authors": "Doe", "year": 2020, "url": "https://doi.org/10.1/a", "source": "Crossref (live)"}], "momentum: test")
    library.record([{"title": "A real paper", "url": "https://doi.org/10.1/a", "source": "Crossref (live)"}], "chat: another question")
    d = client.get("/api/library").json()
    e = [x for x in d["looked_at"] if x["url"] == "https://doi.org/10.1/a"][0]
    assert e["uses"] == 2 and len(e["contexts"]) == 2 and "reference" not in d
    assert client.get("/api/benchmark").status_code in (200, 404) and client.post("/api/nullrerun", json={}).status_code in (404, 405)


def test_chat_context_steers_run_and_reaches_prosecutors(monkeypatch):
    from skeptic.backend import runner as R, literature as LIT
    monkeypatch.setattr(LIT, "live_search", lambda q, n=6: ([], ["none"]))
    ctx = {"question": "Does short-term reversal work in stocks?",
           "papers": [{"kind": "paper", "title": "Short-term reversal in stock returns", "by": "A. Lee", "year": 2005, "url": "https://doi.org/10.1/r",
                       "snippet": "Short-term reversal of stock returns after one week", "source": "OpenAlex"},
                      {"kind": "dataset", "title": "ignored dataset", "url": "https://x"}]}
    assert len(R.clean_context(ctx)["papers"]) == 1 and R.clean_context("junk") is None
    ev, out = R.run_scenario("market", {"n_worlds": 60, "rounds": 1, "per_hunter": 4, "context": ctx})
    fams = {e["family"] for e in ev if e["type"] == "proposal"}
    assert fams == {"mean_reversion"}                                   # question steered the hunters
    lits = [e for e in ev if e["type"] == "literature"]
    assert lits and any(p.get("from_chat") for e in lits for p in e["papers"])   # chat paper reached the Librarian evidence


def test_tear_sheet_pdf_for_demo_and_live_run():
    r = client.get("/api/report", params={"demo": "betting"})
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf" and r.content[:5] == b"%PDF-" and len(r.content) > 2000
    assert client.get("/api/report").status_code == 400
    assert client.get("/api/report", params={"demo": "nope"}).status_code == 404
    assert client.get("/api/report", params={"run": "nope"}).status_code == 404
    rid = client.post("/api/run", json={"scenario": "synthetic", "params": {"n_worlds": 60, "rounds": 1, "per_hunter": 3}}).json()["run_id"]
    import time
    for _ in range(300):
        d = client.get(f"/api/runs/{rid}/events").json()
        if d["done"]:
            break
        time.sleep(0.5)
    r = client.get("/api/report", params={"run": rid})
    assert r.status_code == 200 and r.content[:5] == b"%PDF-"


def test_methods_research_uses_only_live_results_and_verbatim_equations(monkeypatch):
    from skeptic.backend import methods as M, literature as LIT
    page = r'<h2>2 Method</h2><math class="ltx_Math" alttext="\alpha_{t}=\gamma_{t}(R_{1},\dots,R_{t-1})" display="block"></math><math alttext="x" display="inline"></math>'
    monkeypatch.setattr(LIT, "_get", lambda url, t: page.encode())
    monkeypatch.setattr(LIT, "_cached", lambda kind, q, fn: fn())
    eq = M.equations("1710.00499")
    assert len(eq) == 1 and eq[0]["latex"].startswith(r"\alpha_{t}=") and eq[0]["section"] == "2 Method"          # inline math ignored, LaTeX verbatim
    assert M._arxiv_id("https://doi.org/10.48550/arxiv.1809.07292") == "1809.07292" and M._arxiv_id("https://doi.org/10.3905/x") is None
    paper = {"title": "Online control of the false discovery rate", "authors": "A", "year": 2017, "venue": "", "url": "http://arxiv.org/abs/1710.00499v1",
             "abstract": "online false discovery rate alpha-investing procedure", "source": "arXiv (live)"}
    monkeypatch.setattr(LIT, "live_search", lambda q, n=6: ([paper], ["arXiv: 1 results"]))
    M._CACHE.clear()
    monkeypatch.delenv("SKEPTIC_OFFLINE", raising=False)
    d = M.research()
    got = [p for t in d["topics"] for p in t["papers"]]
    assert got and all(p["title"] == paper["title"] for p in got) and got[0]["equations"]
    c = M.conclusion(d, "three bullets")
    assert "[S1]" in c["text"] and "http" not in c["text"] and c["llm"].startswith("offline")
    M._CACHE.clear()


def test_methods_endpoint_offline_finds_nothing():
    r = client.post("/api/methods", json={"style": "short"})
    assert r.status_code == 200 and r.json()["topics"] and all(t["papers"] == [] for t in r.json()["topics"])
    assert "nothing to conclude" in r.json()["conclusion"]["text"]
    assert client.post("/api/methods", json={"style": 5}).status_code == 400


def test_conclusion_uses_only_valid_findings(tmp_path):
    import json
    from skeptic.backend import conclude as CC, runner as R
    from skeptic.backend.llm import LLM

    seen = {}

    class Spy:
        name = "spy"

        def complete(self, model, system, prompt, max_tokens):
            seen["prompt"] = prompt
            return json.dumps({"text": "Supported by [F1] and [F99]. See https://evil.example"}), 1, 1

    ev, _ = R.run_scenario("health", {"n_worlds": 80, "question": "Do supplements help?"})
    c = next(e for e in ev if e["type"] == "conclusion")
    verd = {}
    for e in ev:
        if e["type"] == "verdict":
            verd[e["hypothesis_id"]] = e
    valid_ids = {k for k, v in verd.items() if v["verdict"] == "DISCOVERY"}
    assert {f["id"] for f in c["findings"]} <= valid_ids and c["status"] in ("valid", "none")
    rejected = [v["description"] for v in verd.values() if v["verdict"] != "DISCOVERY"]
    assert rejected
    import re as _re
    listed = [_re.split(r" \(|;", ln.split("] ", 1)[1])[0] for ln in c["text"].splitlines() if ln.startswith("- [F")]
    assert set(listed) <= {verd[k]["description"] for k in valid_ids}                # the summary lists valid findings only
    if valid_ids:                                                                    # the model prompt never contains rejected ideas either
        out = CC.conclude("Do supplements help?", ev, LLM(provider=Spy(), cache_dir=tmp_path, use_cache=False))
        assert not any(("] " + d + " |") in seen["prompt"] for d in rejected) and "[F99]" not in out["text"] and "http" not in out["text"] and "[F1]" in out["text"]
    none = CC.conclude("Does anything increase it?", [{"type": "verdict", "hypothesis_id": "a", "verdict": "LUCK", "description": "lucky idea"}])
    assert none["status"] == "none" and "No valid conclusion" in none["text"] and "lucky idea" not in none["text"]

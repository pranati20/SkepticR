"""Single LLM abstraction: providers, role-based model routing, on-disk cache, token/cost meter, MockLLM.

Principles
  * Statistics do the heavy lifting; LLMs only propose hypotheses, prosecute, read papers and write reports.
  * Cheap model for bulk generation (Hunters/Cleaner), stronger model for Prosecutor/Judge/report.
  * Every call is cached on disk by prompt hash, so demos replay offline and re-runs cost nothing.
  * Structured JSON in/out with schema validation. LLM text is DATA: it is never eval/exec'd anywhere.
  * No API key needed: the default provider is MockLLM (deterministic, schema-valid).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import jsonschema
except Exception:  # pragma: no cover
    jsonschema = None

ROLE_MODELS = {
    "cleaner": "claude-haiku-5-5",
    "hunter": "claude-haiku-5-5",
    "extractor": "claude-sonnet-5-5",
    "prosecutor": "claude-sonnet-5-5",
    "judge": "claude-sonnet-5-5",
    "report": "claude-sonnet-5-5",
}
# USD per 1M tokens (input, output). ESTIMATES for the cost meter; edit to match your price sheet.
PRICES = {"claude-haiku-5-5": (1.0, 5.0), "claude-sonnet-5-5": (3.0, 15.0), "gemini-4-argon": (0.0, 0.0), "mock": (0.0, 0.0)}      # Argon price unknown: cost meter shows $0 until you set it

CACHE_DIR = Path(os.environ.get("SKEPTIC_LLM_CACHE", Path(__file__).resolve().parents[1] / ".llm_cache"))


@dataclass
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    model: str = "mock"


class TokenMeter:
    """Live token + dollar meter by role, plus a counter of zero-token statistical work."""

    def __init__(self):
        self._lock = threading.Lock()
        self.by_role: dict[str, dict] = {}
        self.stat_ops: dict[str, int] = {}
        self.listeners: list[Callable[[dict], None]] = []

    def add(self, role: str, u: Usage):
        with self._lock:
            r = self.by_role.setdefault(role, {"calls": 0, "cached_calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0})
            r["calls"] += 1
            r["cached_calls"] += int(u.cached)
            r["tokens_in"] += u.tokens_in
            r["tokens_out"] += u.tokens_out
            r["cost_usd"] += 0.0 if u.cached else u.cost_usd
        self._emit()

    def stat(self, name: str, n: int = 1):
        """Count zero-token work (null worlds evaluated, backtests, regressions...)."""
        with self._lock:
            self.stat_ops[name] = self.stat_ops.get(name, 0) + n

    def snapshot(self) -> dict:
        with self._lock:
            tot_tok = sum(r["tokens_in"] + r["tokens_out"] for r in self.by_role.values())
            return {"by_role": json.loads(json.dumps(self.by_role)), "stat_ops": dict(self.stat_ops),
                    "total_tokens": tot_tok, "total_cost_usd": sum(r["cost_usd"] for r in self.by_role.values()),
                    "llm_calls": sum(r["calls"] for r in self.by_role.values()),
                    "zero_token_ops": sum(self.stat_ops.values())}

    def _emit(self):
        snap = None
        for f in self.listeners:
            snap = snap or self.snapshot()
            try:
                f(snap)
            except Exception:
                pass


def _validate(obj: Any, schema: dict | None):
    if schema is None or jsonschema is None:
        return
    jsonschema.validate(obj, schema)


def _extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("\n") + 1:] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        a, b = min([i for i in (text.find("{"), text.find("[")) if i >= 0] or [0]), max(text.rfind("}"), text.rfind("]"))
        return json.loads(text[a:b + 1])


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None = None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def complete(self, model: str, system: str, prompt: str, max_tokens: int) -> tuple[str, int, int]:
        msg = self.client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                          messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return text, msg.usage.input_tokens, msg.usage.output_tokens


class GeminiProvider:
    """Gemini 4 Argon (gemini-4-argon) via the Gemini `generateContent` REST API. Key: ARGON_API_KEY.

    ASSUMPTION (unverified without a key): the endpoint is Google's Gemini API shape. If Argon is served from a different
    gateway, set ARGON_BASE_URL (default https://generativelanguage.googleapis.com/v1beta) and ARGON_MODEL."""
    name = "gemini"

    def __init__(self, api_key: str | None = None):
        self.key = api_key or os.environ["ARGON_API_KEY"]
        self.base = os.environ.get("ARGON_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")

    def complete(self, model: str, system: str, prompt: str, max_tokens: int) -> tuple[str, int, int]:
        import urllib.request
        body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens, "responseMimeType": "application/json"}}
        req = urllib.request.Request(f"{self.base}/models/{model}:generateContent", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "x-goog-api-key": self.key})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read().decode())
        text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
        u = d.get("usageMetadata", {})
        return text, int(u.get("promptTokenCount", 0)), int(u.get("candidatesTokenCount", 0))


class MockLLM:
    """Deterministic offline provider. Role handlers receive the structured `hint` and return schema-valid JSON.
    Hunters etc. pass a hint dict; the mock never parses prose, so it is robust and reproducible."""
    name = "mock"

    def __init__(self):
        self.handlers: dict[str, Callable[[dict], Any]] = {}

    def register(self, role: str, fn: Callable[[dict], Any]):
        self.handlers[role] = fn


class LLM:
    def __init__(self, provider: Any | None = None, meter: TokenMeter | None = None, cache_dir: Path | None = None,
                 use_cache: bool = True, max_retries: int = 3):
        if provider is None:
            want = os.environ.get("SKEPTIC_USE_LLM")
            pick = os.environ.get("SKEPTIC_LLM_PROVIDER", "").lower()     # "gemini" | "anthropic" (default: whichever key exists, Argon first)
            if want and os.environ.get("ARGON_API_KEY") and pick in ("", "gemini"):
                provider = GeminiProvider()
            elif want and os.environ.get("ANTHROPIC_API_KEY") and pick in ("", "anthropic"):
                provider = AnthropicProvider()
            else:
                provider = MockLLM()
        self.provider = provider
        self.meter = meter or TokenMeter()
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.use_cache = use_cache
        self.max_retries = max_retries

    @property
    def is_mock(self) -> bool:
        return isinstance(self.provider, MockLLM)

    def register_mock(self, role: str, fn: Callable[[dict], Any]):
        if self.is_mock:
            self.provider.register(role, fn)

    def _key(self, model, system, prompt, schema) -> str:
        return hashlib.sha256(json.dumps([model, system, prompt, schema], sort_keys=True).encode()).hexdigest()

    def complete_json(self, role: str, system: str, prompt: str, schema: dict | None = None,
                      mock_hint: dict | None = None, max_tokens: int = 2000) -> tuple[Any, Usage]:
        if self.is_mock:
            model = "mock"
        elif isinstance(self.provider, GeminiProvider):
            model = os.environ.get("ARGON_MODEL", "gemini-4-argon")        # one flagship model for every role
        else:
            model = ROLE_MODELS.get(role, "claude-sonnet-5-5")
        if self.is_mock:
            handler = self.provider.handlers.get(role)
            if handler is None:
                raise RuntimeError(f"MockLLM has no handler for role '{role}'")
            obj = handler(mock_hint or {})
            _validate(obj, schema)
            u = Usage(tokens_in=len(prompt) // 4, tokens_out=len(json.dumps(obj)) // 4, cost_usd=0.0, cached=False, model="mock")
            self.meter.add(role, u)
            return obj, u
        key = self._key(model, system, prompt, schema)
        path = self.cache_dir / f"{key}.json"
        if self.use_cache and path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            u = Usage(d["tokens_in"], d["tokens_out"], d["cost_usd"], True, model)
            self.meter.add(role, u)
            return d["obj"], u
        last_err = None
        for attempt in range(self.max_retries):
            try:
                text, tin, tout = self.provider.complete(model, system, prompt, max_tokens)
                obj = _extract_json(text)
                _validate(obj, schema)
                pin, pout = PRICES.get(model, (3.0, 15.0))
                u = Usage(tin, tout, (tin * pin + tout * pout) / 1e6, False, model)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"obj": obj, "tokens_in": tin, "tokens_out": tout, "cost_usd": u.cost_usd}),
                                encoding="utf-8")
                self.meter.add(role, u)
                return obj, u
            except Exception as e:  # malformed JSON / schema / transient API error -> retry with backoff
                last_err = e
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"LLM call failed for role '{role}' after {self.max_retries} tries: {last_err}")

"""LLM-assisted Prosecutor: reviews a hypothesis SPEC (not data) for leakage / survivorship / data-snooping smells.
Advisory only (severity 'medium'): deterministic re-runs are what can kill a finding."""
from __future__ import annotations

from ..llm import LLM

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {"concerns": {"type": "array", "items": {"type": "object", "properties": {
        "kind": {"enum": ["lookahead", "survivorship", "overfitting", "data_snooping", "other"]},
        "detail": {"type": "string"}}, "required": ["kind", "detail"]}},
                   "verdict": {"enum": ["clean", "suspicious"]}},
    "required": ["concerns", "verdict"],
}


def mock_review(hint: dict) -> dict:
    spec, desc = hint["spec"], hint.get("description", "")
    c = []
    if spec.get("uses_future_bar"):
        c.append({"kind": "lookahead", "detail": "formula or description references next-bar information"})
    if spec.get("universe") == "current_constituents":
        c.append({"kind": "survivorship", "detail": "universe is today's listed names"})
    if len(spec.get("windows", [])) >= 3:
        c.append({"kind": "overfitting", "detail": "many tunable windows"})
    return {"concerns": c, "verdict": "suspicious" if c else "clean"}


def register_mock(llm: LLM):
    llm.register_mock("prosecutor", mock_review)


def make_llm_review(llm: LLM):
    def run(search, k):
        h = search.hyps[k]
        prompt = (f"Review this hypothesis spec as a hostile peer reviewer. Look for lookahead bias, survivorship bias, "
                  f"overfitting and data snooping. Spec: {h['spec']}\nDescription: {h.get('description')}\n"
                  f"Rationale: {h.get('rationale')}")
        obj, _ = llm.complete_json("prosecutor", "You are an adversarial quantitative reviewer. JSON only.", prompt,
                                   REVIEW_SCHEMA, mock_hint={"spec": h["spec"], "description": h.get("description", "")})
        return {"attack": "spec_review", "severity": "medium", "survived": obj["verdict"] == "clean",
                "evidence": {"concerns": obj["concerns"]}}
    return run

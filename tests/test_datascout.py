"""Data Scout (prompt-only flow): question parsing, column mapping and preparation. Offline: no network is used."""
import numpy as np
import pandas as pd
import pytest

from skeptic.backend import datascout as DS
from skeptic.backend.adapters import tabular as TB


def test_parse_cause_effect_and_comparisons():
    p = DS.parse_question("Does smoking increase heart disease?")
    assert "smoking" in p["cause"] and "tobacco" in p["cause"] and p["effect"] == ["heart", "disease"]
    assert "does" not in p["all"]
    p = DS.parse_question("Do men earn more than women?")
    assert {"men", "women", "sex"} <= set(p["cause"]) and "income" in p["effect"]
    p = DS.parse_question("Do older passengers have lower survival on the titanic?")
    assert "age" in p["cause"] and "survival" in p["effect"]
    assert DS.parse_question("What predicts wine quality?").get("open")


def test_find_dataset_refuses_open_question_and_offline(monkeypatch):
    with pytest.raises(RuntimeError, match="specific cause"):
        DS.find_dataset("What predicts wine quality?", lambda *a, **k: None)
    monkeypatch.setenv("SKEPTIC_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="turned off"):
        DS.find_dataset("Does smoking increase heart disease?", lambda *a, **k: None)


def _table(n=400, seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame({"PassengerId": np.arange(n), "Age": r.integers(1, 80, n).astype(float), "Sex": r.choice(["male", "female"], n),
                         "DiabetesPedigreeFunction": r.random(n), "T2Diabetic": r.integers(0, 2, n), "education": r.choice(["HS", "BA", "MA"], n),
                         "education-num": r.integers(1, 16, n)})


def test_map_columns_prefers_yes_no_outcome_and_ordered_cause_and_skips_ids():
    df = _table()
    m = DS.map_columns(df, DS.parse_question("Does age increase the risk of diabetes?"))
    assert m["treat"] == "Age" and m["outcome"] == "T2Diabetic"          # a yes/no diabetes column beats a pedigree score
    df["income"] = np.random.default_rng(1).random(len(df))
    m = DS.map_columns(df, DS.parse_question("Does education increase income?"))
    assert m["treat"] == "education-num"                                 # ordered beats a list of categories
    m = DS.map_columns(df.assign(Survived=np.random.default_rng(2).integers(0, 2, len(df))), DS.parse_question("Do older passengers have lower survival?"))
    assert m["treat"] == "Age"                                           # never the PassengerId column


def test_map_columns_says_which_side_is_missing():
    df = _table()
    with pytest.raises(ValueError, match="cause"):
        DS.map_columns(df.assign(heart_disease=1 - df["T2Diabetic"]), DS.parse_question("Does smoking increase heart disease?"))
    with pytest.raises(ValueError, match="effect"):
        DS.map_columns(df.assign(smoker=df["T2Diabetic"]), DS.parse_question("Does smoking increase heart disease?"))


def test_prepare_names_columns_by_rule_and_shuffles():
    df = _table().sort_values("T2Diabetic").reset_index(drop=True)      # sorted like many public tables
    d, info = DS.prepare(df, "Age", "T2Diabetic", seed=0)
    assert info["treat_col"].startswith("Age above ") and info["outcome_col"] == "T2Diabetic = 1"
    assert set(d[info["treat_col"]].unique()) == {0.0, 1.0}
    tail = d[info["outcome_col"]].iloc[-100:]
    assert 0.2 < tail.mean() < 0.8                                       # the held-out tail is no longer all one class
    d, info = DS.prepare(df, "Sex", "DiabetesPedigreeFunction", seed=0)
    assert info["outcome_col"] == "DiabetesPedigreeFunction" and "Sex = " in info["treat_col"]


def test_arff_reader():
    raw = b"""% comment
@relation t
@attribute 'age' numeric
@attribute sex {male,female}
@attribute "income class" {'<=50K','>50K'}
@data
39,male,'<=50K'
50,female,'>50K'
?,male,'<=50K'
"""
    df = DS._read_arff(raw)
    assert list(df.columns) == ["age", "sex", "income class"] and len(df) == 3 and df["age"].isna().sum() == 1
    assert df["income class"].iloc[1] == ">50K"


def test_tabular_bins_survive_heavily_repeated_values():
    s = pd.Series([0] * 200 + list(range(1, 9)) * 20, name="siblings")     # two of the three tercile edges are 0: they merge
    b = TB._bin(s)                                                          # used to raise "Bin labels must be one fewer..."
    assert b.nunique() == 2 and all(v.startswith("siblings:Q") for v in b.unique())


def test_effect_illusions_for_non_money_runs():
    """Tabular runs price the illusion in outcome units: refused findings whose effect shrinks on held-back data."""
    from skeptic.backend import runner
    ev = []
    runner.run_scenario("health", {"question": "Do supplement users really have better health scores?"}, emit=ev.append)
    il = [e for e in ev if e["type"] == "illusion"]
    assert il and all(e["kind"] == "effect" for e in il)
    for e in il:
        assert e["direction"] * (e["in_total"] - e["real_total"]) > 0          # fresh data disappoints: it is an illusion
        assert len(e["in_curve"]) >= 2 and abs(e["in_curve"][-1] - e["in_total"]) < 1e-9
        assert e["n_hold"] > 0 and e["binary"] in (True, False)

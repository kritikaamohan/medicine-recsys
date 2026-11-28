"""Core logic for MediGuide. No Streamlit imports, so it can be unit-tested."""
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

DISCLAIMER = ("MediGuide gives general information only. It is not a diagnosis or medical "
              "advice. Always check with a doctor or pharmacist before taking any medicine.")
EMERGENCY = {"Heart attack", "Paralysis (brain hemorrhage)"}
RED_FLAG_NAMES = ["chest_pain", "breathlessness", "coma", "altered_sensorium",
                  "weakness_of_one_body_side", "slurred_speech"]
DISPLAY = {"(vertigo) Paroymsal Positional Vertigo": "Vertigo (positional)"}


def norm(x):
    """Collapse extra spaces and fix a known label typo so files match."""
    x = " ".join(str(x).split())
    return {"Peptic ulcer diseae": "Peptic ulcer disease"}.get(x, x)


def nice(s):
    return s.replace("_", " ")


@lru_cache(maxsize=1)
def load():
    bundle = joblib.load(BASE / "models" / "symptom_model.joblib")
    out = BASE / "outputs"
    rank = pd.read_csv(out / "drug_rankings.csv")
    desc = pd.read_csv(out / "description.csv")
    prec = pd.read_csv(out / "precautions_df.csv")
    prof = pd.read_csv(out / "disease_profiles.csv")
    stress = pd.read_csv(out / "stress_test.csv")
    desc["Disease"] = desc["Disease"].map(norm)
    prec["Disease"] = prec["Disease"].map(norm)
    symptoms = list(bundle["symptoms"])
    return {
        "model": bundle["model"], "symptoms": symptoms, "rank": rank, "desc": desc,
        "prec": prec, "stress": stress,
        "profiles": {norm(r.disease): r.symptoms.split("|") for r in prof.itertuples()},
        "red_flags": [s for s in RED_FLAG_NAMES if s in symptoms],
    }


def match_text(share):
    if share >= 0.7:
        return "Strong match"
    if share >= 0.4:
        return "Possible match"
    return "Weak match"


def match_icon(share):
    return "🟢" if share >= 0.7 else "🟡" if share >= 0.4 else "⚪"


def analyse(chosen, age, allergies_text=""):
    """Run the model and collect everything the results page and reports need."""
    d = load()
    symptoms, model = d["symptoms"], d["model"]
    x = np.zeros((1, len(symptoms)))
    for s in chosen:
        x[0, symptoms.index(s)] = 1
    proba = model.predict_proba(pd.DataFrame(x, columns=symptoms))[0]
    top = np.argsort(proba)[::-1][:3]
    allergies = [a.strip().lower() for a in allergies_text.split(",") if a.strip()]

    results = []
    for i in top:
        disease = norm(model.classes_[i])
        typical = d["profiles"].get(disease, [])
        matched = [s for s in typical if s in chosen]
        others = [s for s in typical if s not in chosen]
        share = len(matched) / len(typical) if typical else 0.0
        desc = d["desc"][d["desc"].Disease == disease].Description
        prec = d["prec"][d["prec"].Disease == disease]
        r = d["rank"][d["rank"].disease == disease].sort_values("rank")
        removed = []
        if allergies:
            hit = r.drugName.str.lower().apply(
                lambda n: any(a in n for a in allergies)).astype(bool)
            removed = r[hit].drugName.tolist()
            r = r[~hit]
        results.append({
            "disease": disease, "label": DISPLAY.get(disease, disease),
            "share": float(share), "matched": matched, "others": others,
            "description": desc.iloc[0] if len(desc) else "",
            "precautions": [str(v) for v in prec.iloc[0, 2:6].dropna()] if len(prec) else [],
            "emergency": disease in EMERGENCY,
            "drugs": [{"drugName": str(row.drugName), "n_reviews": int(row.n_reviews),
                       "mean_rating": round(float(row.mean_rating), 1)}
                      for row in r.head(5).itertuples()],
            "removed": removed,
        })
    return {"symptoms": list(chosen), "age": int(age), "results": results,
            "red_flags": [s for s in chosen if s in d["red_flags"]],
            "time": datetime.now().strftime("%d %b %Y, %H:%M")}


def summary_text(res):
    """Plain-text fallback summary (used if PDF generation is unavailable)."""
    lines = ["MediGuide summary (general information only, not medical advice)",
             f"Date: {res['time']}", f"Age: {res['age']}",
             "Symptoms: " + ", ".join(nice(s) for s in res["symptoms"]), ""]
    for n, r in enumerate(res["results"], 1):
        total = len(r["matched"]) + len(r["others"])
        lines.append(f"{n}. {r['label']} ({match_text(r['share'])}, "
                     f"{len(r['matched'])} of {total} typical symptoms)")
        if r["precautions"]:
            lines.append("   Self-care: " + ", ".join(r["precautions"]))
        if r["drugs"]:
            lines.append("   Medicines other patients rated well: " +
                         ", ".join(x["drugName"] for x in r["drugs"][:3]))
        lines.append("")
    lines.append("Please discuss this with a doctor or pharmacist.")
    return "\n".join(lines)

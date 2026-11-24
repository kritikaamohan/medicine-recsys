from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MediGuide", page_icon="💊", layout="centered")

DISCLAIMER = ("MediGuide gives general information only. It is not a diagnosis or medical "
              "advice. Always check with a doctor or pharmacist before taking any medicine.")
EMERGENCY = {"Heart attack", "Paralysis (brain hemorrhage)"}
RED_FLAGS = ["chest_pain", "breathlessness", "coma", "altered_sensorium",
             "weakness_of_one_body_side", "slurred_speech"]


def norm(x):
    """Collapse extra spaces and fix a known label typo so files match."""
    x = " ".join(str(x).split())
    return {"Peptic ulcer diseae": "Peptic ulcer disease"}.get(x, x)


DISPLAY = {"(vertigo) Paroymsal Positional Vertigo": "Vertigo (positional)"}


def nice(s):
    return s.replace("_", " ")


@st.cache_resource
def load():
    bundle = joblib.load("models/symptom_model.joblib")
    rank = pd.read_csv("outputs/drug_rankings.csv")
    desc = pd.read_csv("outputs/description.csv")
    prec = pd.read_csv("outputs/precautions_df.csv")
    prof = pd.read_csv("outputs/disease_profiles.csv")
    stress = pd.read_csv("outputs/stress_test.csv")
    desc["Disease"] = desc["Disease"].map(norm)
    prec["Disease"] = prec["Disease"].map(norm)
    profiles = {norm(r.disease): r.symptoms.split("|") for r in prof.itertuples()}
    return bundle["model"], bundle["symptoms"], rank, desc, prec, profiles, stress


model, SYMPTOMS, rank, desc, prec, profiles, stress = load()
RED_FLAGS = [s for s in RED_FLAGS if s in SYMPTOMS]


def analyse(chosen, age, allergies_text):
    """Run the model and collect everything the results page needs."""
    x = np.zeros((1, len(SYMPTOMS)))
    for s in chosen:
        x[0, SYMPTOMS.index(s)] = 1
    proba = model.predict_proba(x)[0]
    top = np.argsort(proba)[::-1][:3]
    allergies = [a.strip().lower() for a in allergies_text.split(",") if a.strip()]
    results = []
    for i in top:
        disease = norm(model.classes_[i])
        typical = profiles.get(disease, [])
        matched = [s for s in typical if s in chosen]
        others = [s for s in typical if s not in chosen]
        share = len(matched) / len(typical) if typical else 0.0
        d = desc[desc.Disease == disease].Description
        p = prec[prec.Disease == disease]
        r = rank[rank.disease == disease].sort_values("rank")
        removed = []
        if allergies:
            hit = r.drugName.str.lower().apply(
                lambda n: any(a in n for a in allergies)).astype(bool)
            removed = r[hit].drugName.tolist()
            r = r[~hit]
        results.append({
            "disease": disease, "label": DISPLAY.get(disease, disease), "share": share, "matched": matched, "others": others,
            "description": d.iloc[0] if len(d) else "",
            "precautions": [str(v) for v in p.iloc[0, 2:6].dropna()] if len(p) else [],
            "emergency": disease in EMERGENCY,
            "drugs": r.head(5)[["drugName", "n_reviews", "mean_rating"]].round(1),
            "removed": removed,
        })
    return {"symptoms": chosen, "age": age, "results": results,
            "red_flags": [s for s in chosen if s in RED_FLAGS],
            "time": datetime.now().strftime("%d %b %Y, %H:%M")}


def match_label(share):
    if share >= 0.7:
        return "🟢 Strong match"
    if share >= 0.4:
        return "🟡 Possible match"
    return "⚪ Weak match"


def summary_text(res):
    lines = ["MediGuide summary (general information only, not medical advice)",
             f"Date: {res['time']}", f"Age: {res['age']}",
             "Symptoms: " + ", ".join(nice(s) for s in res["symptoms"]), ""]
    for n, r in enumerate(res["results"], 1):
        lines.append(f"{n}. {r['label']} ({match_label(r['share'])[2:]}, "
                     f"{len(r['matched'])} of {len(r['matched']) + len(r['others'])} typical symptoms)")
        if r["precautions"]:
            lines.append("   Self-care: " + ", ".join(r["precautions"]))
        if len(r["drugs"]):
            lines.append("   Medicines other patients rated well: " +
                         ", ".join(r["drugs"].drugName.head(3)))
        lines.append("")
    lines.append("Please discuss this with a doctor or pharmacist.")
    return "\n".join(lines)


# ------------------------------------------------------------------ header
st.title("💊 MediGuide")
st.caption("Tell us how you feel and get general guidance, simple self-care tips, and "
           "medicines other patients rated well.")
st.info(DISCLAIMER)

tab_check, tab_about = st.tabs(["🩺 Check symptoms", "ℹ️ About"])

# ------------------------------------------------------------------ check tab
with tab_check:
    st.subheader("1. About you")
    c1, c2 = st.columns([1, 2])
    age = c1.number_input("Age", min_value=0, max_value=120, value=30)
    allergies_text = c2.text_input("Medicine allergies (optional)",
                                   placeholder="e.g. aspirin, penicillin")

    st.subheader("2. Your symptoms")
    chosen = st.multiselect("Type to search and pick everything you are feeling",
                            SYMPTOMS, format_func=nice)
    if chosen:
        st.caption(f"{len(chosen)} selected. The more you add, the more reliable the guidance.")

    if st.button("Get guidance", type="primary", width="stretch"):
        if not chosen:
            st.error("Please pick at least one symptom.")
        else:
            res = analyse(chosen, age, allergies_text)
            st.session_state["last"] = res
            st.session_state.setdefault("history", []).append(
                {"time": res["time"], "symptoms": ", ".join(nice(s) for s in chosen),
                 "top": res["results"][0]["label"]})

    res = st.session_state.get("last")
    if res:
        st.divider()
        st.subheader("3. Your guidance")

        if res["red_flags"] or any(r["emergency"] for r in res["results"]):
            st.error("🚨 Some of what you described can be serious (for example "
                     + ", ".join(nice(s) for s in res["red_flags"][:3] or ["your symptoms"])
                     + "). If this is happening now or getting worse, seek urgent medical "
                       "care or call your local emergency number.")
        if len(res["symptoms"]) < 3:
            st.warning("You gave only a few symptoms, so these results are less reliable. "
                       "Adding more symptoms helps.")

        for n, r in enumerate(res["results"], 1):
            with st.container(border=True):
                st.markdown(f"### {n}. {r['label']}")
                st.markdown(f"**{match_label(r['share'])}**: you matched "
                            f"{len(r['matched'])} of {len(r['matched']) + len(r['others'])} "
                            "typical symptoms")
                st.progress(r["share"])
                if r["description"]:
                    st.write(r["description"])
                if r["matched"]:
                    st.write("**You mentioned:** " + ", ".join(nice(s) for s in r["matched"]))
                if r["others"]:
                    st.write("**Also common with this condition:** "
                             + ", ".join(nice(s) for s in r["others"][:6])
                             + ". Tell a doctor if you have these.")
                if r["emergency"]:
                    st.error("This can be an emergency. Please get urgent medical care.")
                    continue
                if r["precautions"]:
                    st.write("**What you can do:**")
                    for tip in r["precautions"]:
                        st.write(f"- {tip.capitalize()}")
                with st.expander("💊 Medicines other patients rated well"):
                    if r["removed"]:
                        st.info("Left out because of your allergy: " + ", ".join(r["removed"]))
                    if len(r["drugs"]):
                        t = r["drugs"].rename(columns={"drugName": "Medicine",
                                                       "n_reviews": "Reviews",
                                                       "mean_rating": "Avg rating (out of 10)"})
                        st.dataframe(t.reset_index(drop=True), hide_index=True,
                                     width="stretch")
                        st.caption("Based on patient reviews, not medical evidence. Ask a "
                                   "pharmacist or doctor before taking any medicine.")
                        if res["age"] < 18 or res["age"] >= 65:
                            st.caption("Doses differ for children and older adults. "
                                       "Please check with a doctor.")
                    else:
                        st.write("We don't have patient reviews for this condition. "
                                 "Please speak to a doctor about treatment.")

        st.info("**See a doctor** if your symptoms are severe, getting worse, or last more "
                "than a few days.")
        st.download_button("⬇️ Download summary for your doctor", summary_text(res),
                           file_name="mediguide_summary.txt", width="stretch")

    hist = st.session_state.get("history", [])
    if len(hist) > 1:
        with st.expander("Your earlier checks (this visit only)"):
            st.dataframe(pd.DataFrame(hist[:-1][::-1]).rename(
                columns={"time": "When", "symptoms": "Symptoms", "top": "Top result"}),
                hide_index=True, width="stretch")

# ------------------------------------------------------------------ about tab
with tab_about:
    st.subheader("How MediGuide works")
    st.write("1. You pick your symptoms. A machine-learning model compares them with the "
             "symptom patterns of 41 conditions.\n"
             "2. It shows the 3 closest matches and how many of each condition's typical "
             "symptoms you reported.\n"
             "3. For conditions with enough patient reviews, it lists medicines other "
             "patients rated well, and leaves out any you are allergic to.")

    st.subheader("How reliable is it?")
    st.write("MediGuide works best when you enter **three or more symptoms**. In our testing "
             "the right condition was usually among the top three even when only half of a "
             "condition's typical symptoms were entered, but it became less reliable with "
             "fewer. It can be wrong, and it cannot examine you.")

    st.subheader("What it cannot do")
    st.write("- It does not diagnose, and it does not replace a doctor.\n"
             "- Medicine suggestions are only available for 18 of the 41 conditions.\n"
             "- Ratings are patient opinions, and allergy checks only match medicine names.")

    st.subheader("Data used")
    st.write("A public symptom-to-condition dataset (Kaggle) and patient drug reviews from "
             "Drugs.com (UCI / Kaggle).")

    with st.expander("For reviewers: model evaluation"):
        st.write("The raw training file has 4,920 rows but only 304 are unique, so a random "
                 "split leaks (99.6% of test rows had an identical copy in training). "
                 "Results below use the 304 unique rows with 5-fold cross-validation, "
                 "hiding symptoms at random.")
        t = stress.copy()
        t["Symptoms given"] = (t["symptoms_kept"] * 100).round().astype(int).astype(str) + "%"
        t["Top-1 accuracy"] = (t["top1"] * 100).round(1).astype(str) + "%"
        t["Top-3 accuracy"] = (t["top3"] * 100).round(1).astype(str) + "%"
        st.dataframe(t[["Symptoms given", "Top-1 accuracy", "Top-3 accuracy"]],
                     hide_index=True, width="stretch")
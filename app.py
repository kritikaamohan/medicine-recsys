import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Medicine Recommender", page_icon="💊")

DISCLAIMER = ("Educational decision-support only, not medical advice. "
              "Consult a doctor or pharmacist before taking any medicine.")
EMERGENCY = {"Heart attack", "Paralysis (brain hemorrhage)"}


def norm(x):
    """Collapse extra spaces and fix a known label typo so files match."""
    x = " ".join(str(x).split())
    return {"Peptic ulcer diseae": "Peptic ulcer disease"}.get(x, x)


@st.cache_resource
def load():
    bundle = joblib.load("models/symptom_model.joblib")
    rank = pd.read_csv("outputs/drug_rankings.csv")
    desc = pd.read_csv("outputs/description.csv")
    prec = pd.read_csv("outputs/precautions_df.csv")
    desc["Disease"] = desc["Disease"].map(norm)
    prec["Disease"] = prec["Disease"].map(norm)
    return bundle["model"], bundle["symptoms"], rank, desc, prec


model, SYMPTOMS, rank, desc, prec = load()

st.title("💊 Personalized Medicine Recommender")
st.warning(DISCLAIMER)

chosen = st.multiselect("Select your symptoms", SYMPTOMS,
                        format_func=lambda s: s.replace("_", " "))
age = st.number_input("Age", min_value=0, max_value=120, value=30)
allergies_text = st.text_input("Drug allergies (comma separated, optional)")

if st.button("Get recommendations"):
    if not chosen:
        st.error("Please select at least one symptom.")
    else:
        x = np.zeros((1, len(SYMPTOMS)))
        for s in chosen:
            x[0, SYMPTOMS.index(s)] = 1
        proba = model.predict_proba(x)[0]
        top = np.argsort(proba)[::-1][:3]
        allergies = [a.strip().lower() for a in allergies_text.split(",") if a.strip()]

        for place, i in enumerate(top, start=1):
            disease = norm(model.classes_[i])
            st.subheader(f"{place}. {disease}")
            d = desc[desc.Disease == disease].Description
            if len(d):
                st.write(d.iloc[0])
            if disease in EMERGENCY:
                st.error("Possible emergency. Seek urgent medical care. No medicines suggested.")
                continue
            p = prec[prec.Disease == disease]
            if len(p):
                st.write("**Precautions:** " + ", ".join(str(v) for v in p.iloc[0, 2:6].dropna()))
            r = rank[rank.disease == disease].sort_values("rank")
            if allergies:
                hit = r.drugName.str.lower().apply(
                    lambda n: any(a in n for a in allergies)).astype(bool)
                if hit.any():
                    st.info("Excluded for allergy: " + ", ".join(r[hit].drugName))
                r = r[~hit]
            if len(r):
                st.write("**Top-rated medicines (from patient reviews):**")
                st.dataframe(r.head(5)[["drugName", "n_reviews", "mean_rating", "score"]]
                             .round(2).reset_index(drop=True))
            else:
                st.write("No review data for this condition. Please see a doctor.")
            if age < 18 or age >= 65:
                st.caption("Dosing differs for children and older adults. Check with a doctor.")
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Medicine Recommender", page_icon="💊", layout="wide")

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
    stress = pd.read_csv("outputs/stress_test.csv")
    desc["Disease"] = desc["Disease"].map(norm)
    prec["Disease"] = prec["Disease"].map(norm)
    return bundle["model"], bundle["symptoms"], rank, desc, prec, stress


model, SYMPTOMS, rank, desc, prec, stress = load()

st.title("Personalized Medicine Recommender")
st.warning(DISCLAIMER)

tab_rec, tab_ins = st.tabs(["🩺 Recommend", "📊 Insights"])

# ---------------- Recommend tab ----------------
with tab_rec:
    left, right = st.columns([2, 1])
    with left:
        chosen = st.multiselect("Select your symptoms", SYMPTOMS,
                                format_func=lambda s: s.replace("_", " "))
    with right:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        allergies_text = st.text_input("Drug allergies (comma separated, optional)")

    if st.button("Get recommendations", type="primary"):
        if not chosen:
            st.error("Please select at least one symptom.")
        else:
            if len(chosen) < 3:
                st.warning("You selected fewer than 3 symptoms. In our stress test, accuracy "
                           "dropped noticeably when only a few symptoms were given, so treat "
                           "these results as low confidence and add more symptoms if you can.")
            x = np.zeros((1, len(SYMPTOMS)))
            for s in chosen:
                x[0, SYMPTOMS.index(s)] = 1
            proba = model.predict_proba(x)[0]
            top = np.argsort(proba)[::-1][:3]
            allergies = [a.strip().lower() for a in allergies_text.split(",") if a.strip()]

            cols = st.columns(3)
            for place, i in enumerate(top):
                disease = norm(model.classes_[i])
                with cols[place]:
                    with st.container(border=True):
                        st.subheader(f"{place + 1}. {disease}")
                        d = desc[desc.Disease == disease].Description
                        if len(d):
                            st.write(d.iloc[0])
                        if disease in EMERGENCY:
                            st.error("Possible emergency. Seek urgent medical care. "
                                     "No medicines suggested.")
                            continue
                        p = prec[prec.Disease == disease]
                        if len(p):
                            st.write("**Precautions:** " +
                                     ", ".join(str(v) for v in p.iloc[0, 2:6].dropna()))
                        r = rank[rank.disease == disease].sort_values("rank")
                        if allergies:
                            hit = r.drugName.str.lower().apply(
                                lambda n: any(a in n for a in allergies)).astype(bool)
                            if hit.any():
                                st.info("Excluded for allergy: " + ", ".join(r[hit].drugName))
                            r = r[~hit]
                        if len(r):
                            st.write("**Top-rated medicines (patient reviews):**")
                            st.dataframe(r.head(5)[["drugName", "n_reviews", "mean_rating"]]
                                         .round(2).reset_index(drop=True),
                                         use_container_width=True)
                        else:
                            st.write("No review data for this condition. Please see a doctor.")
                        if age < 18 or age >= 65:
                            st.caption("Dosing differs for children and older adults. "
                                       "Check with a doctor.")

# ---------------- Insights tab ----------------
with tab_ins:
    st.subheader("Data quality")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows in Training.csv", "4,920")
    c2.metric("Unique rows", "304")
    c3.metric("Test rows duplicated in training", "99.6%")
    st.caption("A random split of the raw data leaks: almost every test row has an identical "
               "twin in the training set, so the raw 100% accuracy only shows memorization.")

    st.subheader("Accuracy when the user gives fewer symptoms")
    s = stress.assign(pct_given=(stress.symptoms_kept * 100).astype(int))
    s = s.set_index("pct_given")[["top1", "top3"]].sort_index()
    st.line_chart(s)
    st.caption("5-fold cross-validation on the 304 unique rows; symptoms hidden at random. "
               "Top-3 stays high because the app shows three candidate conditions.")

    st.subheader("Top-ranked medicine score by condition")
    best = rank[rank["rank"] == 1].set_index("disease")["score"].sort_values()
    st.bar_chart(best)

    st.subheader("Explore drug rankings")
    pick = st.selectbox("Condition", sorted(rank.disease.unique()))
    st.dataframe(rank[rank.disease == pick].head(10)
                 [["drugName", "n_reviews", "mean_rating", "score"]]
                 .round(2).reset_index(drop=True), use_container_width=True)
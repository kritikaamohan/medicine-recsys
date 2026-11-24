# 💊 MediGuide: Personalized Symptom & Medicine Guide

A patient-facing web app that takes a person's symptoms, age, and medicine allergies, then shows the most likely matching conditions, simple self-care tips, and medicines that other patients rated well.

**Live app:** https://kritikamohan-medicine-recsys.streamlit.app/

> ⚠️ **Disclaimer:** MediGuide is an educational project. It gives general information only and is not a diagnosis or medical advice. Always check with a doctor or pharmacist before taking any medicine.

## Features

- **Symptom search:** pick from 131 symptoms by typing, with age and allergy inputs.
- **Top 3 matching conditions** out of 41, from a machine-learning model.
- **Match strength:** shows how many of a condition's typical symptoms the user reported (Strong, Possible, or Weak match), plus the other common symptoms to mention to a doctor.
- **Safety checks:** an urgent-care banner for red-flag symptoms (such as chest pain or breathlessness) and for emergency conditions, and a low-reliability warning when fewer than 3 symptoms are entered.
- **Medicine guidance:** medicines ranked from patient reviews, with allergy filtering and an age caution for children and adults over 65.
- **Self-care tips** and condition descriptions.
- **Downloadable summary** to share with a doctor.
- **About tab** with how it works, its limitations, and a model-evaluation section for reviewers.

## How it works

1. **Symptoms to condition:** a Random Forest classifier takes the selected symptoms (131 binary features) and returns the top 3 likely conditions.
2. **Match strength:** each condition's symptom profile is compared with the user's symptoms, so the result is easy to explain.
3. **Condition to medicines:** drugs for each condition are ranked using Drugs.com patient reviews. The score combines the rating, the number of reviews, and how useful readers found each review. A Bayesian-style adjustment stops a drug with only a few reviews from outranking one with many.
4. **Personalization and safety:** allergy matches are removed from the medicine list, and emergency or red-flag cases show an urgent-care message.

## Project structure

```
medicine-recsys/
├── app.py                          # Streamlit app
├── requirements.txt                # pinned dependencies
├── .streamlit/config.toml          # theme
├── models/
│   └── symptom_model.joblib        # trained classifier + symptom list
├── outputs/
│   ├── drug_rankings.csv           # ranked drugs per condition
│   ├── description.csv             # condition descriptions
│   ├── precautions_df.csv          # self-care tips
│   ├── disease_profiles.csv        # typical symptoms per condition
│   ├── stress_test.csv             # accuracy with fewer symptoms
│   └── stress_test.png             # stress-test chart
└── notebooks/
    └── 01_explore_symptoms.ipynb   # cleaning, evaluation, drug ranking
```

## Datasets

- **Symptom-condition data** (`Training.csv`, `description.csv`, `precautions_df.csv`): the public Kaggle disease-symptom dataset with 4,920 rows, 131 symptoms, and 41 conditions.
- **Drug reviews** (`drugsComTest_raw.csv`): the UCI / Kaggle Drugs.com review dataset (test split, 53,766 reviews).

The raw data is not included in this repository. Check each dataset's license on its source page.

## Key findings

- **Heavy duplication:** `Training.csv` has 4,920 rows but only **304 unique** ones (4,616 exact duplicates, 120 copies per condition).
- **Data leakage:** in a random 70/30 split of the raw data, **1,470 of 1,476** test rows (99.6%) had an identical copy in the training set. The raw 100% accuracy therefore only shows memorization.
- **Honest evaluation:** after removing duplicates and using stratified 5-fold cross-validation, Random Forest scored 1.0 top-1 and 1.0 top-3 on complete symptom lists, because each condition has its own small symptom set.
- **Stress test (fewer symptoms given):** accuracy falls when users give fewer symptoms.

  | Symptoms given | Top-1 | Top-3 |
  |---|---|---|
  | 100% | 100% | 100% |
  | 75% | 98.4% | 100% |
  | 50% | 85.9% | 98.4% |
  | 30% | 62.5% | 85.8% |

  This is why the app shows three candidate conditions and warns users who enter few symptoms. The results come from one random run on 304 rows, so expect small variation.
- **Data bug fixed:** the file has two columns named `fluid_overload`. One is all zeros and the other holds the real data (only in Alcoholic hepatitis), so the empty one was dropped and the real one kept.
- **Unusable file:** `medications.csv` appears misaligned (for example, Pneumonia lists laxatives and Tuberculosis lists cough suppressants), so it was not used.
- **Drug ranking:** 7,701 reviews were mapped to 18 conditions.

## Limitations

- The symptom dataset looks synthetic and nearly perfectly separable, so the accuracy does not reflect real clinical performance.
- Medicine suggestions exist for only 18 of the 41 conditions. For the others, the app advises seeing a doctor.
- Model probabilities are not calibrated, so the app shows match strength instead of percentages.
- Drug ratings are patient opinions, not clinical evidence. Brand and generic names are listed separately, and the allergy filter only matches medicine names, not ingredients.
- The red-flag symptom list is a simple safety net and has not been reviewed by a medical professional.

## Run locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Tech stack

Python 3.11, pandas, NumPy, scikit-learn, joblib, Streamlit. Deployed on Streamlit Community Cloud.
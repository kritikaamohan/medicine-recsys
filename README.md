# 💊 MediGuide: Personalized Symptom & Medicine Guide

A patient-facing web app that takes a person's symptoms, age, and medicine allergies, then shows the most likely matching conditions, simple self-care tips, and medicines that other patients rated well. Signing in is optional and lets users save their checks and download reports.

> ⚠️ **Disclaimer:** MediGuide is an educational project. It gives general information only and is not a diagnosis or medical advice. Always check with a doctor or pharmacist before taking any medicine. Please do not enter real personal or medical-record details.

## Project at a glance

| Item | Detail |
|---|---|
| Project | Personalized Healthcare & Medicine Recommendation System |
| Domain | Machine Learning / Data Science |
| ML task | Multi-class classification (41 conditions) + review-based medicine ranking |
| Model | Random Forest (200 trees), evaluated with stratified 5-fold cross-validation |
| Data | Public Kaggle symptom-condition dataset; Drugs.com patient reviews (UCI / Kaggle) |
| App | Streamlit (optional sign-in, saved history, PDF and CSV downloads) |
| Storage | SQLite (accounts and history), salted PBKDF2 password hashes |
| Deployment | Streamlit Community Cloud |
| Live app | <paste your .streamlit.app link here> |
| Author | <your name> |

## Features

- **Symptom search:** pick from 131 symptoms by typing, with age and allergy inputs.
- **Top 3 matching conditions** out of 41.
- **Match strength:** shows how many of a condition's typical symptoms the user reported (Strong, Possible, or Weak match), plus other common symptoms to mention to a doctor.
- **Safety checks:** an urgent-care banner for red-flag symptoms (such as chest pain or breathlessness) and emergency conditions, and a low-reliability warning when fewer than 3 symptoms are entered.
- **Medicine guidance:** medicines ranked from patient reviews, with allergy filtering and an age caution for children and adults over 65.
- **Optional accounts:** register and sign in to save checks. Guests can use everything except saved history.
- **My history:** view saved checks, export them as **CSV**, reopen a past check, and delete all saved data.
- **PDF report** of any check, to share with a doctor.
- **About tab** with how it works, limitations, a data-privacy note, and a model-evaluation section for reviewers.
- **Automated tests** for the model pipeline, accounts, history, and PDF generation.

## Screenshots

Screenshots live in `docs/screenshots/`.

| Screen | File |
|---|---|
| Symptom entry | ![Symptom entry](docs/screenshots/01_symptom_entry.png) |
| Results with match strength | ![Results](docs/screenshots/02_results.png) |
| Urgent-care banner for a red-flag symptom | ![Urgent banner](docs/screenshots/03_urgent_banner.png) |
| Allergy exclusion in the medicine list | ![Allergy filter](docs/screenshots/04_allergy_filter.png) |
| My history (sign-in, CSV export) | ![History](docs/screenshots/05_history.png) |
| PDF report | ![PDF report](docs/screenshots/06_pdf_report.png) |

## How it works

```
                      User (guest or signed in)
                                  |
                                  v
                        Streamlit app (app.py)
                                  |
        +-------------------------+--------------------------+
        |                         |                          |
        v                         v                          v
   Account & history        Analysis engine              PDF report
     (db.py, SQLite)         (engine.py)                 (report.py)
                                  |
              +-------------------+-------------------+
              |                   |                   |
              v                   v                   v
      Random Forest model   Condition profiles    Drug rankings
     (symptoms -> top 3)    (match strength)     (patient reviews)
```

1. **Symptoms to condition:** a Random Forest classifier takes the selected symptoms (131 binary features) and returns the top 3 likely conditions.
2. **Match strength:** each condition's symptom profile is compared with the user's symptoms, so the result is easy to explain.
3. **Condition to medicines:** drugs for each condition are ranked from Drugs.com patient reviews. The score combines the rating, the number of reviews, and how useful readers found each review. A Bayesian-style adjustment stops a drug with only a few reviews from outranking one with many.
4. **Personalization and safety:** allergy matches are removed from the medicine list, and emergency or red-flag cases show an urgent-care message.
5. **Accounts:** passwords are stored only as a random salt plus a PBKDF2-HMAC-SHA256 hash (200,000 iterations). Each user can only see their own history.

## Project structure

```
medicine-recsys/
├── app.py                          # Streamlit user interface
├── engine.py                       # model, match strength, drug filtering (no Streamlit)
├── db.py                           # accounts and saved history (SQLite)
├── report.py                       # PDF report builder
├── requirements.txt                # pinned dependencies
├── models/
│   └── symptom_model.joblib        # trained classifier + symptom list
├── outputs/
│   ├── drug_rankings.csv           # ranked drugs per condition
│   ├── description.csv             # condition descriptions
│   ├── precautions_df.csv          # self-care tips
│   ├── disease_profiles.csv        # typical symptoms per condition
│   └── stress_test.csv             # accuracy with fewer symptoms
├── notebooks/
│   └── 01_explore_symptoms.ipynb   # cleaning, evaluation, drug ranking
├── tests/
│   └── test_mediguide.py           # unit tests
└── docs/
    ├── data_dictionary.csv         # meaning of every output column
    └── screenshots/
```

## Datasets

- **Symptom-condition data** (`Training.csv`, `description.csv`, `precautions_df.csv`): the public Kaggle disease-symptom dataset with 4,920 rows, 131 symptoms, and 41 conditions.
- **Drug reviews** (`drugsComTest_raw.csv`): the UCI / Kaggle Drugs.com review dataset (test split, 53,766 reviews).

The raw data is not included in this repository. Check each dataset's license on its source page.

## Key findings

- **Heavy duplication:** `Training.csv` has 4,920 rows but only **304 unique** ones (4,616 exact duplicates, 120 copies per condition).
- **Data leakage:** in a random 70/30 split of the raw data, **1,470 of 1,476** test rows (99.6%) had an identical copy in the training set, so the raw 100% accuracy only shows memorization.
- **Honest evaluation:** after removing duplicates and using stratified 5-fold cross-validation, Random Forest scored 1.0 top-1 and 1.0 top-3 on complete symptom lists, because each condition has its own small symptom set.
- **Stress test (fewer symptoms given):**

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
- Accounts are stored in a local SQLite file. On Streamlit Community Cloud this file can reset when the app restarts, so saved history is demo-level. A hosted PostgreSQL database (for example Supabase) would be the production upgrade, along with login rate limiting and password reset.

## Run locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Run the tests

```bash
python -m unittest discover -s tests -v
```

The tests check the model artifacts, predictions, allergy filtering, drug-ranking sanity, account creation and login, per-user history isolation, password hashing, and PDF generation.

## Tech stack

Python 3.11, pandas, NumPy, scikit-learn, joblib, Streamlit, ReportLab, SQLite. Deployed on Streamlit Community Cloud.

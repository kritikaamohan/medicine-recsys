# Personalized Healthcare & Medicine Recommendation System

A machine-learning app that takes a user's symptoms and basic profile, predicts the most likely conditions, and suggests top-rated medicines from real patient reviews.

**Live app:** (https://kritikamohan-medicine-recsys.streamlit.app/)

> ⚠️ **Disclaimer:** This project is for educational purposes only. It is decision support, not medical advice. Always consult a doctor or pharmacist before taking any medicine.

## How it works

1. **Symptom → condition:** a Random Forest classifier takes the selected symptoms (131 binary features) and returns the top 3 likely diseases out of 41.
2. **Condition → medicines:** for each predicted disease, drugs are ranked using patient reviews from Drugs.com. The score uses the rating, how many reviews the drug has, and how useful other readers found each review.
3. **Personalization:** drugs matching a user's entered allergies are removed, and an extra caution is shown for children and adults over 65.
4. **Safety:** emergency conditions (heart attack, brain hemorrhage) show an urgent-care alert and no medicines.

## Project structure

```
medicine-recsys/
├── app.py                  # Streamlit app
├── requirements.txt        # pinned dependencies
├── models/
│   └── symptom_model.joblib    # trained classifier + symptom list
├── outputs/
│   ├── drug_rankings.csv       # ranked drugs per disease
│   ├── description.csv         # disease descriptions
│   └── precautions_df.csv      # precautions per disease
└── notebooks/                  # data exploration, training, drug ranking
```

## Datasets

- **Symptom–disease data** (`Training.csv`, `description.csv`, `precautions_df.csv`): the public Kaggle disease-symptom dataset, 4,920 rows, 132 symptoms, 41 diseases.
- **Drug reviews** (`drugsComTest_raw.csv`): the UCI / Kaggle Drugs.com review dataset, 53,766 reviews in the test split.

Raw data is not included in this repository. Check each dataset's license on its source page.

## Key findings

- **Heavy duplication:** `Training.csv` has 4,920 rows but only **304 unique** ones (4,616 exact duplicates, 120 copies per disease).
- **Leakage:** in a random 70/30 split of the raw data, **1,470 of 1,476** test rows (99.6%) had an identical copy in the training set, so the raw 100% accuracy only shows memorization.
- **Honest evaluation:** after removing duplicates and using stratified 5-fold cross-validation, Random Forest still scored **1.0 top-1 and 1.0 top-3**. The diseases are easy to tell apart because each has its own small symptom set (about 8 symptoms per disease on average). Real users may enter fewer symptoms, so real-world accuracy will likely be lower than this score.
- **Data bug fixed:** the file has two columns named `fluid_overload`. One is all zeros and the other holds the real data (only in Alcoholic hepatitis), so the empty one was dropped and the real one kept.
- **Unusable file:** `medications.csv` appears misaligned (for example, Pneumonia lists laxatives and Tuberculosis lists cough suppressants), so it was not used. Medicines come from review data instead.
- **Drug ranking:** 7,701 reviews were mapped to 18 diseases. A Bayesian-style score stops drugs with only a few reviews from outranking well-reviewed ones.

## Limitations

- The symptom dataset looks synthetic and near-perfectly separable, so the accuracy does not reflect real clinical performance.
- Only 18 of the 41 diseases

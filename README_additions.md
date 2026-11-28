<!-- Paste each block into your existing README where indicated. Delete this file afterwards. -->

<!-- ===== 1. Under "Features", add these bullets ===== -->
- **Optional accounts:** register and sign in to save checks. Guests can use everything except saved history.
- **My history:** view saved checks, export them as **CSV**, reopen a past check, and delete all saved data.
- **PDF report** of any check, to share with a doctor.
- **Automated tests** for the model files, accounts, history, and PDF generation.

<!-- ===== 2. New section, put it after "Features" ===== -->
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

<!-- ===== 3. New section: architecture (after "How it works" or in its place) ===== -->
## Architecture

```
                      User (guest or signed in)
                                  |
                                  v
                        Streamlit app (app.py)
                                  |
        +-------------------------+--------------------------+
        |                         |                          |
        v                         v                          v
   Account & history        Prediction & ranking          PDF report
     (db.py, SQLite)      (Random Forest + drug table)     (report.py)
```

Passwords are stored only as a random salt plus a PBKDF2-HMAC-SHA256 hash (200,000 iterations). Each user can only see their own history.

<!-- ===== 4. In "Project structure", add these lines ===== -->
├── db.py                           # accounts and saved history (SQLite)
├── report.py                       # PDF report builder
├── tests/
│   └── test_mediguide.py           # unit tests
└── docs/
    ├── data_dictionary.csv         # meaning of every output column
    └── screenshots/

<!-- ===== 5. New section, put it after "Run locally" ===== -->
## Run the tests

```bash
python -m unittest discover -s tests -v
```

The tests check the model files, predictions, drug-ranking sanity, account creation and login, per-user history isolation, password hashing, and PDF generation.

<!-- ===== 6. Under "Limitations", add this bullet ===== -->
- Accounts are stored in a local SQLite file. On Streamlit Community Cloud this file can reset when the app restarts, so saved history is demo-level. A hosted PostgreSQL database (for example Supabase) would be the production upgrade, along with login rate limiting and password reset.

<!-- ===== 7. In "Tech stack", add ===== -->
ReportLab (PDF), SQLite (accounts and history)

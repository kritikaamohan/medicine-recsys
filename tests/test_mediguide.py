"""Run from the project root:  python -m unittest discover -s tests -v"""
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import db  # noqa: E402

try:
    import report
except ImportError:
    report = None


class TestArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bundle = joblib.load(BASE / "models" / "symptom_model.joblib")
        cls.model, cls.symptoms = bundle["model"], list(bundle["symptoms"])

    def test_model_matches_feature_list(self):
        self.assertEqual(len(self.model.classes_), 41)
        self.assertEqual(len(self.symptoms), self.model.n_features_in_)

    def test_skin_symptoms_predict_fungal_infection(self):
        row = pd.DataFrame([[1 if s in ("itching", "skin_rash", "nodal_skin_eruptions") else 0
                             for s in self.symptoms]], columns=self.symptoms)
        top = self.model.classes_[self.model.predict_proba(row)[0].argmax()]
        self.assertEqual(" ".join(top.split()), "Fungal infection")

    def test_condition_profiles(self):
        prof = pd.read_csv(BASE / "outputs" / "disease_profiles.csv")
        self.assertEqual(len(prof), 41)
        self.assertTrue((prof["symptoms"].str.count(r"\|") >= 3).all())

    def test_drug_rankings_are_sane(self):
        rank = pd.read_csv(BASE / "outputs" / "drug_rankings.csv")
        self.assertTrue(rank["score"].between(1, 10).all())
        self.assertFalse(rank.duplicated(["disease", "drugName"]).any())
        for _, g in rank.groupby("disease"):
            self.assertEqual(sorted(g["rank"]), list(range(1, len(g) + 1)))

    def test_stress_test_accuracy_falls_with_fewer_symptoms(self):
        s = pd.read_csv(BASE / "outputs" / "stress_test.csv").sort_values("symptoms_kept")
        self.assertTrue(s["top1"].is_monotonic_increasing)
        self.assertTrue((s["top3"] >= s["top1"]).all())


class TestDatabase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        self.path = tmp.name
        db.init(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_register_and_login(self):
        self.assertTrue(db.register("kritika_1", "longpassword", self.path)[0])
        self.assertIsNotNone(db.login("kritika_1", "longpassword", self.path))
        self.assertIsNone(db.login("kritika_1", "wrongpassword", self.path))
        self.assertIsNone(db.login("nobody", "longpassword", self.path))

    def test_duplicate_and_weak_inputs_rejected(self):
        db.register("sam", "longpassword", self.path)
        self.assertFalse(db.register("SAM", "otherpassword", self.path)[0])
        self.assertFalse(db.register("ab", "longpassword", self.path)[0])
        self.assertFalse(db.register("good_name", "short", self.path)[0])

    def test_password_is_not_stored_in_plain_text(self):
        db.register("sam", "longpassword", self.path)
        conn = sqlite3.connect(self.path)
        try:
            row = conn.execute("SELECT salt, pw_hash FROM users").fetchone()
        finally:
            conn.close()
        self.assertNotIn("longpassword", "".join(row))

    def _sample_result(self):
        # same shape the app produces; 'drugs' is a DataFrame there
        return {"symptoms": ["itching"], "age": 30, "time": "now", "red_flags": [],
                "results": [{"disease": "Acne", "label": "Acne", "share": 0.5,
                             "matched": ["itching"], "others": ["skin_rash"],
                             "description": "d", "precautions": ["wash face"],
                             "emergency": False, "removed": [],
                             "drugs": pd.DataFrame({"drugName": ["X"], "n_reviews": [5],
                                                    "mean_rating": [8.1]})}]}

    def test_history_save_list_delete(self):
        db.register("sam", "longpassword", self.path)
        user = db.login("sam", "longpassword", self.path)
        db.save_check(user["id"], self._sample_result(), self.path)
        saved = db.list_checks(user["id"], self.path)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["result"]["results"][0]["drugs"][0]["drugName"], "X")
        db.delete_checks(user["id"], self.path)
        self.assertEqual(db.list_checks(user["id"], self.path), [])

    def test_users_only_see_their_own_history(self):
        db.register("amy", "longpassword", self.path)
        db.register("bob", "longpassword", self.path)
        amy = db.login("amy", "longpassword", self.path)
        bob = db.login("bob", "longpassword", self.path)
        db.save_check(amy["id"], self._sample_result(), self.path)
        self.assertEqual(len(db.list_checks(amy["id"], self.path)), 1)
        self.assertEqual(db.list_checks(bob["id"], self.path), [])


@unittest.skipIf(report is None, "reportlab not installed")
class TestReport(unittest.TestCase):
    def test_pdf_from_dataframe_and_from_saved_list(self):
        res = TestDatabase._sample_result(None)
        self.assertTrue(report.build_pdf(res, "sam").startswith(b"%PDF"))
        res["results"][0]["drugs"] = res["results"][0]["drugs"].to_dict("records")
        self.assertTrue(report.build_pdf(res).startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()

"""Run from the project root:  python -m unittest discover -s tests -v"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db  # noqa: E402
import engine  # noqa: E402

try:
    import report
except ImportError:
    report = None


class TestEngine(unittest.TestCase):
    def test_artifacts_are_consistent(self):
        d = engine.load()
        self.assertEqual(len(d["profiles"]), 41)
        self.assertTrue(all(len(v) >= 4 for v in d["profiles"].values()))
        self.assertEqual(len(d["model"].classes_), 41)
        self.assertEqual(len(d["symptoms"]), d["model"].n_features_in_)

    def test_returns_three_predictions(self):
        res = engine.analyse(["itching", "skin_rash", "nodal_skin_eruptions"], 30)
        self.assertEqual(len(res["results"]), 3)
        self.assertEqual(res["results"][0]["disease"], "Fungal infection")

    def test_match_share_is_a_fraction(self):
        res = engine.analyse(["headache", "nausea"], 40)
        for r in res["results"]:
            self.assertTrue(0.0 <= r["share"] <= 1.0)

    def test_red_flag_symptom_is_reported(self):
        res = engine.analyse(["chest_pain"], 50)
        self.assertIn("chest_pain", res["red_flags"])

    def test_allergy_removes_matching_drugs(self):
        syms = ["headache", "nausea", "visual_disturbances", "blurred_and_distorted_vision"]
        res = engine.analyse(syms, 30, "aspirin")
        for r in res["results"]:
            for drug in r["drugs"]:
                self.assertNotIn("aspirin", drug["drugName"].lower())

    def test_drug_rankings_are_sane(self):
        rank = engine.load()["rank"]
        self.assertTrue(rank["score"].between(1, 10).all())
        self.assertFalse(rank.duplicated(["disease", "drugName"]).any())
        for _, g in rank.groupby("disease"):
            self.assertEqual(sorted(g["rank"]), list(range(1, len(g) + 1)))

    def test_unknown_symptom_raises(self):
        with self.assertRaises(ValueError):
            engine.analyse(["not_a_symptom"], 30)


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.path = self.tmp.name
        db.init(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_register_and_login(self):
        ok, _ = db.register("kritika_1", "longpassword", self.path)
        self.assertTrue(ok)
        self.assertIsNotNone(db.login("kritika_1", "longpassword", self.path))
        self.assertIsNone(db.login("kritika_1", "wrongpassword", self.path))
        self.assertIsNone(db.login("nobody", "longpassword", self.path))

    def test_duplicate_and_weak_inputs_rejected(self):
        db.register("sam", "longpassword", self.path)
        self.assertFalse(db.register("SAM", "otherpassword", self.path)[0])   # case-insensitive
        self.assertFalse(db.register("ab", "longpassword", self.path)[0])     # too short
        self.assertFalse(db.register("good_name", "short", self.path)[0])     # weak password

    def test_password_is_not_stored_in_plain_text(self):
        db.register("sam", "longpassword", self.path)
        import sqlite3
        row = sqlite3.connect(self.path).execute("SELECT salt, pw_hash FROM users").fetchone()
        self.assertNotIn("longpassword", "".join(row))

    def test_history_save_list_delete(self):
        db.register("sam", "longpassword", self.path)
        user = db.login("sam", "longpassword", self.path)
        res = engine.analyse(["itching", "skin_rash"], 25)
        db.save_check(user["id"], res, self.path)
        saved = db.list_checks(user["id"], self.path)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["result"]["symptoms"], ["itching", "skin_rash"])
        db.delete_checks(user["id"], self.path)
        self.assertEqual(db.list_checks(user["id"], self.path), [])

    def test_users_only_see_their_own_history(self):
        db.register("amy", "longpassword", self.path)
        db.register("bob", "longpassword", self.path)
        amy = db.login("amy", "longpassword", self.path)
        bob = db.login("bob", "longpassword", self.path)
        db.save_check(amy["id"], engine.analyse(["itching"], 20), self.path)
        self.assertEqual(len(db.list_checks(amy["id"], self.path)), 1)
        self.assertEqual(db.list_checks(bob["id"], self.path), [])


@unittest.skipIf(report is None, "reportlab not installed")
class TestReport(unittest.TestCase):
    def test_pdf_is_built(self):
        pdf = report.build_pdf(engine.analyse(["itching", "skin_rash", "chest_pain"], 70), "sam")
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreater(len(pdf), 1500)


if __name__ == "__main__":
    unittest.main()

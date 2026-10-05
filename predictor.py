from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
reg = joblib.load(BASE_DIR / "models" / "regressor.joblib")
clf = joblib.load(BASE_DIR / "models" / "classifier.joblib")


def grade_band(score: float) -> str:
    if score >= 80:
        return "Excellent"
    if score >= 65:
        return "Good"
    if score >= 50:
        return "Average"
    return "At Risk"


def risk_level(pass_probability: float) -> str:
    if pass_probability < 0.40:
        return "High Risk"
    if pass_probability < 0.70:
        return "Moderate Risk"
    return "Low Risk"


def predict(student: dict) -> dict:
    row = pd.DataFrame([student])
    score = float(min(100, max(0, reg.predict(row)[0])))
    p_pass = float(clf.predict_proba(row)[0][1])

    tips = []
    if student["study_hours"] < 2.5:
        tips.append("Increase daily study time toward 3 or more focused hours.")
    if student["attendance"] < 75:
        tips.append("Improve attendance and aim for at least 85% where possible.")
    if student["absences"] > 8:
        tips.append("Reduce avoidable absences and catch up on missed lessons quickly.")
    if student["sleep_hours"] < 6:
        tips.append("Target 7–8 hours of sleep to support concentration and recovery.")
    if student["failures"] > 0:
        tips.append("Prioritize subjects linked to previous failures and use mentor support.")
    if student["previous_grade"] < 50:
        tips.append("Start with foundational topics and use weekly practice tests.")
    if not tips:
        tips.append("Maintain the current study routine and monitor progress regularly.")

    return {
        "score": round(score, 1),
        "band": grade_band(score),
        "result": "Pass" if p_pass >= 0.5 else "Fail",
        "pass_probability": round(p_pass * 100, 1),
        "risk": risk_level(p_pass),
        "tips": tips,
    }

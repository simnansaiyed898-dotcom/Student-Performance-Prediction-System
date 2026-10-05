"""Generates a synthetic student dataset (1000 rows) whose columns mirror the
public UCI 'Student Performance' dataset. Replace data/students.csv with the
real UCI file if your teacher requires it - the rest of the pipeline is unchanged."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 1000
df = pd.DataFrame({
    "gender": rng.choice(["Male", "Female"], n),
    "age": rng.integers(15, 20, n),
    "study_hours": np.clip(rng.normal(3, 1.5, n), 0, 8).round(1),
    "attendance": np.clip(rng.normal(82, 12, n), 40, 100).round(0),
    "previous_grade": np.clip(rng.normal(62, 15, n), 15, 100).round(0),
    "absences": np.clip(rng.poisson(5, n), 0, 30),
    "failures": rng.choice([0, 1, 2, 3], n, p=[.72, .17, .07, .04]),
    "parent_education": rng.choice(["None", "School", "Graduate", "Postgraduate"], n, p=[.08, .40, .37, .15]),
    "internet_access": rng.choice(["Yes", "No"], n, p=[.82, .18]),
    "extra_activities": rng.choice(["Yes", "No"], n),
    "sleep_hours": np.clip(rng.normal(7, 1.1, n), 4, 10).round(1),
    "free_time": rng.integers(1, 6, n),
})
edu = df.parent_education.map({"None": 0, "School": 1, "Graduate": 2, "Postgraduate": 3})
score = (0.55*df.previous_grade + 4.2*df.study_hours + 0.22*df.attendance
         - 0.6*df.absences - 5.5*df.failures + 1.8*edu
         + 2.5*(df.internet_access == "Yes") + 1.2*(df.extra_activities == "Yes")
         + 1.5*(df.sleep_hours - 7).clip(-3, 1) - 0.8*(df.free_time - 3)
         + rng.normal(0, 4.5, n) - 3)
df["final_score"] = np.clip(score, 0, 100).round(0)
df["result"] = np.where(df.final_score >= 60, "Pass", "Fail")
df.to_csv("data/students.csv", index=False)
print(df.shape, df.result.value_counts().to_dict())

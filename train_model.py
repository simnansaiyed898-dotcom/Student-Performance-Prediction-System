"""EDA charts + regression & classification model training."""
import json, joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)

sns.set_theme(style="whitegrid")
df = pd.read_csv("data/students.csv")
CAT = ["gender", "parent_education", "internet_access", "extra_activities"]
NUM = ["age", "study_hours", "attendance", "previous_grade", "absences", "failures", "sleep_hours", "free_time"]
FEATURES = CAT + NUM

# ---------- EDA ----------
df.describe().T.round(2).to_csv("models/describe.csv")
fig, ax = plt.subplots(figsize=(6, 4)); sns.histplot(df.final_score, bins=25, kde=True, color="#2C7A7B", ax=ax)
ax.set_title("Distribution of Final Score"); fig.tight_layout(); fig.savefig("charts/score_dist.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(6, 4)); df.result.value_counts().plot.bar(color=["#2C7A7B", "#E07A5F"], ax=ax)
ax.set_title("Pass vs Fail Count"); ax.set_xlabel(""); plt.xticks(rotation=0); fig.tight_layout(); fig.savefig("charts/pass_fail.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(7, 6)); sns.heatmap(df[NUM + ["final_score"]].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 7})
ax.set_title("Correlation Heatmap"); fig.tight_layout(); fig.savefig("charts/corr.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(6, 4)); sns.scatterplot(data=df, x="study_hours", y="final_score", hue="result", palette=["#E07A5F", "#2C7A7B"], alpha=.6, ax=ax)
ax.set_title("Study Hours vs Final Score"); fig.tight_layout(); fig.savefig("charts/study_vs_score.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(6, 4)); sns.boxplot(data=df, x="failures", y="final_score", color="#A7C4C2", ax=ax)
ax.set_title("Past Failures vs Final Score"); fig.tight_layout(); fig.savefig("charts/failures_box.png", dpi=150); plt.close()

# ---------- Preprocessing ----------
pre = ColumnTransformer([("cat", OneHotEncoder(drop="first"), CAT), ("num", StandardScaler(), NUM)])
X = df[FEATURES]
Xtr, Xte, ytr_r, yte_r, ytr_c, yte_c = train_test_split(X, df.final_score, (df.result == "Pass").astype(int),
                                                       test_size=0.2, random_state=42, stratify=df.result)
results = {"regression": {}, "classification": {}}

# ---------- Regression ----------
regs = {"Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42)}
best_reg, best_r2 = None, -9
for name, m in regs.items():
    p = Pipeline([("pre", pre), ("m", m)]).fit(Xtr, ytr_r); pr = p.predict(Xte)
    r = dict(MAE=mean_absolute_error(yte_r, pr), RMSE=float(np.sqrt(mean_squared_error(yte_r, pr))), R2=r2_score(yte_r, pr))
    results["regression"][name] = {k: round(float(v), 3) for k, v in r.items()}
    if r["R2"] > best_r2: best_r2, best_reg, best_reg_pipe, best_pred = r["R2"], name, p, pr
fig, ax = plt.subplots(figsize=(5.5, 5)); ax.scatter(yte_r, best_pred, alpha=.5, color="#2C7A7B")
ax.plot([0, 100], [0, 100], "--", color="#E07A5F"); ax.set_xlabel("Actual score"); ax.set_ylabel("Predicted score")
ax.set_title(f"Actual vs Predicted ({best_reg})"); fig.tight_layout(); fig.savefig("charts/actual_vs_pred.png", dpi=150); plt.close()

# ---------- Classification ----------
clfs = {"Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)}
best_clf, best_f1 = None, -1
for name, m in clfs.items():
    p = Pipeline([("pre", pre), ("m", m)]).fit(Xtr, ytr_c); pr = p.predict(Xte)
    cv = cross_val_score(Pipeline([("pre", pre), ("m", m)]), X, (df.result == "Pass").astype(int), cv=5, scoring="accuracy").mean()
    r = dict(Accuracy=accuracy_score(yte_c, pr), Precision=precision_score(yte_c, pr), Recall=recall_score(yte_c, pr),
             F1=f1_score(yte_c, pr), CV_Accuracy=cv)
    results["classification"][name] = {k: round(float(v), 3) for k, v in r.items()}
    if r["F1"] > best_f1: best_f1, best_clf, best_clf_pipe, best_cpred = r["F1"], name, p, pr
fig, ax = plt.subplots(figsize=(4.5, 4)); ConfusionMatrixDisplay(confusion_matrix(yte_c, best_cpred), display_labels=["Fail", "Pass"]).plot(cmap="Blues", ax=ax, colorbar=False)
ax.set_title(f"Confusion Matrix ({best_clf})"); fig.tight_layout(); fig.savefig("charts/confusion.png", dpi=150); plt.close()

# ---------- Feature importance (Random Forest classifier) ----------
rf = Pipeline([("pre", pre), ("m", RandomForestClassifier(n_estimators=200, random_state=42))]).fit(Xtr, ytr_c)
names = rf.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf.named_steps["m"].feature_importances_, index=[n.split("__")[1] for n in names]).sort_values().tail(10)
fig, ax = plt.subplots(figsize=(6, 4.5)); imp.plot.barh(color="#2C7A7B", ax=ax); ax.set_title("Top 10 Feature Importances")
fig.tight_layout(); fig.savefig("charts/feature_importance.png", dpi=150); plt.close()
results["top_features"] = {k: round(float(v), 3) for k, v in imp.iloc[::-1].items()}

results.update(best_regressor=best_reg, best_classifier=best_clf, n_rows=len(df), n_train=len(Xtr), n_test=len(Xte))
joblib.dump(best_reg_pipe, "models/regressor.joblib"); joblib.dump(best_clf_pipe, "models/classifier.joblib")
json.dump(results, open("models/results.json", "w"), indent=2)
print(json.dumps(results, indent=2))

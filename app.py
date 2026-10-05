"""Production-style Streamlit dashboard for Student Performance Prediction."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
from predictor import predict

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "students.csv"
RESULTS_PATH = BASE_DIR / "models" / "results.json"

st.set_page_config(page_title="Student Performance AI", page_icon="🎓", layout="wide")

# ---------- Helpers ----------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_csv(student, result):
    record = {**student,
              "predicted_score": result["score"],
              "performance_band": result["band"],
              "predicted_result": result["result"],
              "pass_probability": result["pass_probability"],
              "risk_level": result["risk"]}
    return pd.DataFrame([record]).to_csv(index=False).encode("utf-8")

results = load_json(RESULTS_PATH)
df = pd.read_csv(DATA_PATH)

st.title("🎓 Student Performance Prediction System")
st.caption("AI-assisted academic performance analysis • Regression + Classification")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Prediction", "Analytics", "Model Performance", "About"], label_visibility="collapsed")
    st.divider()
    st.info("Demo data is synthetic. Replace `data/students.csv` with an approved real dataset before making real-world decisions.")

if page == "Prediction":
    st.subheader("Enter Student Details")
    st.write("The system predicts an estimated final score, pass probability, performance band, risk level, and improvement suggestions.")

    with st.form("student_form"):
        left, right = st.columns(2)
        with left:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 15, 19, 17)
            study_hours = st.slider("Study hours per day", 0.0, 8.0, 3.0, 0.5)
            attendance = st.slider("Attendance (%)", 40, 100, 80)
            previous_grade = st.slider("Previous exam grade (%)", 15, 100, 60)
            absences = st.slider("Number of absences", 0, 30, 4)
        with right:
            failures = st.selectbox("Past failures", [0, 1, 2, 3])
            parent_education = st.selectbox("Parent education", ["None", "School", "Graduate", "Postgraduate"], index=1)
            internet_access = st.selectbox("Internet access at home", ["Yes", "No"])
            extra_activities = st.selectbox("Extra-curricular activities", ["Yes", "No"])
            sleep_hours = st.slider("Sleep hours per night", 4.0, 10.0, 7.0, 0.5)
            free_time = st.slider("Free time after school (1–5)", 1, 5, 3)
        submitted = st.form_submit_button("🚀 Predict Performance", type="primary", use_container_width=True)

    if submitted:
        student = dict(gender=gender, age=age, study_hours=study_hours, attendance=attendance,
                       previous_grade=previous_grade, absences=absences, failures=failures,
                       parent_education=parent_education, internet_access=internet_access,
                       extra_activities=extra_activities, sleep_hours=sleep_hours, free_time=free_time)
        result = predict(student)
        st.session_state["last_prediction"] = (student, result)

    if "last_prediction" in st.session_state:
        student, result = st.session_state["last_prediction"]
        st.divider()
        st.subheader("Prediction Result")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Score", f"{result['score']} / 100")
        m2.metric("Result", result["result"])
        m3.metric("Pass Probability", f"{result['pass_probability']}%")
        m4.metric("Risk Level", result["risk"])

        if result["result"] == "Pass":
            st.success(f"Performance band: **{result['band']}**")
        else:
            st.warning(f"Performance band: **{result['band']}**")

        progress = max(0, min(100, result["score"])) / 100
        st.progress(progress, text=f"Estimated score: {result['score']} / 100")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 📌 Student Profile")
            st.dataframe(pd.DataFrame([student]).T.rename(columns={0: "Value"}), use_container_width=True)
        with c2:
            st.markdown("### 💡 Recommended Actions")
            for tip in result["tips"]:
                st.write("• " + tip)

        st.download_button("⬇️ Download Prediction Report", build_csv(student, result),
                           file_name="student_prediction_report.csv", mime="text/csv")

elif page == "Analytics":
    st.subheader("📊 Dataset Analytics")
    a, b, c, d = st.columns(4)
    a.metric("Students", len(df))
    b.metric("Average Score", f"{df.final_score.mean():.1f}")
    c.metric("Pass Rate", f"{(df.result.eq('Pass').mean()*100):.1f}%")
    d.metric("Average Attendance", f"{df.attendance.mean():.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Study Hours vs Final Score**")
        st.scatter_chart(df, x="study_hours", y="final_score", use_container_width=True)
    with c2:
        st.markdown("**Attendance vs Final Score**")
        st.scatter_chart(df, x="attendance", y="final_score", use_container_width=True)

    st.markdown("### Sample Dataset")
    st.dataframe(df.head(20), use_container_width=True)

elif page == "Model Performance":
    st.subheader("🤖 Model Performance")
    st.caption("Metrics are calculated on the project's held-out test set; classification also includes 5-fold cross-validation accuracy.")

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("### Regression")
        reg_df = pd.DataFrame(results["regression"]).T
        st.dataframe(reg_df.style.format("{:.3f}"), use_container_width=True)
        st.info(f"Selected model: **{results['best_regressor']}**")
    with r2:
        st.markdown("### Classification")
        clf_df = pd.DataFrame(results["classification"]).T
        st.dataframe(clf_df.style.format("{:.3f}"), use_container_width=True)
        st.info(f"Selected model: **{results['best_classifier']}**")

    st.markdown("### Most Influential Features")
    features = pd.Series(results["top_features"]).sort_values(ascending=False)
    st.bar_chart(features)

elif page == "About":
    st.subheader("ℹ️ About the Project")
    st.markdown("""
    **Goal:** provide an educational decision-support prototype that estimates student performance from academic and lifestyle inputs.

    **Machine learning:**
    - Regression predicts a continuous final score.
    - Classification predicts Pass/Fail probability.
    - Categorical data is one-hot encoded and numeric features are standardized.
    - Candidate models are evaluated and the strongest model by the configured metric is saved for the application.

    **Important:** this is a demonstration model, not a replacement for teachers, counselors, or institutional assessment. Predictions should be treated as estimates and not used as the sole basis for high-impact decisions.
    """)
    st.markdown("### Project Workflow")
    st.code("Student details → Preprocessing → Trained ML models → Score + Pass probability → Risk + Suggestions", language="text")

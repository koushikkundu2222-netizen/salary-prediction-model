import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# Streamlit Page Setup
st.set_page_config(page_title="Salary Prediction (Ensemble Models)", layout="wide")
st.title("💼 Salary Prediction using Ensemble Machine Learning Models")

st.markdown("""
This app predicts **employee salaries** based on:
- Age  
- Gender  
- Education Level  
- Job Title  
- Years of Experience  

Using **Random Forest**, **Gradient Boosting**, and a **Voting Ensemble** for improved accuracy.
""")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data(path="salary_data_cleaned.csv"):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        return None

data = load_data()

if data is None:
    st.error("❌ Could not find `salary_data_cleaned.csv`. Please place it in the same folder as this app.")
    st.stop()

# ===============================
# DATA PREVIEW
# ===============================
st.subheader("📘 Dataset Preview")
if st.checkbox("Show first 10 rows"):
    st.dataframe(data.head(10))

# Ensure expected columns exist
expected_columns = ["Age", "Gender", "Education Level", "Job Title", "Years of Experience", "Salary"]
missing_cols = [col for col in expected_columns if col not in data.columns]

if missing_cols:
    st.error(f"❌ Missing columns: {missing_cols}. Please make sure your dataset has these exact columns.")
    st.stop()

# ===============================
# FEATURE / TARGET SPLIT
# ===============================
X = data[["Age", "Gender", "Education Level", "Job Title", "Years of Experience"]]
y = data["Salary"]

categorical_cols = ["Gender", "Education Level", "Job Title"]
numeric_cols = ["Age", "Years of Experience"]

# ===============================
# PREPROCESSING PIPELINES
# ===============================
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

# ⚠️ FIX: Make OneHotEncoder output dense for compatibility with HistGradientBoostingRegressor
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# ===============================
# TRAIN-TEST SPLIT
# ===============================
test_size = st.slider("Test set size (%)", 10, 40, 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=42
)

st.write(f"🧠 Training samples: {X_train.shape[0]} | 🧪 Test samples: {X_test.shape[0]}")

# ===============================
# MODEL BUILDING
# ===============================
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])

gb_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", HistGradientBoostingRegressor(random_state=42))
])

voting_model = VotingRegressor(estimators=[
    ("rf", rf_model),
    ("gb", gb_model)
])

# ===============================
# TRAINING
# ===============================
def train_models(rf, gb, voting, X_train, y_train):
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    voting.fit(X_train, y_train)
    return rf, gb, voting

with st.spinner("⏳ Training models... please wait"):
    rf_model, gb_model, voting_model = train_models(rf_model, gb_model, voting_model, X_train, y_train)
st.success("✅ Models trained successfully!")

# ===============================
# EVALUATION
# ===============================
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    return {"MSE": mse, "RMSE": rmse, "R²": r2, "preds": preds}

rf_metrics = evaluate(rf_model, X_test, y_test)
gb_metrics = evaluate(gb_model, X_test, y_test)
voting_metrics = evaluate(voting_model, X_test, y_test)

metrics_df = pd.DataFrame({
    "Model": ["Random Forest", "Gradient Boosting", "Voting Ensemble"],
    "MSE": [rf_metrics["MSE"], gb_metrics["MSE"], voting_metrics["MSE"]],
    "RMSE": [rf_metrics["RMSE"], gb_metrics["RMSE"], voting_metrics["RMSE"]],
    "R² Score": [rf_metrics["R²"], gb_metrics["R²"], voting_metrics["R²"]]
})

st.subheader("📊 Model Performance Comparison")
st.dataframe(metrics_df.style.format({"MSE": "{:.2f}", "RMSE": "{:.2f}", "R² Score": "{:.3f}"}))

best_model_name = metrics_df.sort_values("RMSE").iloc[0]["Model"]
st.success(f"🏆 Best model: **{best_model_name}**")

# ===============================
# VISUALIZATION
# ===============================
st.subheader("📈 Actual vs Predicted Salaries")

best_model = {
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "Voting Ensemble": voting_model
}[best_model_name]

best_preds = best_model.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(y_test, best_preds, alpha=0.7, color="teal")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
ax.set_xlabel("Actual Salary")
ax.set_ylabel("Predicted Salary")
ax.set_title(f"Actual vs Predicted ({best_model_name})")
st.pyplot(fig)

# ===============================
# FEATURE IMPORTANCE (RF)
# ===============================
st.subheader("🔥 Feature Importance (Random Forest)")

rf_fitted = rf_model.named_steps["regressor"]
pre = rf_model.named_steps["preprocessor"]
cat_names = pre.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_cols)
feature_names = numeric_cols + list(cat_names)
importances = rf_fitted.feature_importances_

fi = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False).head(15)
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.barplot(y="Feature", x="Importance", data=fi, ax=ax2)
ax2.set_title("Top Feature Importances (Random Forest)")
st.pyplot(fig2)

# ===============================
# MANUAL PREDICTION FORM
# ===============================
st.subheader("🔮 Predict Salary Manually")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    gender = st.selectbox("Gender", options=sorted(data["Gender"].dropna().unique()))
    education = st.selectbox("Education Level", options=sorted(data["Education Level"].dropna().unique()))
    job = st.selectbox("Job Title", options=sorted(data["Job Title"].dropna().unique()))
    experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    submit = st.form_submit_button("Predict Salary 💰")

if submit:
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Education Level": education,
        "Job Title": job,
        "Years of Experience": experience
    }])
    pred_salary = best_model.predict(input_df)[0]
    st.success(f"💰 Predicted Salary: ₹{pred_salary:,.2f}")

# ===============================
# FOOTER
# ===============================
st.write("---")
st.markdown("📘 **Tip:** Use a large, clean dataset (like yours!) for better accuracy. Ensemble models combine the power of multiple algorithms to make stronger predictions.")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# === Load Data ===
df = pd.read_csv(r"C:\Users\Admin\Desktop\Edu_Student_Teacher\data\edu_mentor_dataset_final.csv")
df = df.sample(20000, random_state=42)

# === Encode Categorical ===
categorical_cols = ['teacher_comments_summary', 'learning_style', 'content_type_preference']
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# === Columns Setup ===
target_col = 'risk_score'
exclude_cols = ['student_id', 'student_name', 'email_id', 'password', 'std', 'suggestions_required', 'is_at_risk']
feature_cols = [col for col in df.columns if col not in exclude_cols + [target_col]]

X = df[feature_cols]
y = df[target_col]

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Define Models ===
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(eval_metric='rmse', random_state=42)
}

results = {}

# === Train and Evaluate All Models ===
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {"model": model, "RMSE": rmse, "R2": r2}
    print(f"✅ {name} → RMSE: {rmse:.2f}, R²: {r2:.2f}")

# === Select Best Model (by lowest RMSE) ===
best_model_name = min(results, key=lambda k: results[k]["RMSE"])
best_model = results[best_model_name]["model"]
best_rmse = results[best_model_name]["RMSE"]
best_r2 = results[best_model_name]["R2"]

print(f"\n🏆 Best Model: {best_model_name} → RMSE: {best_rmse:.2f}, R²: {best_r2:.2f}")

# === Save Best Model and Artifacts ===
project = {
    "model": best_model,
    "model_name": best_model_name,
    "scaler": scaler,
    "label_encoders": le_dict,
    "feature_columns": feature_cols,
    "metrics": {
        "RMSE": best_rmse,
        "R2": best_r2,
        "all_models": results
    }
}
joblib.dump(project, r"C:\Users\Admin\Desktop\Edu_Student_Teacher\model\risk_score_model.pkl")

print("📦 Best model saved to risk_score_model.pkl")
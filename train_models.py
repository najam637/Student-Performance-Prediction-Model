"""Train the regression and classification models. Run: python train_models.py"""
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error

data = pd.read_csv("student_performance.csv")
features = [c for c in data.columns if c != "FinalGrade"]
X = data[features]
y_number = data["FinalGrade"]
labels = {0: "High", 1: "Good", 2: "Average", 3: "Low"}
y_category = data["FinalGrade"].map(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_number, test_size=0.2, random_state=42
)
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
reg_predictions = regression_model.predict(X_test)
r2 = r2_score(y_test, reg_predictions)
mae = mean_absolute_error(y_test, reg_predictions)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_category, test_size=0.2, random_state=42, stratify=y_category
)
classification_model = RandomForestClassifier(n_estimators=150, random_state=42)
classification_model.fit(X_train_c, y_train_c)
accuracy = accuracy_score(y_test_c, classification_model.predict(X_test_c))

joblib.dump(regression_model, "regression_model.joblib")
joblib.dump(classification_model, "classification_model.joblib")
metrics = {"r2": round(float(r2), 4), "mae": round(float(mae), 4),
           "accuracy": round(float(accuracy), 4)}
with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=2)

print(f"Regression R²: {r2:.4f}")
print(f"Regression MAE: {mae:.4f}")
print(f"Classification accuracy: {accuracy:.4f}")
print("Saved both joblib models.")

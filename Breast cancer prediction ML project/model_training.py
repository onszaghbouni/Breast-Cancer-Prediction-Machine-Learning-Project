import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# =======================
# 1. Load & Prepare Data
# =======================

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# =======================
# 2. Train Two Models
# =======================

log_model = LogisticRegression(max_iter=500)
log_model.fit(X_train, y_train)

tree_model = DecisionTreeClassifier(max_depth=5)
tree_model.fit(X_train, y_train)

# =======================
# 3. Evaluate
# =======================

log_pred = log_model.predict(X_test)
tree_pred = tree_model.predict(X_test)

acc_log = accuracy_score(y_test, log_pred)
acc_tree = accuracy_score(y_test, tree_pred)

print("Logistic Regression Accuracy:", acc_log)
print("Decision Tree Accuracy:", acc_tree)

print("\n--- Logistic Regression Report ---")
print(classification_report(y_test, log_pred))

print("\n--- Decision Tree Report ---")
print(classification_report(y_test, tree_pred))

# =======================
# 4. Save BEST Model
# =======================

if acc_log > acc_tree:
    best_model = log_model
    model_name = "best_model.pkl"
else:
    best_model = tree_model
    model_name = "best_model.pkl"

joblib.dump(best_model, model_name)
joblib.dump(scaler, "scaler.pkl")

print("\nMODEL SAVED AS:", model_name)
print("SCALER SAVED AS: scaler.pkl")

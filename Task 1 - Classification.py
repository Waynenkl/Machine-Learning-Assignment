import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from tensorflow import keras
from tensorflow.keras import layers
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, f1_score

sns.set(style="whitegrid", palette="muted")

# Load dataset
df = pd.read_csv("C:\\Users\\User\\Downloads\\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preprocessing & feature/target setup
target = "Churn"
df = df.drop(columns=["customerID"])  # drop ID
df[target] = df[target].map({"Yes":1, "No":0})

# Identify numeric / categorical columns
num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_features = [col for col in df.columns if col not in num_features + [target]]

# Convert TotalCharges to numeric and drop missing
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

X = df[num_features + cat_features]
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Preprocessor pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
])

# Handle imbalance on training set
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_proc, y_train)

# ================================
#  Model A: Traditional ML — Random Forest
# ================================
rf_model = ImbPipeline([
    ("preprocess", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", RandomForestClassifier(random_state=42))
])
param_grid = {
    "classifier__n_estimators": [100, 300],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5]
}
grid_rf = GridSearchCV(rf_model, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=0)
grid_rf.fit(X_train, y_train)

print("Best RF hyperparameters:", grid_rf.best_params_)
best_rf = grid_rf.best_estimator_

y_pred_rf = best_rf.predict(X_test)
y_pred_rf_proba = best_rf.predict_proba(X_test)[:,1]

print("RF Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("RF ROC AUC:", roc_auc_score(y_test, y_pred_rf_proba))

# ================================
#  Model B: Deep Neural Network (DNN)
# ================================
# Prepare data for DNN
X_proc = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(
    scaler.fit_transform(X_proc), y, test_size=0.2, stratify=y, random_state=42
)

# Handle imbalance for DNN training
X_train_dl_res, y_train_dl_res = SMOTE(random_state=42).fit_resample(X_train_dl, y_train_dl)

dnn = keras.Sequential([
    keras.Input(shape=(X_train_dl_res.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])
dnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC", "accuracy"])

history = dnn.fit(
    X_train_dl_res, y_train_dl_res,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

y_pred_dl_proba = dnn.predict(X_test_dl).flatten()
y_pred_dl = (y_pred_dl_proba > 0.5).astype(int)

print("DNN Classification Report:")
print(classification_report(y_test_dl, y_pred_dl))
print("DNN ROC AUC:", roc_auc_score(y_test_dl, y_pred_dl_proba))

# Visualization & Comparison
# Compute metrics
metrics = pd.DataFrame({
    "Model": ["Random Forest", "Deep Neural Network"],
    "ROC AUC": [
        roc_auc_score(y_test, y_pred_rf_proba),
        roc_auc_score(y_test_dl, y_pred_dl_proba)
    ],
    "Accuracy": [
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test_dl, y_pred_dl)
    ],
    "F1-Score": [
        f1_score(y_test, y_pred_rf),
        f1_score(y_test_dl, y_pred_dl)
    ]
})

# --- Bar Chart: Model Comparison ---
plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
sns.barplot(x="Model", y="ROC AUC", data=metrics, palette="Blues_d")
plt.title("ROC AUC Comparison")

plt.subplot(1,3,2)
sns.barplot(x="Model", y="Accuracy", data=metrics, palette="Greens_d")
plt.title("Accuracy Comparison")

plt.subplot(1,3,3)
sns.barplot(x="Model", y="F1-Score", data=metrics, palette="Purples_d")
plt.title("F1-Score Comparison")

plt.tight_layout()
plt.show()

# --- ROC Curves ---
plt.figure(figsize=(6,6))
RocCurveDisplay.from_predictions(y_test, y_pred_rf_proba, name="Random Forest")
RocCurveDisplay.from_predictions(y_test_dl, y_pred_dl_proba, name="DNN")
plt.title("ROC Curve Comparison")
plt.plot([0,1],[0,1],'k--', label="Random Guess")
plt.legend()
plt.show()

# --- Confusion Matrices ---
fig, axes = plt.subplots(1, 2, figsize=(12,5))

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Random Forest Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(confusion_matrix(y_test_dl, y_pred_dl), annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title("DNN Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# --- DNN Training Curves ---
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("DNN Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Binary Crossentropy Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["AUC"], label="Train AUC")
plt.plot(history.history["val_auc"], label="Val AUC")
plt.title("DNN Training vs Validation AUC")
plt.xlabel("Epochs")
plt.ylabel("AUC")
plt.legend()

plt.tight_layout()
plt.show()

# --- Display table of metrics ---
print("\n📋 Model Performance Summary:")
print(metrics.to_string(index=False))
# ===============================================
# REGRESSION TASK
# Dataset: Medical Insurance Cost Prediction
# ===============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from tensorflow import keras
from tensorflow.keras import layers

sns.set(style="whitegrid", palette="muted")

#  Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

#  Preprocessing setup
cat_features = ["sex", "smoker", "region"]
num_features = ["age", "bmi", "children"]
target = "charges"

X = df[cat_features + num_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Preprocessor for traditional ML
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_features)
])

# =======================================================
#  Model 1: Linear Regression (Traditional)
# =======================================================
lr_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", LinearRegression())
])
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# =======================================================
#  Model 2: Random Forest Regressor (Traditional)
# =======================================================
rf_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42))
])
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# =======================================================
#  Model 3: XGBoost Regressor (Advanced Ensemble)
# =======================================================
xgb_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror"
    ))
])
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# =======================================================
#  Model 4: Deep Neural Network (Deep Learning)
# =======================================================
# DNN needs manual preprocessing
X_proc = pd.get_dummies(X, drop_first=True)
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(X_proc, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_dl = scaler.fit_transform(X_train_dl)
X_test_dl = scaler.transform(X_test_dl)

dnn = keras.Sequential([
    keras.Input(shape=(X_train_dl.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1)
])
dnn.compile(optimizer="adam", loss="mse", metrics=["mae"])

history = dnn.fit(X_train_dl, y_train_dl, validation_split=0.2, epochs=100, batch_size=16, verbose=0)
y_pred_dl = dnn.predict(X_test_dl).flatten()

# =======================================================
#  Evaluation Metrics
# =======================================================
def evaluate_model(y_true, y_pred):
    return (
        mean_absolute_error(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred)),
        r2_score(y_true, y_pred)
    )

lr_mae, lr_rmse, lr_r2 = evaluate_model(y_test, y_pred_lr)
rf_mae, rf_rmse, rf_r2 = evaluate_model(y_test, y_pred_rf)
xgb_mae, xgb_rmse, xgb_r2 = evaluate_model(y_test, y_pred_xgb)
dl_mae, dl_rmse, dl_r2 = evaluate_model(y_test_dl, y_pred_dl)

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "XGBoost", "DNN"],
    "MAE": [lr_mae, rf_mae, xgb_mae, dl_mae],
    "RMSE": [lr_rmse, rf_rmse, xgb_rmse, dl_rmse],
    "R²": [lr_r2, rf_r2, xgb_r2, dl_r2]
})

print("\n Model Performance Comparison (Regression):")
print(results)

# =======================================================
#  Visualization — Performance Metrics & Predictions
# =======================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 🔸 Bar chart — MAE, RMSE, R²
sns.barplot(x="Model", y="MAE", data=results, ax=axes[0, 0])
axes[0, 0].set_title("Mean Absolute Error (↓)")

sns.barplot(x="Model", y="RMSE", data=results, ax=axes[0, 1])
axes[0, 1].set_title("Root Mean Squared Error (↓)")

sns.barplot(x="Model", y="R²", data=results, ax=axes[1, 0])
axes[1, 0].set_title("R² Score (↑)")

# 🔸 Scatter plot — Actual vs Predicted (Best 2 Models)
best_models = {"Random Forest": y_pred_rf, "XGBoost": y_pred_xgb}
colors = ["blue", "orange"]
for (name, preds), c in zip(best_models.items(), colors):
    sns.scatterplot(x=y_test, y=preds, alpha=0.6, label=name, color=c, ax=axes[1, 1])
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1, 1].set_title("Actual vs Predicted (Top 2 Models)")
axes[1, 1].set_xlabel("Actual Charges")
axes[1, 1].set_ylabel("Predicted Charges")
axes[1, 1].legend()

plt.tight_layout()
plt.suptitle("Regression Model Comparison — Insurance Charges", fontsize=16, y=1.02)
plt.show()

# 🔸 DNN Learning Curve (Training vs Validation Loss)
plt.figure(figsize=(6,4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("DNN Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()

#  Summary
print("\n📘 Summary:")
print("• Linear Regression gives a quick baseline but underfits (limited flexibility).")
print("• Random Forest improves accuracy by capturing nonlinear interactions.")
print("• XGBoost often performs best on structured/tabular data by boosting weak learners.")
print("• DNN captures complex relationships but may overfit small datasets.")
print("\nIn this dataset (~1300 rows), XGBoost or Random Forest usually provide the best trade-off "
      "between accuracy and generalization.")
# =======================================================
#  Regression Task — Comparing Tuned vs Untuned Models
# =======================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

#  Preprocessor
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_features)
])

# =======================================================
#  Model 1: Random Forest (Untuned)
# =======================================================
rf_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42))
])
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_r2 = r2_score(y_test, y_pred_rf)

# =======================================================
#  Model 2: Random Forest (Tuned via GridSearchCV)
# =======================================================
param_grid = {
    "model__n_estimators": [100, 300],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
}
rf_tuned = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])
grid_search = GridSearchCV(rf_tuned, param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

print("\n Best Hyperparameters for Random Forest (Tuned):")
print(grid_search.best_params_)
print(f"Best Cross-Validation R² Score: {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_
y_pred_rf_tuned = best_rf.predict(X_test)

rf_mae_tuned = mean_absolute_error(y_test, y_pred_rf_tuned)
rf_rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_rf_tuned))
rf_r2_tuned = r2_score(y_test, y_pred_rf_tuned)

# =======================================================
#  Model 3: DNN (Untuned)
# =======================================================
X_proc = pd.get_dummies(X, drop_first=True)
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(X_proc, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_dl = scaler.fit_transform(X_train_dl)
X_test_dl = scaler.transform(X_test_dl)

dnn = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=[X_train_dl.shape[1]]),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1)
])
dnn.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = dnn.fit(X_train_dl, y_train_dl, validation_split=0.2, epochs=100, batch_size=16, verbose=0)

y_pred_dl = dnn.predict(X_test_dl).flatten()
dl_mae = mean_absolute_error(y_test_dl, y_pred_dl)
dl_rmse = np.sqrt(mean_squared_error(y_test_dl, y_pred_dl))
dl_r2 = r2_score(y_test_dl, y_pred_dl)

# =======================================================
#  Model 4: DNN (Tuned)
# =======================================================
dnn_tuned = keras.Sequential([
    keras.Input(shape=(X_train_dl.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1)
])
dnn_tuned.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
history_tuned = dnn_tuned.fit(X_train_dl, y_train_dl, validation_split=0.2, epochs=150, batch_size=32, verbose=0)

y_pred_dl_tuned = dnn_tuned.predict(X_test_dl).flatten()
dl_mae_tuned = mean_absolute_error(y_test_dl, y_pred_dl_tuned)
dl_rmse_tuned = np.sqrt(mean_squared_error(y_test_dl, y_pred_dl_tuned))
dl_r2_tuned = r2_score(y_test_dl, y_pred_dl_tuned)

# =======================================================
#  Model Performance Comparison
# =======================================================
results = pd.DataFrame({
    "Model": [
        "Random Forest (Untuned)",
        "Random Forest (Tuned)",
        "DNN (Untuned)",
        "DNN (Tuned)"
    ],
    "MAE": [rf_mae, rf_mae_tuned, dl_mae, dl_mae_tuned],
    "RMSE": [rf_rmse, rf_rmse_tuned, dl_rmse, dl_rmse_tuned],
    "R²": [rf_r2, rf_r2_tuned, dl_r2, dl_r2_tuned]
})
print("\nModel Performance Comparison:")
print(results)

# =======================================================
#  Combined Visualization — All Models
# =======================================================
rf_residuals = y_test - y_pred_rf
rf_residuals_tuned = y_test - y_pred_rf_tuned
dl_residuals = y_test_dl - y_pred_dl
dl_residuals_tuned = y_test_dl - y_pred_dl_tuned

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
sns.set(style="whitegrid")

# Row 1: Bar Charts
sns.barplot(x="Model", y="MAE", data=results, ax=axes[0, 0])
axes[0, 0].set_title("Mean Absolute Error (↓)")
sns.barplot(x="Model", y="RMSE", data=results, ax=axes[0, 1])
axes[0, 1].set_title("Root Mean Squared Error (↓)")
sns.barplot(x="Model", y="R²", data=results, ax=axes[0, 2])
axes[0, 2].set_title("R² Score (↑)")
for ax in axes[0]:
    ax.tick_params(axis='x', rotation=20)

# Row 2: Prediction & Residuals
axes[1, 0].scatter(y_test, y_pred_rf, alpha=0.4, color='blue', label="RF Untuned")
axes[1, 0].scatter(y_test, y_pred_rf_tuned, alpha=0.4, color='orange', label="RF Tuned")
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1, 0].set_title("Random Forest: Actual vs Predicted")
axes[1, 0].set_xlabel("Actual Charges")
axes[1, 0].set_ylabel("Predicted Charges")
axes[1, 0].legend()

axes[1, 1].scatter(y_test_dl, y_pred_dl, alpha=0.4, color='green', label="DNN Untuned")
axes[1, 1].scatter(y_test_dl, y_pred_dl_tuned, alpha=0.4, color='purple', label="DNN Tuned")
axes[1, 1].plot([y_test_dl.min(), y_test_dl.max()], [y_test_dl.min(), y_test_dl.max()], 'r--')
axes[1, 1].set_title("DNN: Actual vs Predicted")
axes[1, 1].set_xlabel("Actual Charges")
axes[1, 1].set_ylabel("Predicted Charges")
axes[1, 1].legend()

sns.histplot(rf_residuals, color="blue", label="RF Untuned", kde=True, ax=axes[1, 2])
sns.histplot(rf_residuals_tuned, color="orange", label="RF Tuned", kde=True, ax=axes[1, 2])
sns.histplot(dl_residuals, color="green", label="DNN Untuned", kde=True, ax=axes[1, 2])
sns.histplot(dl_residuals_tuned, color="purple", label="DNN Tuned", kde=True, ax=axes[1, 2])
axes[1, 2].set_title("Residual Distribution Comparison")
axes[1, 2].set_xlabel("Prediction Error (Actual - Predicted)")
axes[1, 2].legend()

plt.tight_layout()
plt.suptitle("Regression Model Comparison — Tuned vs Untuned", fontsize=16, y=1.03)
plt.show()

# Optional DNN Learning Curves
plt.figure(figsize=(6,4))
plt.plot(history.history["loss"], label="Untuned Train Loss")
plt.plot(history.history["val_loss"], label="Untuned Val Loss")
plt.plot(history_tuned.history["loss"], label="Tuned Train Loss")
plt.plot(history_tuned.history["val_loss"], label="Tuned Val Loss")
plt.title("DNN Learning Curves (Untuned vs Tuned)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()


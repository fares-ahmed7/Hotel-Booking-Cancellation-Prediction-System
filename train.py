import pandas as pd
import warnings
import joblib

warnings.filterwarnings("ignore")

from src.preprocessing import (
    load_data,
    split_data,
    handle_missing_values,
    encode_categorical_features,
    scale_features,
)

from src.modeling import (
    build_logistic_regression,
    build_decision_tree,
    build_random_forest,
    build_gradient_boosting,
    build_xgboost,
    build_lightgbm,
)

from src.evaluation import (
    cross_validate_model,
    evaluate_model,
    print_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
)


# ==========================
# Load Data
# ==========================
print("Loading data...")

X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)

# ==========================
# Preprocessing
# ==========================
X_train, X_test = handle_missing_values(X_train, X_test)
X_train, X_test = encode_categorical_features(X_train, X_test)
X_train, X_test = scale_features(X_train, X_test)

# ==========================
# Models (NO FIT HERE)
# ==========================
models = {
    "Logistic Regression": build_logistic_regression(),
    "Decision Tree": build_decision_tree(),
    "Random Forest": build_random_forest(),
    "Gradient Boosting": build_gradient_boosting(),
    "XGBoost": build_xgboost(),
    "LightGBM": build_lightgbm(),
}

# ==========================
# Cross Validation ONLY
# ==========================
results = {}

print("\nRunning Cross Validation...\n")

for name, model in models.items():

    print(f"CV for: {name}")

    cv_mean, cv_std, _ = cross_validate_model(
        model,
        X_train,
        y_train,
        scoring="roc_auc",
        cv=5
    )

    results[name] = {
        "CV AUC": cv_mean,
        "CV Std": cv_std
    }

# ==========================
# Results Table
# ==========================
results_df = pd.DataFrame(results).T.sort_values("CV AUC", ascending=False)

print("\n================ RESULTS ================\n")
print(results_df)

# ==========================
# Best Model Selection
# ==========================
best_model_name = results_df.index[0]
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)

# ==========================
# FINAL FIT ONLY (BEST MODEL)
# ==========================
print("\nTraining Best Model on full training data...")

best_model.fit(X_train, y_train)

# ==========================
# Save Model
# ==========================
joblib.dump(best_model, "models/best_model.pkl")

# ==========================
# Save Results
# ==========================
results_df.to_csv("results/cv_results.csv")

# ==========================
# FULL EVALUATION ON TEST SET
# ==========================
print("\nEvaluating Best Model on Test Set...\n")

metrics = evaluate_model(best_model, X_test, y_test)

print("\nMetrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# ==========================
# Classification Report
# ==========================
print("\nClassification Report:")
print_classification_report(best_model, X_test, y_test)

# ==========================
# Plots
# ==========================
plot_confusion_matrix(best_model, X_test, y_test)
plot_roc_curve(best_model, X_test, y_test)

# ==========================
# Done
# ==========================
print("\n==============================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("==============================")
print("Best Model:", best_model_name)
print("Saved to: models/best_model.pkl")
print("Results saved to: results/cv_results.csv")
"""
Train and hyperparameter-tune the classical ML baseline: Random Forest and
SVM, both on the handcrafted features from extract_features.py.

Phase 4, Steps 17-18 of project_execution_algorithm.md.

Uses RandomizedSearchCV (a capped, smart search — not every possible
combination, which would be too slow for SVM on ~15k samples) with 5-fold
cross-validation on train+val combined. The test set is NEVER touched here —
it stays frozen until Phase 7.

Whichever of the two models scores higher becomes the official
"Classical ML Baseline" used in the final comparison against the CNN.

Run: python -m src.models.classical_ml.train_ml
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES_CSV = Path("data/deepweeds_features.csv")
RESULTS_TABLE = Path("results/tables/classical_ml_results.csv")
CHECKPOINT_DIR = Path("results/checkpoints/classical_ml")
RANDOM_SEED = 42       # same seed used everywhere else in this project — keeps folds comparable
CV_FOLDS = 5
SEARCH_ITERATIONS = 15  # how many random hyperparameter combos to try per model, not every combo

NON_FEATURE_COLS = {"filepath", "species", "split"}


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    # Test set excluded here on purpose — only train+val are used for
    # hyperparameter search. Test stays frozen until Phase 7.
    train_val_df = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    # Convert explicitly to plain numpy arrays. With newer pandas (3.x), text
    # columns are stored in a pyarrow-backed type, and `.values` hands back that
    # type instead of a numpy array. scikit-learn's cross-validation then can't
    # index it by fold, and crashes with "only integer scalar arrays can be
    # converted to a scalar index". to_numpy() avoids this on every pandas version.
    X = train_val_df[feature_cols].to_numpy(dtype=float)
    y = train_val_df["species"].astype(str).to_numpy(dtype=object)
    return X, y, feature_cols


def build_search_spaces():
    """
    Each model is wrapped in a Pipeline with StandardScaler first — this
    matters more for SVM (which is sensitive to feature scale) than Random
    Forest, but doing it for both keeps the two searches directly comparable.
    """
    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=RANDOM_SEED)),
    ])
    rf_param_space = {
        "clf__n_estimators": [100, 300, 500, 800],
        "clf__max_depth": [10, 20, 30, None],
        "clf__min_samples_split": [2, 5, 10],
    }

    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(random_state=RANDOM_SEED)),
    ])
    svm_param_space = {
        "clf__C": [0.1, 1, 10, 100],
        "clf__kernel": ["rbf", "linear"],
        "clf__gamma": ["scale", "auto"],
    }

    return {
        "random_forest": (rf_pipeline, rf_param_space),
        "svm": (svm_pipeline, svm_param_space),
    }


def run_search(name, pipeline, param_space, X, y, cv):
    print(f"\n{'=' * 60}\nSearching hyperparameters for {name}...\n{'=' * 60}")
    search = RandomizedSearchCV(
        pipeline,
        param_space,
        n_iter=SEARCH_ITERATIONS,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=1,
    )
    search.fit(X, y)
    print(f"Best params for {name}: {search.best_params_}")
    print(f"Best CV accuracy for {name}: {search.best_score_:.4f}")
    return search


def evaluate_with_cv(name, best_estimator, X, y, cv):
    """
    Get mean +/- std for accuracy, F1, precision, recall using the winning
    hyperparameters, via a fresh round of 5-fold CV. This is the format
    Section 4.3 requires you to report: mean and standard deviation across
    folds, not a single number.
    """
    scoring = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    results = cross_validate(best_estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    summary = {"model": name}
    for metric in scoring:
        key = f"test_{metric}"
        summary[f"{metric}_mean"] = float(np.mean(results[key]))
        summary[f"{metric}_std"] = float(np.std(results[key]))
    return summary


def main():
    if not FEATURES_CSV.exists():
        print(f"ERROR: {FEATURES_CSV} not found. "
              f"Run `python -m src.models.classical_ml.extract_features` first.")
        return

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLE.parent.mkdir(parents=True, exist_ok=True)

    X, y, feature_cols = load_data()
    print(f"Loaded {len(X)} samples (train+val only — test set stays frozen), "
          f"{len(feature_cols)} features.")
    print(f"Species: {sorted(set(y))}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    search_spaces = build_search_spaces()
    all_results = []

    for name, (pipeline, param_space) in search_spaces.items():
        search = run_search(name, pipeline, param_space, X, y, cv)

        # Re-run CV with the winning hyperparameters to get accuracy, F1,
        # precision, AND recall — the search above only optimized accuracy.
        summary = evaluate_with_cv(name, search.best_estimator_, X, y, cv)
        summary["best_params"] = str(search.best_params_)
        all_results.append(summary)

        checkpoint_path = CHECKPOINT_DIR / f"{name}.pkl"
        joblib.dump(search.best_estimator_, checkpoint_path)
        print(f"Saved {name} model to {checkpoint_path}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_TABLE, index=False)

    print(f"\nSaved results table to {RESULTS_TABLE}")
    print(results_df[["model", "accuracy_mean", "accuracy_std", "f1_macro_mean", "f1_macro_std"]])

    best_row = results_df.loc[results_df["accuracy_mean"].idxmax()]
    print(f"\nBest classical ML model: {best_row['model']} "
          f"(accuracy {best_row['accuracy_mean']:.4f} +/- {best_row['accuracy_std']:.4f})")
    print("This is your official Baseline 1 for Phase 7's comparison against the CNN.")


if __name__ == "__main__":
    main()

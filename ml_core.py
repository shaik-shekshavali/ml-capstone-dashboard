"""
Core machine-learning logic for the Week 4 capstone projects.

Supported projects:
  * heart - Heart Disease Prediction (classification)
  * loan  - Bank Loan Approval Prediction (classification)
  * mall  - Customer Segmentation with K-Means (clustering)

All logic lives in plain functions so it can be unit-tested without Streamlit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

PROJECTS = {
    "heart": {
        "title": "Heart Disease Prediction",
        "task": "classification",
        "dataset": "Kaggle - Heart Disease UCI",
        "target": "target",
        "target_map": None,
        "drop": ["num"],  # 'num' is an alternative label in some copies of the file
        "labels": {0: "No heart disease", 1: "Heart disease likely"},
    },
    "loan": {
        "title": "Bank Loan Approval Prediction",
        "task": "classification",
        "dataset": "Kaggle - Loan Prediction Dataset",
        "target": "Loan_Status",
        "target_map": {"Y": 1, "N": 0},
        "drop": ["Loan_ID"],
        "labels": {0: "Not approved", 1: "Approved"},
    },
    "mall": {
        "title": "Customer Segmentation (K-Means)",
        "task": "clustering",
        "dataset": "Kaggle - Mall Customers Dataset",
        "features": ["Age", "Annual Income (k$)", "Spending Score (1-100)"],
        "spending_col": "Spending Score (1-100)",
    },
}

CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    # Used in place of XGBoost so the project has no extra install step.
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "SVM": CalibratedClassifierCV(SVC(random_state=RANDOM_STATE), cv=3),
}

SVM_GRID = {
    "model__C": [0.1, 1, 10],
    "model__kernel": ["linear", "rbf"],
    "model__gamma": ["scale", 0.1],
}


@dataclass
class TrainResult:
    pipeline: Pipeline
    model_name: str
    metrics: dict
    cv_scores: np.ndarray
    confusion: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    report: str
    feature_names_in: list = field(default_factory=list)


# ---------------------------------------------------------------- data prep
def prepare_xy(df: pd.DataFrame, project: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return (features, target) for a classification project."""
    cfg = PROJECTS[project]
    data = df.copy()
    target = cfg["target"]

    if project == "heart" and target not in data.columns and "num" in data.columns:
        data[target] = (data["num"] > 0).astype(int)

    if target not in data.columns:
        raise ValueError(
            f"Target column '{target}' not found. Columns in your file: {list(df.columns)}"
        )

    data = data.dropna(subset=[target])
    if cfg["target_map"] and not pd.api.types.is_numeric_dtype(data[target]):
        data[target] = data[target].map(cfg["target_map"])
        data = data.dropna(subset=[target])

    y = data[target].astype(int)
    X = data.drop(columns=[target] + [c for c in cfg["drop"] if c in data.columns])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Median-impute and scale numbers; most-frequent-impute and one-hot encode text."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])


# ---------------------------------------------------------- classification
def train_classifier(
    df: pd.DataFrame,
    project: str,
    model_name: str,
    test_size: float = 0.2,
    cv_folds: int = 5,
) -> TrainResult:
    """Split, cross-validate on the training set, fit, and evaluate on the test set."""
    if PROJECTS[project]["task"] != "classification":
        raise ValueError(f"'{project}' is not a classification project")

    X, y = prepare_xy(df, project)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    pipe = Pipeline([
        ("prep", build_preprocessor(X)),
        ("model", clone(CLASSIFIERS[model_name])),
    ])

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    return TrainResult(
        pipeline=pipe,
        model_name=model_name,
        metrics=metrics,
        cv_scores=np.asarray(cv_scores),
        confusion=confusion_matrix(y_test, y_pred),
        fpr=fpr,
        tpr=tpr,
        report=classification_report(y_test, y_pred, zero_division=0),
        feature_names_in=list(X.columns),
    )


def tune_svm(df: pd.DataFrame, project: str, cv_folds: int = 5, test_size: float = 0.2):
    """GridSearchCV over an SVM pipeline. Tuning uses only the training split."""
    X, y = prepare_xy(df, project)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    pipe = Pipeline([("prep", build_preprocessor(X)), ("model", SVC())])
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipe, SVM_GRID, cv=cv, scoring="accuracy")
    search.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, search.predict(X_test))
    return search, test_acc


# ------------------------------------------------------ save / load / predict
def save_model(pipeline: Pipeline, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(path)


def predict_new(pipeline: Pipeline, new_df: pd.DataFrame, project: str) -> pd.DataFrame:
    """Predict on new rows. Drops the target and ID columns automatically if present."""
    cfg = PROJECTS[project]
    to_drop = [c for c in cfg.get("drop", []) + [cfg.get("target", "")]
               if c and c in new_df.columns]
    X_new = new_df.drop(columns=to_drop)

    out = new_df.copy()
    out["prediction"] = pipeline.predict(X_new)
    if "labels" in cfg:
        out["prediction_label"] = out["prediction"].map(cfg["labels"])
    out["probability"] = pipeline.predict_proba(X_new)[:, 1]
    return out


# ------------------------------------------------------------- clustering
def elbow_inertias(df: pd.DataFrame, features: list[str], max_k: int = 10):
    scaled = StandardScaler().fit_transform(df[features].dropna())
    ks = list(range(1, max_k + 1))
    inertias = [
        KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(scaled).inertia_
        for k in ks
    ]
    return ks, inertias


def segment_customers(df: pd.DataFrame, features: list[str], k: int = 3):
    """Cluster customers and name the segments by average spending score."""
    data = df[features].dropna()
    scaled = StandardScaler().fit_transform(data)
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(scaled)

    result = data.copy()
    result["cluster"] = km.labels_

    spend = PROJECTS["mall"]["spending_col"]
    spend = spend if spend in features else features[-1]
    order = result.groupby("cluster")[spend].mean().sort_values().index.tolist()

    names = (["Low Spenders", "Medium Spenders", "High Spenders"]
             if k == 3 else [f"Group {i + 1}" for i in range(k)])
    mapping = {cluster: names[i] for i, cluster in enumerate(order)}
    result["segment"] = result["cluster"].map(mapping)
    return result, km


# ------------------------------------------------------- synthetic samples
def make_sample(project: str, n: int = 200, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Synthetic data with the same columns as the Kaggle files, for demos and tests."""
    rng = np.random.default_rng(seed)

    if project == "heart":
        age = rng.integers(29, 78, n)
        sex = rng.integers(0, 2, n)
        cp = rng.integers(0, 4, n)
        trestbps = rng.normal(132, 17, n).round()
        chol = rng.normal(246, 51, n).round()
        thalach = rng.normal(150, 23, n).round()
        oldpeak = np.round(rng.exponential(1.0, n), 1)
        score = (0.05 * (age - 54) + 0.4 * (cp == 0) + 0.5 * oldpeak
                 - 0.02 * (thalach - 150) + 0.005 * (chol - 246)
                 + rng.normal(0, 0.8, n))
        target = (score > np.median(score)).astype(int)
        return pd.DataFrame({
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "thalach": thalach, "oldpeak": oldpeak, "target": target,
        })

    if project == "loan":
        income = rng.lognormal(8.6, 0.5, n).round()
        loan_amt = np.clip(income / 80 + rng.normal(0, 20, n), 10, 500).round()
        credit = rng.choice([1.0, 0.0], n, p=[0.85, 0.15])
        score = 2.0 * credit + 0.00003 * income - 0.005 * loan_amt + rng.normal(0, 0.5, n)
        df = pd.DataFrame({
            "Loan_ID": [f"LP{1000 + i}" for i in range(n)],
            "Gender": rng.choice(["Male", "Female"], n, p=[0.8, 0.2]),
            "Married": rng.choice(["Yes", "No"], n),
            "Education": rng.choice(["Graduate", "Not Graduate"], n, p=[0.8, 0.2]),
            "Self_Employed": rng.choice(["Yes", "No"], n, p=[0.15, 0.85]),
            "ApplicantIncome": income,
            "LoanAmount": loan_amt,
            "Credit_History": credit,
            "Property_Area": rng.choice(["Urban", "Semiurban", "Rural"], n),
            "Loan_Status": np.where(score > np.median(score), "Y", "N"),
        })
        # Real Kaggle data has missing values, so the sample does too.
        df.loc[rng.choice(n, max(1, n // 20), replace=False), "Gender"] = np.nan
        df.loc[rng.choice(n, max(1, n // 25), replace=False), "LoanAmount"] = np.nan
        return df

    if project == "mall":
        centers = {0: (45, 40, 20), 1: (32, 60, 50), 2: (26, 95, 82)}  # age, income, spend
        g = rng.integers(0, 3, n)
        age = np.array([rng.normal(centers[i][0], 5) for i in g]).clip(18, 70).round()
        income = np.array([rng.normal(centers[i][1], 8) for i in g]).clip(15, 140).round()
        spend = np.array([rng.normal(centers[i][2], 8) for i in g]).clip(1, 100).round()
        return pd.DataFrame({
            "CustomerID": np.arange(1, n + 1),
            "Genre": rng.choice(["Male", "Female"], n),
            "Age": age.astype(int),
            "Annual Income (k$)": income.astype(int),
            "Spending Score (1-100)": spend.astype(int),
        })

    raise ValueError(f"Unknown project: {project}")

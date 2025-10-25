"""
train_model.py: Complete ML training pipeline for AI Resume Screener

Includes:
- Data loading or synthetic data generation (1000 samples)
- Multiple algorithms: LogisticRegression, RandomForestClassifier, GradientBoostingClassifier
- Train/test split, hyperparameter tuning with GridSearchCV
- Cross-validation utilities
- Comprehensive evaluation metrics
- Model persistence with joblib
- ResumeScreenerModel class encapsulating the workflow
- train_pipeline() function to run end-to-end
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib


@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    scoring: str = "f1"
    tune_hyperparams: bool = True
    model_name: str = "logistic_regression"  # one of: 'logistic_regression', 'random_forest', 'gradient_boosting'
    target_col: str = "hired"


class ResumeScreenerModel:
    def __init__(self, config: Optional[TrainConfig] = None):
        self.config = config or TrainConfig()
        self.model: Optional[Any] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.feature_cols: Optional[List[str]] = None

    def load_training_data(self, path: str) -> pd.DataFrame:
        """Load training data from CSV or parquet. Must include target column.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Training data not found at {path}")
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        elif path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            raise ValueError("Unsupported file format. Use .csv or .parquet")
        if self.config.target_col not in df.columns:
            raise ValueError(f"Target column '{self.config.target_col}' missing from data")
        return df

    def create_synthetic_training_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Create a synthetic resume screening dataset with numeric and categorical features."""
        rng = np.random.default_rng(self.config.random_state)

        # Numeric features
        years_experience = rng.normal(loc=5, scale=2.5, size=n_samples).clip(0, None)
        num_skills = rng.integers(low=1, high=20, size=n_samples)
        education_level = rng.integers(low=0, high=3, size=n_samples)  # 0: HS, 1: Bachelor, 2: Master
        certifications = rng.integers(low=0, high=5, size=n_samples)
        projects = rng.integers(low=0, high=15, size=n_samples)
        previous_companies = rng.integers(low=0, high=10, size=n_samples)
        has_portfolio = rng.integers(low=0, high=2, size=n_samples)

        # Latent score indicating candidate quality
        latent = (
            0.35 * years_experience
            + 0.25 * num_skills
            + 0.5 * (education_level >= 1).astype(float)
            + 0.4 * certifications
            + 0.2 * projects
            + 0.1 * previous_companies
            + 0.3 * has_portfolio
        )
        # Convert to probability via logistic function
        prob = 1 / (1 + np.exp(-(latent - np.median(latent)) / (np.std(latent) + 1e-6)))
        hired = (rng.random(n_samples) < prob).astype(int)

        df = pd.DataFrame(
            {
                "years_experience": years_experience.round(2),
                "num_skills": num_skills,
                "education_level": education_level,
                "certifications": certifications,
                "projects": projects,
                "previous_companies": previous_companies,
                "has_portfolio": has_portfolio,
                "hired": hired,
            }
        )
        return df

    def _get_model_and_params(self, name: Optional[str] = None) -> Tuple[Any, Dict[str, List[Any]]]:
        name = (name or self.config.model_name).lower()
        if name == "logistic_regression":
            model = LogisticRegression(max_iter=200, n_jobs=None, solver="lbfgs")
            params = {
                "model__C": [0.1, 1.0, 10.0],
                "model__penalty": ["l2"],
            }
        elif name == "random_forest":
            model = RandomForestClassifier(random_state=self.config.random_state)
            params = {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_split": [2, 5],
            }
        elif name == "gradient_boosting":
            model = GradientBoostingClassifier(random_state=self.config.random_state)
            params = {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            }
        else:
            raise ValueError("Unsupported model_name. Choose from 'logistic_regression', 'random_forest', 'gradient_boosting'")
        return model, params

    def _build_pipeline(self, estimator: Any) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("model", estimator),
            ]
        )

    def train_model(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        tune_hyperparams: Optional[bool] = None,
        model_name: Optional[str] = None,
    ) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
        """Train model with optional hyperparameter tuning and return fitted model, best params, and eval metrics."""
        target = self.config.target_col
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' missing")

        X = df.drop(columns=[target])
        y = df[target]

        if features:
            missing = [c for c in features if c not in X.columns]
            if missing:
                raise ValueError(f"Requested features missing in data: {missing}")
            X = X[features]
        self.feature_cols = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state, stratify=y
        )

        estimator, param_grid = self._get_model_and_params(model_name)
        pipe = self._build_pipeline(estimator)

        do_tune = self.config.tune_hyperparams if tune_hyperparams is None else tune_hyperparams
        if do_tune:
            gscv = GridSearchCV(
                pipe,
                param_grid=param_grid,
                scoring=self.config.scoring,
                cv=self.config.cv_folds,
                n_jobs=-1,
                refit=True,
                verbose=0,
            )
            gscv.fit(X_train, y_train)
            self.model = gscv.best_estimator_
            self.best_params_ = gscv.best_params_
        else:
            self.model = pipe.fit(X_train, y_train)
            self.best_params_ = {}

        metrics = self.evaluate_model(X_test, y_test)
        return self.model, (self.best_params_ or {}), metrics

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model not trained")
        y_pred = self.model.predict(X_test)
        # Some models/pipelines support predict_proba
        try:
            y_proba = self.model.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_proba)
        except Exception:
            y_proba = None
            roc = float("nan")
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc,
        }
        # You can print or return the full report if desired
        _ = classification_report(y_test, y_pred, zero_division=0)
        _ = confusion_matrix(y_test, y_pred)
        return metrics

    def cross_validate(self, df: pd.DataFrame, model_name: Optional[str] = None) -> Dict[str, float]:
        target = self.config.target_col
        X = df.drop(columns=[target])
        y = df[target]
        estimator, _ = self._get_model_and_params(model_name)
        pipe = self._build_pipeline(estimator)
        scores = cross_val_score(pipe, X, y, cv=self.config.cv_folds, scoring=self.config.scoring, n_jobs=-1)
        return {"cv_mean": float(np.mean(scores)), "cv_std": float(np.std(scores))}

    def save_model(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model not trained")
        artifact = {
            "model": self.model,
            "feature_cols": self.feature_cols,
            "best_params": self.best_params_,
            "config": self.config.__dict__,
        }
        joblib.dump(artifact, path)

    def load_model(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        artifact = joblib.load(path)
        self.model = artifact.get("model")
        self.feature_cols = artifact.get("feature_cols")
        self.best_params_ = artifact.get("best_params")
        cfg = artifact.get("config")
        if cfg:
            self.config = TrainConfig(**cfg)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        # Align columns if feature list is stored
        if self.feature_cols:
            missing = [c for c in self.feature_cols if c not in X.columns]
            for c in missing:
                X[c] = 0
            X = X[self.feature_cols]
        return self.model.predict(X)


def train_pipeline(
    data_path: Optional[str] = None,
    output_model_path: str = "resume_screener_model.joblib",
    model_name: str = "logistic_regression",
    tune_hyperparams: bool = True,
) -> Dict[str, Any]:
    """Run the complete training workflow and persist the model.

    Returns a dictionary with metrics and metadata.
    """
    config = TrainConfig(model_name=model_name, tune_hyperparams=tune_hyperparams)
    trainer = ResumeScreenerModel(config)

    if data_path:
        df = trainer.load_training_data(data_path)
    else:
        df = trainer.create_synthetic_training_data(1000)

    model, best_params, metrics = trainer.train_model(df)
    cv_scores = trainer.cross_validate(df)

    trainer.save_model(output_model_path)

    result = {
        "model_name": model_name,
        "best_params": best_params,
        "metrics": metrics,
        "cv_scores": cv_scores,
        "model_path": output_model_path,
        "feature_cols": trainer.feature_cols,
    }
    # Optionally write a JSON report next to the model
    try:
        report_path = os.path.splitext(output_model_path)[0] + "_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    except Exception:
        pass

    return result


if __name__ == "__main__":
    # Example usage: train on synthetic data and save model
    summary = train_pipeline(
        data_path=None,
        output_model_path="resume_screener_model.joblib",
        model_name="random_forest",
        tune_hyperparams=True,
    )
    print("Training complete:\n", json.dumps(summary, indent=2))

"""Scikit-learn patient risk prediction for IodineIQ."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clinical_rag.config import get_settings
from clinical_rag.logging_setup import get_logger
from clinical_rag.models import RiskPrediction, RiskTier

logger = get_logger(__name__)

FEATURE_COLUMNS = [
    "age",
    "heart_rate_mean",
    "spo2_min",
    "sys_bp_mean",
    "temp_max",
    "resp_rate_mean",
    "num_conditions",
    "los_days",
    "prior_admissions",
    "lab_abnormal_count",
]


def _tier(score: float) -> RiskTier:
    if score >= 0.85:
        return RiskTier.CRITICAL
    if score >= 0.65:
        return RiskTier.HIGH
    if score >= 0.35:
        return RiskTier.MODERATE
    return RiskTier.LOW


def _interventions(tier: RiskTier, features: dict[str, float]) -> list[str]:
    hints: list[str] = []
    if tier in {RiskTier.HIGH, RiskTier.CRITICAL}:
        hints.append("Prioritize rapid response / higher-acuity monitoring")
    if features.get("spo2_min", 100) < 92:
        hints.append("Evaluate hypoxia; consider oxygen titration protocol")
    if features.get("heart_rate_mean", 0) > 110:
        hints.append("Assess tachycardia etiology and fluid status")
    if features.get("num_conditions", 0) >= 5:
        hints.append("Multimorbidity: coordinate CDI query + care management")
    if not hints:
        hints.append("Continue standard pathway; reassess on new vitals")
    return hints


def synthesize_training_frame(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic labeled EHR feature rows for bootstrap training."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 95, n),
            "heart_rate_mean": rng.normal(82, 18, n).clip(40, 180),
            "spo2_min": rng.normal(95, 4, n).clip(70, 100),
            "sys_bp_mean": rng.normal(122, 20, n).clip(70, 220),
            "temp_max": rng.normal(37.2, 0.8, n).clip(35, 41),
            "resp_rate_mean": rng.normal(18, 4, n).clip(8, 40),
            "num_conditions": rng.integers(0, 12, n),
            "los_days": rng.integers(1, 21, n),
            "prior_admissions": rng.integers(0, 8, n),
            "lab_abnormal_count": rng.integers(0, 15, n),
        }
    )
    # Label with a clinically inspired linear risk score + noise
    logits = (
        0.02 * (df["age"] - 50)
        + 0.03 * (df["heart_rate_mean"] - 80)
        + -0.12 * (df["spo2_min"] - 95)
        + 0.08 * (df["temp_max"] - 37)
        + 0.05 * (df["resp_rate_mean"] - 16)
        + 0.15 * df["num_conditions"]
        + 0.1 * df["prior_admissions"]
        + 0.08 * df["lab_abnormal_count"]
        + rng.normal(0, 1.0, n)
    )
    probs = 1 / (1 + np.exp(-logits / 3))
    df["deterioration"] = (probs > 0.5).astype(int)
    return df


def train_and_persist(path: str | None = None, n: int = 800) -> dict[str, Any]:
    settings = get_settings()
    path = path or settings.risk_model_path
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    df = synthesize_training_frame(n=n)
    x = df[FEATURE_COLUMNS]
    y = df["deterioration"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=120,
                    max_depth=3,
                    learning_rate=0.08,
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(x_train, y_train)
    acc = float(pipe.score(x_test, y_test))
    joblib.dump({"pipeline": pipe, "features": FEATURE_COLUMNS}, path)
    logger.info("risk_model_trained", path=path, accuracy=acc, n=n)
    return {"path": path, "accuracy": acc, "n": n}


class PatientRiskModel:
    def __init__(self, model_path: str | None = None) -> None:
        settings = get_settings()
        self.model_path = model_path or settings.risk_model_path
        self._bundle: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._bundle is None:
            path = Path(self.model_path)
            if not path.exists():
                train_and_persist(str(path))
            self._bundle = joblib.load(path)
        return self._bundle

    def extract_features(self, resources: list[dict[str, Any]], age: float = 65.0) -> dict[str, float]:
        vitals: dict[str, list[float]] = {}
        conditions = 0
        for r in resources:
            rtype = r.get("resourceType")
            if rtype == "Observation":
                metric = (
                    (r.get("code") or {}).get("text")
                    or (((r.get("code") or {}).get("coding") or [{}])[0].get("display"))
                    or "unknown"
                )
                val = ((r.get("valueQuantity") or {}).get("value"))
                if val is not None:
                    vitals.setdefault(str(metric).lower(), []).append(float(val))
            elif rtype == "Condition":
                conditions += 1

        def mean_of(*keys: str, default: float) -> float:
            vals: list[float] = []
            for k in keys:
                vals.extend(vitals.get(k, []))
            return float(np.mean(vals)) if vals else default

        def min_of(*keys: str, default: float) -> float:
            vals: list[float] = []
            for k in keys:
                vals.extend(vitals.get(k, []))
            return float(np.min(vals)) if vals else default

        def max_of(*keys: str, default: float) -> float:
            vals: list[float] = []
            for k in keys:
                vals.extend(vitals.get(k, []))
            return float(np.max(vals)) if vals else default

        return {
            "age": float(age),
            "heart_rate_mean": mean_of("heart_rate", default=80.0),
            "spo2_min": min_of("spo2", default=97.0),
            "sys_bp_mean": mean_of("blood_pressure_systolic", default=120.0),
            "temp_max": max_of("temperature", default=37.0),
            "resp_rate_mean": mean_of("respiratory_rate", default=16.0),
            "num_conditions": float(conditions),
            "los_days": 3.0,
            "prior_admissions": 1.0,
            "lab_abnormal_count": float(max(conditions - 1, 0)),
        }

    def predict(self, patient_id: str, features: dict[str, float]) -> RiskPrediction:
        bundle = self._load()
        pipe: Pipeline = bundle["pipeline"]
        cols = bundle["features"]
        row = pd.DataFrame([{c: float(features.get(c, 0.0)) for c in cols}])
        proba = float(pipe.predict_proba(row)[0][1])
        tier = _tier(proba)
        return RiskPrediction(
            patient_id=patient_id,
            risk_score=round(proba, 4),
            risk_tier=tier,
            features=features,
            intervention_hints=_interventions(tier, features),
        )

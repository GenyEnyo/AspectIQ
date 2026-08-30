from __future__ import annotations

from pathlib import Path
import re

import gensim.downloader as api
import joblib
import numpy as np

from src.contrast_aware_features import build_target_clause_local_feature
from src.local_context_embeddings import GLOVE_MODEL_NAME


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
FINAL_MODEL_FILE = MODEL_DIR / "best_target_clause_local_3class.joblib"
CONTEXT_WINDOW = 5


def load_prediction_assets():
    if not FINAL_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Final model not found: {FINAL_MODEL_FILE}\n"
            "Run `python src/train_production_target_clause_local.py` first."
        )

    model = joblib.load(FINAL_MODEL_FILE)
    glove_model = api.load(GLOVE_MODEL_NAME)

    scaler = (
        model.named_steps.get("scaler")
        if hasattr(model, "named_steps")
        else None
    )
    observed = getattr(scaler, "n_features_in_", None)

    if observed is not None and int(observed) != 200:
        raise ValueError(
            "Production model/feature mismatch: "
            f"model expects {observed} features, "
            "but AspectIQ uses 200."
        )

    return model, glove_model


def find_aspect_span(text: str, aspect: str):
    text = str(text)
    aspect = str(aspect).strip()

    if not aspect:
        return False, None, None

    pattern = r"(?<!\w)" + re.escape(aspect) + r"(?!\w)"
    match = re.search(pattern, text, flags=re.IGNORECASE)

    if match is None:
        tokens = [
            re.escape(token)
            for token in aspect.split()
            if token
        ]
        if not tokens:
            return False, None, None

        pattern = (
            r"(?<!\w)"
            + r"\s+".join(tokens)
            + r"(?!\w)"
        )
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

    if match is None:
        return False, None, None

    return True, match.start(), match.end()


def build_prediction_feature(text, aspect, glove_model):
    found, start, end = find_aspect_span(
        text=text,
        aspect=aspect,
    )

    if not found:
        raise ValueError(
            f"Aspect '{aspect}' was not found in the review text."
        )

    feature, metadata, _ = build_target_clause_local_feature(
        text=text,
        aspect=aspect,
        glove_model=glove_model,
        context_window=CONTEXT_WINDOW,
    )

    feature = feature.reshape(1, -1)

    metadata = {
        "original_text": str(text),
        "aspect": str(aspect),
        "aspect_start": int(start),
        "aspect_end": int(end),
        **metadata,
    }

    return feature, metadata


def _softmax(values):
    values = np.asarray(values, dtype=float)
    values = values - np.max(values)
    exp_values = np.exp(values)
    total = exp_values.sum()

    if total == 0:
        return np.ones_like(exp_values) / len(exp_values)

    return exp_values / total


def predict_aspect(
    text,
    aspect,
    model,
    glove_model,
):
    feature, metadata = build_prediction_feature(
        text=text,
        aspect=aspect,
        glove_model=glove_model,
    )

    prediction = model.predict(feature)[0]
    classifier = model.named_steps["classifier"]
    classes = classifier.classes_

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(feature)[0]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(feature)

        if np.asarray(decision).ndim == 1:
            decision = np.column_stack(
                [-decision, decision]
            )

        scores = _softmax(np.asarray(decision)[0])
    else:
        scores = np.zeros(len(classes), dtype=float)
        index = list(classes).index(prediction)
        scores[index] = 1.0

    score_dict = {
        str(class_name): float(scores[index])
        for index, class_name in enumerate(classes)
    }

    confidence = max(score_dict.values())

    return {
        "aspect": str(aspect),
        "sentiment": str(prediction),
        "confidence": float(confidence),
        # Retained for existing app compatibility. For this SVM these
        # are normalized decision scores, not calibrated probabilities.
        "probabilities": score_dict,
        "model_scores": score_dict,
        "scores_are_calibrated": False,
        **metadata,
    }


def predict_multiple_aspects(
    text,
    aspects,
    model,
    glove_model,
):
    results = []

    for aspect_item in aspects:
        if isinstance(aspect_item, dict):
            aspect = str(
                aspect_item.get("aspect", "")
            ).strip()
            detector_source = aspect_item.get(
                "source",
                "automatic",
            )
        else:
            aspect = str(aspect_item).strip()
            detector_source = "automatic"

        if not aspect:
            continue

        found, _, _ = find_aspect_span(
            text=text,
            aspect=aspect,
        )

        if not found:
            continue

        result = predict_aspect(
            text=text,
            aspect=aspect,
            model=model,
            glove_model=glove_model,
        )

        result["aspect_detection_source"] = detector_source
        results.append(result)

    return results

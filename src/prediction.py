from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
import re

import gensim.downloader as api
from gensim.models import KeyedVectors
import joblib
import numpy as np
import pandas as pd

from src.preprocessing import preprocess_text
from src.local_context_embeddings import (
    GLOVE_MODEL_NAME,
    extract_local_context,
    mean_embedding,
)


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
FINAL_MODEL_FILE = MODEL_DIR / "best_3_class_model.joblib"


REVIEW_TEXT_KEYS = (
    "text",
    "review_text",
    "review",
    "original_text",
    "raw_text",
    "content",
    "review_content",
    "comment",
    "sentence",
    "cleaned_text",
    "processed_text",
)

ASPECT_TEXT_KEYS = (
    "aspect",
    "aspect_term",
    "term",
    "name",
    "text",
)

ASPECT_COLLECTION_KEYS = (
    "aspects",
    "detected_aspects",
    "aspect_terms",
    "detected_terms",
    "items",
    "results",
)

PROCESSED_TEXT_KEYS = (
    "processed_text",
    "cleaned_text",
    "lemmatized_text",
    "lemmatised_text",
    "normalized_text",
    "normalised_text",
    "text",
)

TOKEN_KEYS = (
    "lemmatized_tokens",
    "lemmatised_tokens",
    "filtered_tokens",
    "processed_tokens",
    "tokens",
)


_MISSING = object()


def _normalise_key(value: Any) -> str:
    """Normalise dictionary keys for case-insensitive alias matching."""
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).casefold(),
    ).strip("_")


def _mapping_value(
    mapping: Mapping,
    candidate_keys: Sequence[str],
):
    """Return the first matching value from a mapping."""
    for key in candidate_keys:
        if key in mapping:
            return mapping[key]

    normalised_mapping = {
        _normalise_key(key): value
        for key, value in mapping.items()
    }

    for key in candidate_keys:
        normalised_key = _normalise_key(key)

        if normalised_key in normalised_mapping:
            return normalised_mapping[normalised_key]

    return _MISSING


def _normalise_text_value(
    value,
    *,
    field_name: str,
    candidate_keys: Sequence[str],
    allow_empty: bool = False,
    _depth: int = 0,
) -> str:
    """Convert a valid text value or structured text record to a string.

    Dictionaries are never converted with ``str(dictionary)`` because doing so
    would feed record syntax and metadata into the NLP model.
    """
    if isinstance(value, np.str_):
        value = str(value)

    if isinstance(value, str):
        text = value.strip()

        if text or allow_empty:
            return text

        raise ValueError(f"{field_name} is empty.")

    if value is None:
        if allow_empty:
            return ""

        raise TypeError(f"{field_name} cannot be None.")

    if isinstance(value, pd.Series):
        value = value.to_dict()

    if isinstance(value, Mapping):
        if _depth >= 3:
            raise TypeError(
                f"{field_name} contains too many nested dictionaries."
            )

        candidate = _mapping_value(value, candidate_keys)

        if candidate is _MISSING:
            if allow_empty:
                return ""

            raise TypeError(
                f"{field_name} dictionary does not contain a supported "
                f"text field. Available keys: {list(value.keys())}"
            )

        return _normalise_text_value(
            candidate,
            field_name=field_name,
            candidate_keys=candidate_keys,
            allow_empty=allow_empty,
            _depth=_depth + 1,
        )

    raise TypeError(
        f"{field_name} must be a string or a dictionary containing text; "
        f"received {type(value).__name__}: {value!r}"
    )


def extract_review_text(value) -> str:
    """Return clean review text from a string or structured review record."""
    return _normalise_text_value(
        value,
        field_name="Review text",
        candidate_keys=REVIEW_TEXT_KEYS,
    )


def _normalise_aspect_text(
    value,
    *,
    allow_empty: bool = False,
) -> str:
    return _normalise_text_value(
        value,
        field_name="Aspect",
        candidate_keys=ASPECT_TEXT_KEYS,
        allow_empty=allow_empty,
    )


def _join_tokens(value, field_name: str) -> str:
    """Join a token collection without stringifying nested structures."""
    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, pd.Series):
        value = value.tolist()

    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise TypeError(
            f"{field_name} tokens must be a sequence of strings; "
            f"received {type(value).__name__}."
        )

    if not all(isinstance(token, (str, np.str_)) for token in value):
        raise TypeError(
            f"{field_name} tokens contain a non-string value: {value!r}"
        )

    return " ".join(
        str(token).strip()
        for token in value
        if str(token).strip()
    )


def _normalise_preprocessed_output(
    value,
    *,
    field_name: str,
) -> str:
    """Normalise supported outputs returned by ``preprocess_text``.

    The preprocessing function may return a string, a dictionary containing a
    cleaned-text field, or a list of tokens. Downstream embedding functions
    always receive a plain string.
    """
    if isinstance(value, np.str_):
        return str(value).strip()

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, pd.Series):
        value = value.to_dict()

    if isinstance(value, Mapping):
        candidate = _mapping_value(value, PROCESSED_TEXT_KEYS)

        if candidate is not _MISSING:
            return _normalise_preprocessed_output(
                candidate,
                field_name=field_name,
            )

        tokens = _mapping_value(value, TOKEN_KEYS)

        if tokens is not _MISSING:
            return _join_tokens(tokens, field_name)

        raise TypeError(
            f"{field_name} preprocessing returned a dictionary without a "
            f"supported text or token field. Available keys: "
            f"{list(value.keys())}"
        )

    if isinstance(value, np.ndarray):
        return _join_tokens(value, field_name)

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        if all(isinstance(item, (str, np.str_)) for item in value):
            return _join_tokens(value, field_name)

        # Some preprocessors return (processed_text, metadata).
        for item in value:
            if isinstance(item, (str, np.str_)):
                return str(item).strip()

        raise TypeError(
            f"{field_name} preprocessing returned an unsupported sequence: "
            f"{value!r}"
        )

    raise TypeError(
        f"{field_name} preprocessing must return text, a text dictionary, "
        f"or tokens; received {type(value).__name__}: {value!r}"
    )


def _load_glove_model():
    """Load GloVe with a fallback for a damaged Gensim reader module."""
    try:
        return api.load(GLOVE_MODEL_NAME)

    except AttributeError as exc:
        if "load_data" not in str(exc):
            raise

        try:
            glove_path = api.load(
                GLOVE_MODEL_NAME,
                return_path=True,
            )

            return KeyedVectors.load_word2vec_format(
                glove_path,
                binary=False,
            )

        except Exception as fallback_exc:
            raise RuntimeError(
                "GloVe was downloaded, but Gensim could not load its reader "
                "module or vector file. Rename the cached model directory and "
                "download it again."
            ) from fallback_exc


def load_prediction_assets():
    """Load the trained classifier and pretrained GloVe vectors."""
    if not FINAL_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Final model not found: {FINAL_MODEL_FILE}\n"
            "Run the controlled experiments first."
        )

    model = joblib.load(FINAL_MODEL_FILE)

    if not callable(getattr(model, "predict", None)):
        raise TypeError(
            "The saved prediction model does not provide a predict() method."
        )

    glove_model = _load_glove_model()
    return model, glove_model


def find_aspect_span(text, aspect):
    """Locate an aspect term in review text using case-insensitive matching."""
    review_text = extract_review_text(text)
    aspect_text = _normalise_aspect_text(
        aspect,
        allow_empty=True,
    )

    if not aspect_text:
        return False, None, None

    pattern = (
        r"(?<!\w)"
        + re.escape(aspect_text)
        + r"(?!\w)"
    )

    match = re.search(
        pattern,
        review_text,
        flags=re.IGNORECASE,
    )

    if match is None:
        tokens = [
            re.escape(token)
            for token in aspect_text.split()
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
            review_text,
            flags=re.IGNORECASE,
        )

    if match is None:
        return False, None, None

    return True, match.start(), match.end()


def _as_embedding_vector(value, vector_name: str) -> np.ndarray:
    """Validate and flatten one embedding vector."""
    vector = np.asarray(value, dtype=np.float32).reshape(-1)

    if vector.size == 0:
        raise ValueError(f"{vector_name} embedding is empty.")

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            f"{vector_name} embedding contains NaN or infinite values."
        )

    return vector


def build_prediction_feature(
    text,
    aspect,
    glove_model,
):
    """Create sentence, aspect and local-context embedding features."""
    review_text = extract_review_text(text)
    aspect_text = _normalise_aspect_text(aspect)

    found, start, end = find_aspect_span(
        text=review_text,
        aspect=aspect_text,
    )

    if not found:
        raise ValueError(
            f"Aspect '{aspect_text}' was not found in the review text."
        )

    processed_text = _normalise_preprocessed_output(
        preprocess_text(review_text),
        field_name="Review text",
    )

    processed_aspect = _normalise_preprocessed_output(
        preprocess_text(aspect_text),
        field_name="Aspect",
    )

    row = pd.Series(
        {
            "text": review_text,
            "aspect": aspect_text,
            "from": start,
            "to": end,
        }
    )

    local_context, context_source = extract_local_context(row)

    local_context = _normalise_text_value(
        local_context,
        field_name="Local context",
        candidate_keys=REVIEW_TEXT_KEYS,
        allow_empty=True,
    )

    processed_local_context = _normalise_preprocessed_output(
        preprocess_text(local_context),
        field_name="Local context",
    )

    sentence_vector = _as_embedding_vector(
        mean_embedding(processed_text, glove_model),
        "Sentence",
    )

    aspect_vector = _as_embedding_vector(
        mean_embedding(processed_aspect, glove_model),
        "Aspect",
    )

    local_vector = _as_embedding_vector(
        mean_embedding(processed_local_context, glove_model),
        "Local context",
    )

    vector_dimensions = {
        sentence_vector.size,
        aspect_vector.size,
        local_vector.size,
    }

    if len(vector_dimensions) != 1:
        raise ValueError(
            "Sentence, aspect and local-context embeddings have different "
            "dimensions."
        )

    feature = np.concatenate(
        [
            sentence_vector,
            aspect_vector,
            local_vector,
        ]
    ).astype(np.float32)

    feature = feature.reshape(1, -1)

    metadata = {
        "original_text": review_text,
        "aspect": aspect_text,
        "processed_text": processed_text,
        "processed_aspect": processed_aspect,
        "local_context": local_context,
        "processed_local_context": processed_local_context,
        "context_source": str(context_source),
        "feature_dimensions": int(feature.shape[1]),
        "aspect_start": int(start),
        "aspect_end": int(end),
    }

    return feature, metadata


def _softmax(values):
    """Return numerically stable softmax scores."""
    values = np.asarray(values, dtype=float).reshape(-1)

    if values.size == 0:
        raise ValueError("Cannot calculate softmax for an empty array.")

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Cannot calculate softmax for NaN or infinite values."
        )

    values = values - np.max(values)
    exp_values = np.exp(values)
    total = exp_values.sum()

    if not np.isfinite(total) or total <= 0:
        return np.ones_like(exp_values) / len(exp_values)

    return exp_values / total


def _model_classes(model) -> np.ndarray:
    """Return classifier classes from a pipeline or standalone estimator."""
    classes = getattr(model, "classes_", None)

    if classes is None:
        named_steps = getattr(model, "named_steps", {})
        classifier = named_steps.get("classifier")

        if classifier is None and named_steps:
            classifier = list(named_steps.values())[-1]

        classes = getattr(classifier, "classes_", None)

    if classes is None:
        raise AttributeError(
            "The prediction model does not expose classifier classes_."
        )

    classes = np.asarray(classes)

    if classes.size == 0:
        raise ValueError("The prediction model has no classifier classes.")

    return classes


def _prediction_scores(
    model,
    feature: np.ndarray,
    classes: np.ndarray,
    prediction,
) -> np.ndarray:
    """Return class scores using the best interface exposed by the model."""
    if callable(getattr(model, "predict_proba", None)):
        scores = np.asarray(
            model.predict_proba(feature)[0],
            dtype=float,
        )

    elif callable(getattr(model, "decision_function", None)):
        decision = np.asarray(
            model.decision_function(feature),
            dtype=float,
        )

        if decision.ndim == 0:
            decision = decision.reshape(1)

        if decision.ndim == 1:
            if len(classes) == 2 and decision.size == 1:
                scores = _softmax(
                    [0.0, float(decision[0])]
                )
            else:
                scores = _softmax(decision)
        else:
            scores = _softmax(decision[0])

    else:
        scores = np.zeros(len(classes), dtype=float)
        matching_indexes = np.flatnonzero(classes == prediction)

        if matching_indexes.size == 0:
            raise ValueError(
                f"Predicted class {prediction!r} is absent from classes_."
            )

        scores[int(matching_indexes[0])] = 1.0

    scores = np.asarray(scores, dtype=float).reshape(-1)

    if scores.size != len(classes):
        raise ValueError(
            f"The model returned {scores.size} class scores for "
            f"{len(classes)} classes."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError(
            "The model returned NaN or infinite class scores."
        )

    return scores


def predict_aspect(
    text,
    aspect,
    model,
    glove_model,
):
    """Predict sentiment for one aspect found in a review."""
    feature, metadata = build_prediction_feature(
        text=text,
        aspect=aspect,
        glove_model=glove_model,
    )

    expected_features = getattr(model, "n_features_in_", None)

    if (
        expected_features is not None
        and feature.shape[1] != int(expected_features)
    ):
        raise ValueError(
            f"Prediction feature has {feature.shape[1]} dimensions, but the "
            f"trained model expects {int(expected_features)}."
        )

    prediction = model.predict(feature)[0]
    classes = _model_classes(model)

    scores = _prediction_scores(
        model=model,
        feature=feature,
        classes=classes,
        prediction=prediction,
    )

    probability_dict = {
        str(class_name): float(scores[index])
        for index, class_name in enumerate(classes)
    }

    confidence = max(probability_dict.values())

    return {
        "aspect": metadata["aspect"],
        "sentiment": str(prediction),
        "confidence": float(confidence),
        "probabilities": probability_dict,
        **metadata,
    }


def _aspect_items(aspects) -> Iterable:
    """Normalise a single aspect, a list, or a detector result wrapper."""
    if aspects is None:
        return ()

    if isinstance(aspects, pd.Series):
        return aspects.tolist()

    if isinstance(aspects, Mapping):
        nested_aspects = _mapping_value(
            aspects,
            ASPECT_COLLECTION_KEYS,
        )

        if nested_aspects is _MISSING:
            return (aspects,)

        if isinstance(nested_aspects, (str, Mapping)):
            return (nested_aspects,)

        if isinstance(nested_aspects, Iterable):
            return nested_aspects

        raise TypeError(
            "The detected-aspects field must contain one aspect or an "
            "iterable of aspects."
        )

    if isinstance(aspects, str):
        return (aspects,)

    if isinstance(aspects, Iterable):
        return aspects

    raise TypeError(
        "Aspects must be a string, dictionary, or iterable of aspects; "
        f"received {type(aspects).__name__}."
    )


def _normalise_aspect_item(aspect_item):
    """Return ``(aspect, source)`` from a string or detector dictionary."""
    if isinstance(aspect_item, pd.Series):
        aspect_item = aspect_item.to_dict()

    if isinstance(aspect_item, Mapping):
        raw_aspect = _mapping_value(
            aspect_item,
            ASPECT_TEXT_KEYS,
        )

        if raw_aspect is _MISSING:
            return "", "automatic"

        detector_source = _mapping_value(
            aspect_item,
            ("source", "detector_source", "detection_source"),
        )

        if detector_source is _MISSING:
            detector_source = "automatic"

    else:
        raw_aspect = aspect_item
        detector_source = "automatic"

    try:
        aspect = _normalise_aspect_text(
            raw_aspect,
            allow_empty=True,
        )
    except (TypeError, ValueError):
        return "", "automatic"

    if not isinstance(detector_source, (str, np.str_)):
        detector_source = "automatic"
    else:
        detector_source = str(detector_source).strip() or "automatic"

    return aspect, detector_source


def predict_multiple_aspects(
    text,
    aspects,
    model,
    glove_model,
):
    """Predict sentiment for all valid, detected aspects in one review."""
    review_text = extract_review_text(text)
    results = []
    seen_aspects = set()

    for aspect_item in _aspect_items(aspects):
        aspect, detector_source = _normalise_aspect_item(
            aspect_item
        )

        if not aspect:
            continue

        aspect_key = aspect.casefold()

        if aspect_key in seen_aspects:
            continue

        found, _, _ = find_aspect_span(
            text=review_text,
            aspect=aspect,
        )

        if not found:
            continue

        result = predict_aspect(
            text=review_text,
            aspect=aspect,
            model=model,
            glove_model=glove_model,
        )

        result["aspect_detection_source"] = detector_source
        results.append(result)
        seen_aspects.add(aspect_key)

    return results


__all__ = [
    "FINAL_MODEL_FILE",
    "build_prediction_feature",
    "extract_review_text",
    "find_aspect_span",
    "load_prediction_assets",
    "predict_aspect",
    "predict_multiple_aspects",
]

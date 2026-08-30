from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


IMPORTANT_WORDS = {
    "no",
    "not",
    "nor",
    "never",
    "neither",
    "but",
    "however",
    "although",
    "though",
    "yet",
    "despite",
    "very",
    "too",
    "extremely",
    "barely",
    "hardly",
    "only",
}

CONTRAST_MARKERS = {
    "but",
    "however",
    "although",
    "though",
    "yet",
    "whereas",
    "while",
    "nevertheless",
    "nonetheless",
}

STOP_WORDS = set(ENGLISH_STOP_WORDS) - IMPORTANT_WORDS

TOKEN_PATTERN = re.compile(
    r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?"
)
NON_WORD_PATTERN = re.compile(
    r"[^a-z0-9'\.\s]+"
)


def expand_negations(text: str) -> str:
    text = str(text).replace("’", "'")

    replacements = {
        "won't": "will not",
        "can't": "can not",
        "cannot": "can not",
        "shan't": "shall not",
        "ain't": "is not",
    }

    for contraction, expanded in replacements.items():
        text = re.sub(
            rf"\b{re.escape(contraction)}\b",
            expanded,
            text,
            flags=re.IGNORECASE,
        )

    return re.sub(
        r"n['’]t\b",
        " not",
        text,
        flags=re.IGNORECASE,
    )


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize(
        "NFKC",
        str(text),
    ).casefold()

    normalized = expand_negations(
        normalized
    )

    normalized = NON_WORD_PATTERN.sub(
        " ",
        normalized,
    )

    return re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(
        normalize_text(text)
    )


def embedding_tokens(
    tokens: Iterable[str],
) -> list[str]:
    return [
        token
        for token in tokens
        if token not in STOP_WORDS
    ]


def find_token_span(
    tokens: Sequence[str],
    aspect_tokens: Sequence[str],
):
    if (
        not tokens
        or not aspect_tokens
        or len(aspect_tokens) > len(tokens)
    ):
        return None

    width = len(aspect_tokens)

    for start in range(
        len(tokens) - width + 1
    ):
        if (
            list(
                tokens[
                    start:
                    start + width
                ]
            )
            == list(aspect_tokens)
        ):
            return (
                start,
                start + width,
            )

    return None


def distance_from_span(
    position: int,
    start: int,
    end: int,
) -> int:
    if start <= position < end:
        return 0

    if position < start:
        return start - position

    return position - end + 1


def contrast_clause_bounds(
    tokens: Sequence[str],
    aspect_start: int,
    aspect_end: int,
):
    """
    Return the contrast-delimited clause containing the target aspect.

    Example:
        The screen is beautiful but the battery life is disappointing.

    screen       -> the screen is beautiful
    battery life -> the battery life is disappointing
    """
    left_boundary = 0
    right_boundary = len(tokens)

    marker_positions = [
        index
        for index, token in enumerate(tokens)
        if token in CONTRAST_MARKERS
    ]

    for position in marker_positions:
        if position < aspect_start:
            left_boundary = (
                position + 1
            )
            continue

        if position >= aspect_end:
            right_boundary = position
            break

    if not (
        0
        <= left_boundary
        <= aspect_start
        and aspect_end
        <= right_boundary
        <= len(tokens)
    ):
        return (
            0,
            len(tokens),
        )

    return (
        left_boundary,
        right_boundary,
    )


def weighted_glove_average(
    tokens: Sequence[str],
    positions: Sequence[int],
    glove_model,
    weights: Sequence[float] | None = None,
    remove_stopwords: bool = True,
):
    if len(tokens) != len(positions):
        raise ValueError(
            "Token and position sequences "
            "must have equal lengths."
        )

    if (
        weights is not None
        and len(tokens) != len(weights)
    ):
        raise ValueError(
            "Token and weight sequences "
            "must have equal lengths."
        )

    vectors = []
    retained_weights = []
    considered = 0

    for offset, token in enumerate(tokens):
        if (
            remove_stopwords
            and token in STOP_WORDS
        ):
            continue

        considered += 1

        if token not in glove_model:
            continue

        vectors.append(
            np.asarray(
                glove_model[token],
                dtype=np.float32,
            )
        )

        retained_weights.append(
            1.0
            if weights is None
            else float(weights[offset])
        )

    if not vectors:
        return (
            np.zeros(
                glove_model.vector_size,
                dtype=np.float32,
            ),
            0,
            considered,
        )

    matrix = np.vstack(vectors)

    weight_array = np.asarray(
        retained_weights,
        dtype=np.float32,
    )

    if float(weight_array.sum()) <= 0:
        weight_array = np.ones(
            len(vectors),
            dtype=np.float32,
        )

    average = np.average(
        matrix,
        axis=0,
        weights=weight_array,
    )

    return (
        average.astype(
            np.float32
        ),
        len(vectors),
        considered,
    )


def build_contrast_aware_feature(
    text: str,
    aspect: str,
    glove_model,
    context_window: int = 5,
):
    """
    Build the selected 300-dimensional representation:

        100d full sentence GloVe
      + 100d target aspect GloVe
      + 100d distance-weighted, contrast-aware local context GloVe
    """
    if context_window < 1:
        raise ValueError(
            "context_window must be at least 1."
        )

    all_tokens = tokenize(text)
    aspect_tokens = tokenize(aspect)

    aspect_span = find_token_span(
        all_tokens,
        aspect_tokens,
    )

    sentence_positions = list(
        range(len(all_tokens))
    )

    sentence_vector, text_known, text_total = (
        weighted_glove_average(
            all_tokens,
            sentence_positions,
            glove_model,
        )
    )

    aspect_positions = list(
        range(len(aspect_tokens))
    )

    aspect_vector, aspect_known, aspect_total = (
        weighted_glove_average(
            aspect_tokens,
            aspect_positions,
            glove_model,
            remove_stopwords=False,
        )
    )

    if aspect_span is None:
        local_positions = sentence_positions
        local_tokens = all_tokens
        local_weights = [
            1.0
        ] * len(local_tokens)

        clause_left = 0
        clause_right = len(all_tokens)

    else:
        aspect_start, aspect_end = (
            aspect_span
        )

        clause_left, clause_right = (
            contrast_clause_bounds(
                all_tokens,
                aspect_start,
                aspect_end,
            )
        )

        left = max(
            clause_left,
            aspect_start - context_window,
        )

        right = min(
            clause_right,
            aspect_end + context_window,
        )

        local_positions = list(
            range(
                left,
                right,
            )
        )

        local_tokens = [
            all_tokens[position]
            for position
            in local_positions
        ]

        local_weights = [
            1.0
            / (
                1.0
                + distance_from_span(
                    position,
                    aspect_start,
                    aspect_end,
                )
            )
            for position
            in local_positions
        ]

    local_vector, local_known, local_total = (
        weighted_glove_average(
            local_tokens,
            local_positions,
            glove_model,
            local_weights,
        )
    )

    feature = np.concatenate(
        [
            sentence_vector,
            aspect_vector,
            local_vector,
        ]
    ).astype(
        np.float32
    )

    metadata = {
        "aspect_found":
            aspect_span is not None,
        "local_context":
            " ".join(local_tokens),
        "clause_start":
            int(clause_left),
        "clause_end":
            int(clause_right),
        "text_known":
            int(text_known),
        "text_total":
            int(text_total),
        "aspect_known":
            int(aspect_known),
        "aspect_total":
            int(aspect_total),
        "local_known":
            int(local_known),
        "local_total":
            int(local_total),
        "feature_dimensions":
            int(feature.shape[0]),
    }

    oov = Counter(
        token
        for token
        in embedding_tokens(
            all_tokens
            + aspect_tokens
        )
        if token not in glove_model
    )

    return (
        feature,
        metadata,
        oov,
    )


def build_contrast_aware_matrix(
    dataframe,
    glove_model,
    context_window: int = 5,
):
    """
    Build a feature matrix for a DataFrame with `text` and `aspect`.
    """
    features = []
    metadata_rows = []
    all_oov = Counter()

    total = len(dataframe)

    print(
        f"Building contrast-aware 300d "
        f"GloVe features for {total:,} records "
        f"(window={context_window})..."
    )

    for number, row in enumerate(
        dataframe.itertuples(index=False),
        start=1,
    ):
        feature, metadata, oov = (
            build_contrast_aware_feature(
                text=row.text,
                aspect=row.aspect,
                glove_model=glove_model,
                context_window=context_window,
            )
        )

        features.append(
            feature
        )

        metadata_rows.append(
            metadata
        )

        all_oov.update(
            oov
        )

        if (
            number % 1000 == 0
            or number == total
        ):
            print(
                f"  Processed "
                f"{number:,}/{total:,}"
            )

    return (
        np.vstack(features).astype(
            np.float32
        ),
        metadata_rows,
        all_oov,
    )


# ============================================================
# PRODUCTION REPRESENTATION — TARGET CLAUSE + LOCAL CONTEXT 200d
# ============================================================

def build_target_clause_local_feature(
    text: str,
    aspect: str,
    glove_model,
    context_window: int = 5,
):
    """
    Production ABSA representation:
        100d target-clause mean GloVe
      + 100d distance-weighted local-context GloVe
      = 200 dimensions.

    Contrast markers bound the target clause so sentiment from another
    aspect does not leak into the target representation.
    """
    if context_window < 1:
        raise ValueError("context_window must be at least 1.")

    all_tokens = tokenize(text)
    aspect_tokens = tokenize(aspect)
    aspect_span = find_token_span(all_tokens, aspect_tokens)

    if aspect_span is None:
        clause_left = 0
        clause_right = len(all_tokens)
        local_left = 0
        local_right = len(all_tokens)
        aspect_start = None
        aspect_end = None
    else:
        aspect_start, aspect_end = aspect_span
        clause_left, clause_right = contrast_clause_bounds(
            all_tokens,
            aspect_start,
            aspect_end,
        )
        local_left = max(
            clause_left,
            aspect_start - context_window,
        )
        local_right = min(
            clause_right,
            aspect_end + context_window,
        )

    clause_positions = list(range(clause_left, clause_right))
    clause_tokens = [all_tokens[p] for p in clause_positions]
    clause_vector, clause_known, clause_total = weighted_glove_average(
        clause_tokens,
        clause_positions,
        glove_model,
    )

    local_positions = list(range(local_left, local_right))
    local_tokens = [all_tokens[p] for p in local_positions]

    if aspect_span is None:
        local_weights = [1.0] * len(local_positions)
    else:
        local_weights = [
            1.0 / (
                1.0
                + distance_from_span(
                    position,
                    aspect_start,
                    aspect_end,
                )
            )
            for position in local_positions
        ]

    local_vector, local_known, local_total = weighted_glove_average(
        local_tokens,
        local_positions,
        glove_model,
        local_weights,
    )

    feature = np.concatenate(
        [
            clause_vector,
            local_vector,
        ]
    ).astype(np.float32)

    metadata = {
        "aspect_found": aspect_span is not None,
        "target_clause": " ".join(clause_tokens),
        "local_context": " ".join(local_tokens),
        "context_source": "target_clause_distance_weighted",
        "clause_start": int(clause_left),
        "clause_end": int(clause_right),
        "aspect_token_start": (
            None if aspect_start is None else int(aspect_start)
        ),
        "aspect_token_end": (
            None if aspect_end is None else int(aspect_end)
        ),
        "clause_known": int(clause_known),
        "clause_total": int(clause_total),
        "local_known": int(local_known),
        "local_total": int(local_total),
        "feature_dimensions": int(feature.shape[0]),
    }

    oov = Counter(
        token
        for token in embedding_tokens(all_tokens)
        if token not in glove_model
    )

    return feature, metadata, oov


def build_target_clause_local_matrix(
    dataframe,
    glove_model,
    context_window: int = 5,
):
    """Build the production 200d matrix from `text` and `aspect`."""
    features = []
    metadata_rows = []
    all_oov = Counter()

    total = len(dataframe)
    print(
        f"Building Target Clause + Local 200d GloVe features "
        f"for {total:,} records (window={context_window})..."
    )

    for number, row in enumerate(
        dataframe.itertuples(index=False),
        start=1,
    ):
        feature, metadata, oov = build_target_clause_local_feature(
            text=row.text,
            aspect=row.aspect,
            glove_model=glove_model,
            context_window=context_window,
        )

        features.append(feature)
        metadata_rows.append(metadata)
        all_oov.update(oov)

        if number % 1000 == 0 or number == total:
            print(f"  Processed {number:,}/{total:,}")

    return (
        np.vstack(features).astype(np.float32),
        metadata_rows,
        all_oov,
    )


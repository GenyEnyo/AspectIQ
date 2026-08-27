from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import gensim.downloader as api

from src.preprocessing import preprocess_text



# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
FEATURE_DIR = BASE_DIR / "outputs" / "features"
RESULTS_DIR = BASE_DIR / "outputs" / "results"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_FILE = (
    PROCESSED_DIR /
    "absa_train_preprocessed.csv"
)

TEST_FILE = (
    PROCESSED_DIR /
    "absa_test_preprocessed.csv"
)


BASE_X_TRAIN_FILE = (
    FEATURE_DIR /
    "X_train_glove.npy"
)

BASE_X_TEST_FILE = (
    FEATURE_DIR /
    "X_test_glove.npy"
)


# ============================================================
# CONFIGURATION
# ============================================================

GLOVE_MODEL_NAME = "glove-wiki-gigaword-100"

EMBEDDING_DIM = 100

# Words before and after the aspect
CONTEXT_WINDOW = 4


# ============================================================
# HELPER
# ============================================================

def safe_int(value):

    if pd.isna(value):
        return None

    try:
        return int(float(value))

    except (TypeError, ValueError):
        return None


# ============================================================
# EXTRACT LOCAL ASPECT CONTEXT
# ============================================================

def extract_local_context(
    row,
    window=CONTEXT_WINDOW
):
    """
    Extract words immediately surrounding the target aspect.

    First use the official SemEval character offsets.
    If offsets are unavailable, search for the aspect string.
    If that also fails, fall back to the full sentence.
    """

    text = str(
        row.get("text", "")
    )

    aspect = str(
        row.get("aspect", "")
    )


    start = safe_int(
        row.get("from")
    )

    end = safe_int(
        row.get("to")
    )


    source = "offset"


    # --------------------------------------------------------
    # Validate official character offsets
    # --------------------------------------------------------

    valid_offset = (
        start is not None
        and end is not None
        and start >= 0
        and end > start
        and end <= len(text)
    )


    if not valid_offset:

        source = "aspect_search"

        position = (
            text.lower()
            .find(
                aspect.lower()
            )
        )


        if position >= 0:

            start = position

            end = (
                position
                + len(aspect)
            )

        else:

            # Fallback
            return text, "full_sentence_fallback"


    # --------------------------------------------------------
    # Divide sentence around aspect
    # --------------------------------------------------------

    left_text = text[:start]

    target_text = text[start:end]

    right_text = text[end:]


    left_words = (
        left_text.split()
    )

    right_words = (
        right_text.split()
    )


    local_words = (
        left_words[-window:]
        +
        [target_text]
        +
        right_words[:window]
    )


    local_context = " ".join(
        local_words
    ).strip()


    return (
        local_context,
        source
    )


# ============================================================
# MEAN GLOVE EMBEDDING
# ============================================================

def mean_embedding(
    text,
    glove_model
):

    if pd.isna(text):

        return np.zeros(
            EMBEDDING_DIM,
            dtype=np.float32
        )


    tokens = (
        str(text)
        .split()
    )


    vectors = [
        glove_model[token]
        for token in tokens
        if token
        in glove_model.key_to_index
    ]


    if not vectors:

        return np.zeros(
            EMBEDDING_DIM,
            dtype=np.float32
        )


    return np.mean(
        vectors,
        axis=0
    ).astype(
        np.float32
    )


# ============================================================
# PREPARE LOCAL CONTEXT
# ============================================================

def create_local_context(df):

    result_df = df.copy()


    extracted = result_df.apply(
        extract_local_context,
        axis=1
    )


    result_df[
        "local_context"
    ] = extracted.apply(
        lambda x: x[0]
    )


    result_df[
        "context_source"
    ] = extracted.apply(
        lambda x: x[1]
    )


    result_df[
        "preprocessed_local_context"
    ] = result_df[
        "local_context"
    ].apply(
        preprocess_text
    )


    return result_df


# ============================================================
# COVERAGE
# ============================================================

def calculate_coverage(
    series,
    glove_model,
    dataset_name
):

    all_tokens = []


    for text in series.fillna(""):

        all_tokens.extend(
            str(text).split()
        )


    total = len(
        all_tokens
    )


    covered = sum(
        token
        in glove_model.key_to_index
        for token
        in all_tokens
    )


    coverage = (
        covered
        / total
        * 100
        if total
        else 0
    )


    return {
        "dataset":
            dataset_name,

        "total_tokens":
            total,

        "covered_tokens":
            covered,

        "coverage_percentage":
            round(
                coverage,
                2
            )
    }


# ============================================================
# BUILD LOCAL CONTEXT EMBEDDINGS
# ============================================================

def build_local_embeddings(
    df,
    glove_model
):

    vectors = [
        mean_embedding(
            text,
            glove_model
        )
        for text
        in df[
            "preprocessed_local_context"
        ]
    ]


    return np.asarray(
        vectors,
        dtype=np.float32
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ENHANCED ASPECT-AWARE GLOVE REPRESENTATION")
    print("=" * 70)


    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    test_df = pd.read_csv(
        TEST_FILE
    )


    X_train_base = np.load(
        BASE_X_TRAIN_FILE
    )

    X_test_base = np.load(
        BASE_X_TEST_FILE
    )


    if len(train_df) != X_train_base.shape[0]:

        raise ValueError(
            "Training dataframe and baseline "
            "feature matrix do not match."
        )


    if len(test_df) != X_test_base.shape[0]:

        raise ValueError(
            "Test dataframe and baseline "
            "feature matrix do not match."
        )


    # --------------------------------------------------------
    # Extract local context
    # --------------------------------------------------------

    print(
        "\nExtracting aspect-local context..."
    )


    train_df = create_local_context(
        train_df
    )

    test_df = create_local_context(
        test_df
    )


    print(
        "\nTraining context sources:"
    )

    print(
        train_df[
            "context_source"
        ].value_counts()
    )


    print(
        "\nTest context sources:"
    )

    print(
        test_df[
            "context_source"
        ].value_counts()
    )


    # --------------------------------------------------------
    # Load GloVe
    # --------------------------------------------------------

    print(
        "\nLoading GloVe..."
    )


#    glove_model = api.load(
#        GLOVE_MODEL_NAME
#    )



    glove_path = api.load(
        GLOVE_MODEL_NAME,
        return_path=True
    )

    glove_model = KeyedVectors.load_word2vec_format(
        glove_path,
        binary=False
    )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    coverage_df = pd.DataFrame(
        [
            calculate_coverage(
                train_df[
                    "preprocessed_local_context"
                ],
                glove_model,
                "Train Local Context"
            ),

            calculate_coverage(
                test_df[
                    "preprocessed_local_context"
                ],
                glove_model,
                "Test Local Context"
            )
        ]
    )


    coverage_df.to_csv(
        RESULTS_DIR /
        "local_context_glove_coverage.csv",
        index=False
    )


    print("\n")
    print(
        coverage_df.to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Local embeddings
    # --------------------------------------------------------

    print(
        "\nCreating local-context embeddings..."
    )


    X_train_local = (
        build_local_embeddings(
            train_df,
            glove_model
        )
    )


    X_test_local = (
        build_local_embeddings(
            test_df,
            glove_model
        )
    )


    # --------------------------------------------------------
    # Enhanced representation
    #
    # Existing 200d =
    # sentence 100d + aspect 100d
    #
    # New =
    # 200d baseline + local-context 100d
    # --------------------------------------------------------

    X_train_enhanced = np.hstack(
        [
            X_train_base,
            X_train_local
        ]
    ).astype(
        np.float32
    )


    X_test_enhanced = np.hstack(
        [
            X_test_base,
            X_test_local
        ]
    ).astype(
        np.float32
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save(
        FEATURE_DIR /
        "X_train_glove_local_context.npy",
        X_train_enhanced
    )


    np.save(
        FEATURE_DIR /
        "X_test_glove_local_context.npy",
        X_test_enhanced
    )


    train_df.to_csv(
        FEATURE_DIR /
        "train_local_context_metadata.csv",
        index=False
    )


    test_df.to_csv(
        FEATURE_DIR /
        "test_local_context_metadata.csv",
        index=False
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FEATURE SHAPES")
    print("=" * 70)


    print(
        "Original train:",
        X_train_base.shape
    )

    print(
        "Enhanced train:",
        X_train_enhanced.shape
    )


    print(
        "Original test:",
        X_test_base.shape
    )

    print(
        "Enhanced test:",
        X_test_enhanced.shape
    )


    print("\n")
    print("=" * 70)
    print("LOCAL CONTEXT EXAMPLES")
    print("=" * 70)


    print(
        train_df[
            [
                "text",
                "aspect",
                "local_context",
                "preprocessed_local_context"
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
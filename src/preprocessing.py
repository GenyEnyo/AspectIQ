from pathlib import Path
import re
import html

import pandas as pd
import nltk

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = PROCESSED_DATA_DIR / "absa_model_dataset.csv"

OUTPUT_FILE = PROCESSED_DATA_DIR / "absa_preprocessed.csv"

TRAIN_OUTPUT_FILE = (
    PROCESSED_DATA_DIR / "absa_train_preprocessed.csv"
)

TEST_OUTPUT_FILE = (
    PROCESSED_DATA_DIR / "absa_test_preprocessed.csv"
)


# ============================================================
# NLP OBJECTS
# ============================================================

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))


# ============================================================
# IMPORTANT SENTIMENT WORDS TO PRESERVE
# ============================================================

PRESERVE_WORDS = {
    # Negation
    "no",
    "not",
    "nor",
    "never",
    "neither",

    # Contrast
    "but",
    "however",
    "although",
    "though",
    "yet",

    # Intensifiers
    "very",
    "really",
    "extremely",
    "highly",
    "too",
    "quite",
    "so",

    # Comparative sentiment
    "more",
    "most",
    "less",
    "least",
}


# Remove preserved words from stopword list
stop_words = stop_words - PRESERVE_WORDS


# ============================================================
# CONTRACTION HANDLING
# ============================================================

def expand_negative_contractions(text):
    """
    Expand common negative contractions so that negation
    is preserved before punctuation is removed.

    Examples:
        isn't  -> is not
        wasn't -> was not
        don't  -> do not
        can't  -> can not
        won't  -> will not
    """

    # Normalize curly apostrophe
    text = text.replace("’", "'")

    # Special cases
    text = re.sub(
        r"\bcan't\b",
        "can not",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\bwon't\b",
        "will not",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\bshan't\b",
        "shall not",
        text,
        flags=re.IGNORECASE
    )

    # General n't contraction
    text = re.sub(
        r"n't\b",
        " not",
        text,
        flags=re.IGNORECASE
    )

    return text


# ============================================================
# REGEX CLEANING
# ============================================================

def clean_text(text):
    """
    Clean text using regular expressions while preserving
    sentiment-bearing information.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Decode HTML entities
    # Example: &amp; -> &
    text = html.unescape(text)

    # Lowercase
    text = text.lower()

    # Expand negative contractions BEFORE punctuation removal
    text = expand_negative_contractions(text)

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Replace hyphens/slashes with spaces
    text = re.sub(
        r"[-/]",
        " ",
        text
    )

    # Remove digits
    text = re.sub(
        r"\d+",
        " ",
        text
    )

    # Remove punctuation and symbols
    # Keep alphabetic characters and whitespace only
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Normalize repeated whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# POS TAG MAPPING
# ============================================================

def get_wordnet_pos(treebank_tag):
    """
    Convert NLTK POS tags to WordNet POS tags.
    """

    if treebank_tag.startswith("J"):
        return wordnet.ADJ

    if treebank_tag.startswith("V"):
        return wordnet.VERB

    if treebank_tag.startswith("N"):
        return wordnet.NOUN

    if treebank_tag.startswith("R"):
        return wordnet.ADV

    return wordnet.NOUN


# ============================================================
# TOKENISATION
# ============================================================

def tokenize_text(text):

    if not text:
        return []

    return word_tokenize(text)


# ============================================================
# STOPWORD REMOVAL
# ============================================================

def remove_stopwords(tokens):
    """
    Remove ordinary English stopwords but retain words
    important to sentiment.
    """

    return [
        token
        for token in tokens
        if token not in stop_words
    ]


# ============================================================
# LEMMATISATION
# ============================================================

def lemmatize_tokens(tokens):

    if not tokens:
        return []

    tagged_tokens = nltk.pos_tag(tokens)

    lemmatized = []

    for word, tag in tagged_tokens:

        wordnet_pos = get_wordnet_pos(tag)

        lemma = lemmatizer.lemmatize(
            word,
            pos=wordnet_pos
        )

        lemmatized.append(lemma)

    return lemmatized


# ============================================================
# COMPLETE TEXT PREPROCESSOR
# ============================================================

def preprocess_text(text):
    """
    Complete preprocessing pipeline.
    """

    cleaned = clean_text(text)

    tokens = tokenize_text(cleaned)

    filtered_tokens = remove_stopwords(tokens)

    lemmatized_tokens = lemmatize_tokens(
        filtered_tokens
    )

    return " ".join(lemmatized_tokens)


# ============================================================
# PREPROCESS DATAFRAME
# ============================================================

def preprocess_dataset(df):

    processed_df = df.copy()

    print("\nCleaning sentence text...")

    processed_df["clean_text"] = (
        processed_df["text"]
        .apply(clean_text)
    )


    print("Preprocessing sentence text...")

    processed_df["preprocessed_text"] = (
        processed_df["text"]
        .apply(preprocess_text)
    )


    print("Cleaning aspect terms...")

    processed_df["clean_aspect"] = (
        processed_df["aspect"]
        .apply(clean_text)
    )


    print("Preprocessing aspect terms...")

    processed_df["preprocessed_aspect"] = (
        processed_df["aspect"]
        .apply(preprocess_text)
    )


    # --------------------------------------------------------
    # Useful text statistics
    # --------------------------------------------------------

    processed_df["original_word_count"] = (
        processed_df["text"]
        .astype(str)
        .apply(lambda x: len(x.split()))
    )


    processed_df["processed_word_count"] = (
        processed_df["preprocessed_text"]
        .apply(
            lambda x: len(x.split())
            if x
            else 0
        )
    )


    return processed_df


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_preprocessing(df):

    print("\n")
    print("=" * 70)
    print("PREPROCESSING VALIDATION")
    print("=" * 70)


    print(f"\nTotal records: {len(df):,}")


    print(
        "\nEmpty cleaned sentences:",
        (df["clean_text"] == "").sum()
    )


    print(
        "Empty preprocessed sentences:",
        (df["preprocessed_text"] == "").sum()
    )


    print(
        "Empty preprocessed aspects:",
        (df["preprocessed_aspect"] == "").sum()
    )


    print(
        "\nAverage original words:",
        round(
            df["original_word_count"].mean(),
            2
        )
    )


    print(
        "Average processed words:",
        round(
            df["processed_word_count"].mean(),
            2
        )
    )


    print("\nPolarity distribution:")

    print(
        df["polarity"]
        .value_counts(dropna=False)
    )


    print("\nDomain distribution:")

    print(
        df["domain"]
        .value_counts()
    )


    print("\nTrain/Test distribution:")

    print(
        df["split"]
        .value_counts()
    )


# ============================================================
# SHOW BEFORE / AFTER EXAMPLES
# ============================================================

def show_examples(df, number=10):

    columns = [
        "text",
        "aspect",
        "polarity",
        "clean_text",
        "preprocessed_text",
        "preprocessed_aspect",
    ]

    print("\n")
    print("=" * 70)
    print("BEFORE AND AFTER EXAMPLES")
    print("=" * 70)

    print(
        df[columns]
        .head(number)
        .to_string(index=False)
    )


# ============================================================
# SAVE DATA
# ============================================================

def save_preprocessed_data(df):

    # Complete dataset
    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )


    # --------------------------------------------------------
    # Preserve official SemEval Train/Test split
    # --------------------------------------------------------

    train_df = (
        df[
            df["split"]
            .str.lower()
            .eq("train")
        ]
        .copy()
    )


    test_df = (
        df[
            df["split"]
            .str.lower()
            .eq("test")
        ]
        .copy()
    )


    train_df.to_csv(
        TRAIN_OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )


    test_df.to_csv(
        TEST_OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )


    print("\n")
    print("=" * 70)
    print("PREPROCESSED FILES SAVED")
    print("=" * 70)

    print(f"\nAll records : {OUTPUT_FILE}")
    print(f"Train       : {TRAIN_OUTPUT_FILE}")
    print(f"Test        : {TEST_OUTPUT_FILE}")

    print(
        f"\nTraining records: {len(train_df):,}"
    )

    print(
        f"Test records: {len(test_df):,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"""
Input dataset does not exist:

{INPUT_FILE}

Run Step 2 first:

python -m src.data_loader
"""
        )


    print("=" * 70)
    print("ABSA DATA CLEANING AND PREPROCESSING")
    print("=" * 70)


    # --------------------------------------------------------
    # Load Step 2 dataset
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)


    print(
        f"\nLoaded {len(df):,} aspect-level records."
    )


    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    processed_df = preprocess_dataset(df)


    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_preprocessing(processed_df)


    # --------------------------------------------------------
    # Display examples
    # --------------------------------------------------------

    show_examples(
        processed_df,
        number=10
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_preprocessed_data(
        processed_df
    )


if __name__ == "__main__":
    main()
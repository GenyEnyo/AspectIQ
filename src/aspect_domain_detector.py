from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import math
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords

from src.preprocessing import preprocess_text

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_FILE = BASE_DIR / "data" / "processed" / "absa_train_preprocessed.csv"


def normalise_phrase(value: str) -> str:
    value = str(value).lower().strip()
    value = value.replace("’", "'")
    value = re.sub(r'["“”]', "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def phrase_pattern(phrase: str) -> str:
    tokens = [re.escape(token) for token in phrase.split() if token]
    if not tokens:
        return r"(?!x)x"
    return r"(?<!\w)" + r"\s+".join(tokens) + r"(?!\w)"


@lru_cache(maxsize=1)
def load_training_knowledge():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}\n"
            "Run the preprocessing pipeline first."
        )

    df = pd.read_csv(TRAIN_FILE)
    required = {"text", "aspect", "domain"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Training data is missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = df.dropna(subset=["text", "aspect", "domain"]).copy()

    aspect_domain_counts = defaultdict(Counter)
    aspect_total_counts = Counter()

    for row in df.itertuples(index=False):
        aspect = normalise_phrase(getattr(row, "aspect"))
        domain = str(getattr(row, "domain")).strip()
        if not aspect:
            continue
        aspect_domain_counts[aspect][domain] += 1
        aspect_total_counts[aspect] += 1

    lexicon = sorted(
        aspect_total_counts.keys(),
        key=lambda x: (
            -len(x.split()),
            -len(x),
            -aspect_total_counts[x],
            x,
        ),
    )

    domain_token_counts = defaultdict(Counter)
    domain_totals = Counter()
    vocabulary = set()

    text_col = (
        "preprocessed_text"
        if "preprocessed_text" in df.columns
        else "text"
    )

    sentence_cols = [
        col for col in ["sentence_id", "domain", "split", text_col]
        if col in df.columns
    ]
    sentence_df = df[sentence_cols].drop_duplicates().copy()

    for row in sentence_df.itertuples(index=False):
        domain = str(getattr(row, "domain")).strip()
        text_value = str(getattr(row, text_col))
        if text_col != "preprocessed_text":
            text_value = preprocess_text(text_value)

        tokens = [token for token in text_value.split() if token]
        domain_token_counts[domain].update(tokens)
        domain_totals[domain] += len(tokens)
        vocabulary.update(tokens)

    domains = sorted(domain_token_counts.keys())

    return {
        "lexicon": lexicon,
        "aspect_domain_counts": dict(aspect_domain_counts),
        "aspect_total_counts": aspect_total_counts,
        "domain_token_counts": dict(domain_token_counts),
        "domain_totals": domain_totals,
        "vocabulary": vocabulary,
        "domains": domains,
    }


def find_lexicon_aspects(text: str):
    knowledge = load_training_knowledge()
    text = str(text)
    matches = []

    for aspect in knowledge["lexicon"]:
        for match in re.finditer(
            phrase_pattern(aspect),
            text,
            flags=re.IGNORECASE,
        ):
            matches.append(
                {
                    "aspect": text[match.start():match.end()],
                    "canonical_aspect": aspect,
                    "start": match.start(),
                    "end": match.end(),
                    "source": "training_lexicon",
                    "training_frequency": int(
                        knowledge["aspect_total_counts"][aspect]
                    ),
                }
            )

    if not matches:
        return []

    matches.sort(
        key=lambda item: (
            item["start"],
            -(item["end"] - item["start"]),
            -item["training_frequency"],
        )
    )

    selected = []
    for candidate in matches:
        overlaps = any(
            not (
                candidate["end"] <= chosen["start"]
                or candidate["start"] >= chosen["end"]
            )
            for chosen in selected
        )
        if not overlaps:
            selected.append(candidate)

    selected.sort(key=lambda item: item["start"])
    return selected


def find_noun_phrase_fallback(text: str):
    text = str(text)
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

    noun_tags = {"NN", "NNS", "NNP", "NNPS"}
    candidates = []
    current = []

    for token, tag in tagged:
        if tag in noun_tags and re.search(r"[A-Za-z]", token):
            current.append(token)
        else:
            if current:
                candidates.append(current)
                current = []

    if current:
        candidates.append(current)

    stop = set(stopwords.words("english"))
    generic = {
        "thing", "things", "something", "anything",
        "everything", "nothing", "time", "times",
        "day", "days", "way", "ways",
    }

    output = []
    seen = set()

    for candidate_tokens in candidates:
        candidate_tokens = candidate_tokens[-3:]
        phrase = " ".join(candidate_tokens).strip()
        canonical = normalise_phrase(phrase)

        if not canonical or canonical in stop or canonical in generic:
            continue

        match = re.search(
            phrase_pattern(canonical),
            text,
            flags=re.IGNORECASE,
        )
        if match is None:
            continue

        key = (match.start(), match.end(), canonical)
        if key in seen:
            continue
        seen.add(key)

        output.append(
            {
                "aspect": text[match.start():match.end()],
                "canonical_aspect": canonical,
                "start": match.start(),
                "end": match.end(),
                "source": "noun_phrase_fallback",
                "training_frequency": 0,
            }
        )

    return output


def detect_aspects(text: str):
    matches = find_lexicon_aspects(text)
    if matches:
        return matches
    return find_noun_phrase_fallback(text)


def _aspect_domain_score(aspects):
    knowledge = load_training_knowledge()
    scores = Counter()

    for item in aspects:
        canonical = item["canonical_aspect"]
        counts = knowledge["aspect_domain_counts"].get(canonical, {})
        for domain, count in counts.items():
            scores[domain] += math.log1p(count)

    return scores


def _token_domain_score(text: str):
    knowledge = load_training_knowledge()
    processed = preprocess_text(text)
    tokens = [token for token in processed.split() if token]

    domains = knowledge["domains"]
    vocabulary_size = max(len(knowledge["vocabulary"]), 1)
    scores = Counter()

    for domain in domains:
        counts = knowledge["domain_token_counts"][domain]
        total = knowledge["domain_totals"][domain]

        token_score = 0.0
        for token in tokens:
            token_score += math.log(
                (counts.get(token, 0) + 1)
                / (total + vocabulary_size)
            )

        if tokens:
            token_score /= len(tokens)

        scores[domain] = token_score

    return scores


def detect_domain(text: str, aspects=None):
    knowledge = load_training_knowledge()

    if aspects is None:
        aspects = detect_aspects(text)

    aspect_scores = _aspect_domain_score(aspects)
    token_scores = _token_domain_score(text)
    domains = knowledge["domains"]

    if not domains:
        return {
            "domain": "Unknown",
            "confidence": 0.0,
            "method": "unavailable",
            "scores": {},
        }

    token_values = [token_scores.get(domain, 0.0) for domain in domains]
    token_mean = sum(token_values) / len(token_values)

    combined = {}
    for domain in domains:
        token_component = token_scores.get(domain, 0.0) - token_mean
        aspect_component = aspect_scores.get(domain, 0.0)
        combined[domain] = 1.5 * aspect_component + token_component

    best_domain = max(combined, key=combined.get)
    max_score = max(combined.values())

    exp_scores = {
        domain: math.exp(max(min(score - max_score, 50), -50))
        for domain, score in combined.items()
    }
    denominator = sum(exp_scores.values())

    probabilities = {
        domain: (
            exp_scores[domain] / denominator
            if denominator else 0.0
        )
        for domain in domains
    }

    return {
        "domain": best_domain,
        "confidence": float(probabilities[best_domain]),
        "method": (
            "aspect_lexicon_and_domain_vocabulary"
            if aspect_scores
            else "domain_vocabulary"
        ),
        "scores": {
            domain: float(probabilities[domain])
            for domain in domains
        },
    }


def detect_review_structure(text: str):
    aspects = detect_aspects(text)
    domain = detect_domain(text=text, aspects=aspects)
    return {
        "domain": domain,
        "aspects": aspects,
    }

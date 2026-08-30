from __future__ import annotations

from pathlib import Path
import html
import json
import re
import xml.etree.ElementTree as ET

import nltk
import pandas as pd
import plotly.express as px
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AspectIQ",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="auto",
)


# ============================================================
# STYLES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
STYLE_FILE = BASE_DIR / "styles" / "main.css"


def load_css(css_path: Path) -> None:
    """Load the application stylesheet from a separate CSS file."""
    if not css_path.exists():
        st.error(f"Stylesheet not found: {css_path}")
        st.stop()

    css = css_path.read_text(encoding="utf-8")
    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )


load_css(STYLE_FILE)


# ============================================================
# NLTK — LOCAL + STREAMLIT CLOUD SAFE
# ============================================================

@st.cache_resource
def prepare_nltk_resources():
    """
    Download NLTK resources once per application process.

    Streamlit Community Cloud provides /tmp as writable ephemeral storage.
    For local Windows execution, the project-local .nltk_data directory is
    used when /tmp is not available.
    """
    cloud_tmp = Path("/tmp")

    nltk_directory = (
        cloud_tmp / "nltk_data"
        if cloud_tmp.exists()
        else BASE_DIR / ".nltk_data"
    )

    nltk_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    nltk_path = str(nltk_directory)

    if nltk_path not in nltk.data.path:
        nltk.data.path.insert(
            0,
            nltk_path,
        )

    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger_eng",
    ]

    for resource in resources:
        nltk.download(
            resource,
            download_dir=nltk_path,
            quiet=True,
        )

    return nltk_path


NLTK_DATA_PATH = prepare_nltk_resources()


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.aspect_domain_detector import detect_review_structure
from src.prediction import (
    load_prediction_assets,
    predict_multiple_aspects,
)
from src.preprocessing import preprocess_text


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = BASE_DIR / "outputs" / "results"
FIGURE_DIR = BASE_DIR / "outputs" / "figures"
MODEL_DIR = BASE_DIR / "models"
TARGET_CONTEXT_RESULTS_DIR = (
    BASE_DIR
    / "outputs"
    / "target_context_experiments"
)
PRIOR_EVALUATION_DIR = (
    BASE_DIR
    / "outputs"
    / "final_champion_evaluation"
)
TRAIN_DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "absa_train_preprocessed.csv"
)


# ============================================================
# ASSETS
# ============================================================

@st.cache_resource
def load_assets():
    return load_prediction_assets()


try:
    model, glove_model = load_assets()
except Exception as exc:
    st.error("AspectIQ could not load the trained model or GloVe resources.")
    st.exception(exc)
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Overview"

if "analyze_mode" not in st.session_state:
    st.session_state.analyze_mode = "Text"

if "analyses" not in st.session_state:
    st.session_state.analyses = []

if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()


# ============================================================
# INPUT SEGMENTATION
# ============================================================

MAX_ANALYSIS_UNIT_CHARS = 700


def _basic_sentence_split(text: str):
    """
    Lightweight sentence splitting used only to break very long blocks.

    The production sentiment preprocessing remains unchanged.
    """
    text = re.sub(
        r"\.{3,}\s*",
        ". ",
        str(text),
    )

    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z#*])",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def _split_long_block(
    block: str,
    max_chars: int = MAX_ANALYSIS_UNIT_CHARS,
):
    """Pack sentences into manageable ABSA analysis units."""
    block = str(block).strip()

    if not block:
        return []

    if len(block) <= max_chars:
        return [block]

    sentences = _basic_sentence_split(
        block
    )

    if len(sentences) <= 1:
        return [
            block[start:start + max_chars].strip()
            for start in range(
                0,
                len(block),
                max_chars,
            )
            if block[start:start + max_chars].strip()
        ]

    units = []
    current = ""

    for sentence in sentences:
        candidate = (
            f"{current} {sentence}".strip()
            if current
            else sentence
        )

        if (
            current
            and len(candidate) > max_chars
        ):
            units.append(
                current
            )
            current = sentence
        else:
            current = candidate

    if current:
        units.append(
            current
        )

    return units


def segment_input_text(text: str):
    """
    Convert pasted free-form text into domain-sized analysis units.

    A short ordinary review remains one unit. Long or multi-paragraph input
    is segmented before domain detection, aspect extraction and sentiment
    classification. This allows one submission to contain both Restaurant
    and Laptop content without forcing one domain onto the whole document.
    """
    text = html.unescape(
        str(text or "")
    )

    text = (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u200b", "")
    )

    # Put Markdown headings onto their own logical line, including headings
    # pasted inline after another review.
    text = re.sub(
        r"\s*(#{1,6}\s+[^\n]+)",
        r"\n\1\n",
        text,
    )

    text = re.sub(
        r"\s*(\*\*[^*\n]{3,120}\*\*)\s*",
        r"\n\1\n",
        text,
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    blocks = [
        block.strip()
        for block in re.split(
            r"\n+",
            text,
        )
        if block.strip()
    ]

    # Ordinary single-review input should remain untouched.
    if (
        len(blocks) == 1
        and len(blocks[0])
        <= MAX_ANALYSIS_UNIT_CHARS
    ):
        return blocks

    units = []
    pending_heading = ""

    for block in blocks:
        is_markdown_heading = bool(
            re.fullmatch(
                r"#{1,6}\s+.+|\*\*[^*]+\*\*",
                block,
            )
        )

        cleaned = re.sub(
            r"^#{1,6}\s*",
            "",
            block,
        )

        cleaned = (
            cleaned
            .replace("**", "")
            .strip()
        )

        if not cleaned:
            continue

        # A heading is context for the paragraph that follows it.
        if is_markdown_heading:
            pending_heading = (
                f"{pending_heading} {cleaned}"
                .strip()
            )
            continue

        if pending_heading:
            cleaned = (
                f"{pending_heading}. {cleaned}"
            )
            pending_heading = ""

        units.extend(
            _split_long_block(
                cleaned
            )
        )

    if pending_heading:
        if units:
            units[-1] = (
                f"{units[-1]} {pending_heading}"
                .strip()
            )
        else:
            units.append(
                pending_heading
            )

    # Remove fragments that cannot support useful text analysis.
    useful_units = []

    for unit in units:
        word_count = len(
            re.findall(
                r"[A-Za-z]+",
                unit,
            )
        )

        if word_count >= 3:
            useful_units.append(
                unit.strip()
            )

    return (
        useful_units
        if useful_units
        else [
            text.strip()
        ]
    )


def expand_analysis_records(values):
    """
    Preserve normal review rows while segmenting unusually long/mixed rows.
    """
    records = []

    for value in values:
        if pd.isna(value):
            continue

        value = str(value).strip()

        if not value:
            continue

        records.extend(
            segment_input_text(
                value
            )
        )

    return records



# ============================================================
# FILE HELPERS
# ============================================================

TEXT_COLUMN_CANDIDATES = [
    "text",
    "review",
    "review_text",
    "sentence",
    "content",
    "comment",
    "comments",
    "feedback",
    "description",
]


def clean_reviews(values):
    return expand_analysis_records(
        values
    )


def detect_csv_text_column(df):
    lower_map = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]

    text_columns = [
        column
        for column in df.columns
        if (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
        )
    ]

    if not text_columns:
        raise ValueError(
            "No text-like column could be detected in the CSV file."
        )

    scored = []

    for column in text_columns:
        series = df[column].dropna().astype(str)
        average_length = (
            series.str.len().mean()
            if not series.empty
            else 0
        )

        scored.append(
            (
                average_length,
                column,
            )
        )

    scored.sort(reverse=True)
    return scored[0][1]


def read_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".txt":
        content = (
            uploaded_file
            .getvalue()
            .decode(
                "utf-8",
                errors="replace",
            )
        )

        return (
            segment_input_text(
                content
            ),
            "TXT",
            None,
        )

    if suffix == ".csv":
        df = pd.read_csv(uploaded_file)

        if df.empty:
            return [], "CSV", None

        text_column = detect_csv_text_column(df)

        return (
            clean_reviews(df[text_column]),
            "CSV",
            str(text_column),
        )

    if suffix == ".xml":
        root = ET.fromstring(
            uploaded_file.getvalue()
        )

        texts = [
            element.text.strip()
            for element in root.findall(".//sentence/text")
            if element.text and element.text.strip()
        ]

        if not texts:
            texts = [
                element.text.strip()
                for element in root.findall(".//text")
                if element.text and element.text.strip()
            ]

        return (
            expand_analysis_records(
                texts
            ),
            "XML",
            None,
        )

    raise ValueError(
        "Unsupported file type. Use TXT, CSV, or XML."
    )



# ============================================================
# CORPUS / TEACHING HELPERS
# ============================================================

@st.cache_data
def load_training_corpus():
    """Load the processed SemEval training corpus once."""
    if not TRAIN_DATA_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(TRAIN_DATA_FILE)


def first_existing_column(df, candidates):
    """Return the first candidate column that exists in the DataFrame."""
    for column in candidates:
        if column in df.columns:
            return column

    return None


def instructional_preprocessing_steps(text):
    """
    Create transparent intermediate text-processing steps for teaching/demo.

    These stages are inspection aids inspired by the classroom practical.
    They do not replace the production preprocessing pipeline. The final
    model-ready output is always produced by src.preprocessing.preprocess_text.
    """
    raw_text = str(text or "")
    lowercase = raw_text.lower()

    no_html = re.sub(
        r"<.*?>",
        " ",
        lowercase,
    )

    no_urls = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        no_html,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        no_urls,
    ).strip()

    tokens = (
        word_tokenize(normalized)
        if normalized
        else []
    )

    stop_words = set(
        stopwords.words("english")
    )

    # Keep words that can materially affect sentiment interpretation.
    preserved_sentiment_terms = {
        "no",
        "not",
        "nor",
        "never",
        "but",
        "however",
        "very",
        "too",
    }

    filtered_tokens = [
        token
        for token in tokens
        if (
            token.lower() not in stop_words
            or token.lower()
            in preserved_sentiment_terms
        )
    ]

    return {
        "Original Text": raw_text,
        "Lowercase": lowercase,
        "HTML / URL Normalisation": normalized,
        "Tokens": tokens,
        "Stopword-Filtered Tokens": filtered_tokens,
        "Inspection Text": " ".join(
            filtered_tokens
        ),
    }


def safe_preprocess_text(text):
    """Normalise the project's preprocessing output for display."""
    result = preprocess_text(text)

    if isinstance(result, str):
        return result

    if isinstance(result, (list, tuple)):
        return " ".join(
            str(item)
            for item in result
        )

    return str(result)



def build_submitted_reviews_dataframe(analyses):
    """
    Build one row per review from the text/file supplied by the user.

    This is the primary dataset shown in Data Explorer.
    """
    rows = []

    for analysis in analyses:
        sentiments = analysis.get(
            "sentiments",
            [],
        )

        aspects = [
            result.get(
                "aspect",
                "",
            )
            for result in sentiments
            if result.get(
                "aspect"
            )
        ]

        sentiment_labels = [
            str(
                result.get(
                    "sentiment",
                    ""
                )
            ).title()
            for result in sentiments
            if result.get(
                "sentiment"
            )
        ]

        rows.append(
            {
                "Review ID": analysis.get(
                    "review_id"
                ),
                "Review": analysis.get(
                    "text",
                    "",
                ),
                "Detected Domain": analysis.get(
                    "domain",
                    {},
                ).get(
                    "domain",
                    "Unknown",
                ),
                "Domain Confidence": round(
                    float(
                        analysis.get(
                            "domain",
                            {},
                        ).get(
                            "confidence",
                            0.0,
                        )
                    )
                    * 100,
                    2,
                ),
                "Aspects Detected": len(
                    aspects
                ),
                "Aspects": ", ".join(
                    aspects
                ),
                "Sentiments": ", ".join(
                    sentiment_labels
                ),
            }
        )

    return pd.DataFrame(rows)


def build_preprocessing_dataset(analyses):
    """
    Build one row per submitted review with the actual model-ready text.
    """
    rows = []

    for analysis in analyses:
        review_text = str(
            analysis.get(
                "text",
                "",
            )
        )

        rows.append(
            {
                "Review ID": analysis.get(
                    "review_id"
                ),
                "Original Text": review_text,
                "Model-Ready Text": safe_preprocess_text(
                    review_text
                ),
            }
        )

    return pd.DataFrame(rows)

# ============================================================
# ANALYSIS
# ============================================================

def analyse_review(review_id, text):
    structure = detect_review_structure(text)

    domain_result = structure["domain"]
    detected_aspects = structure["aspects"]

    sentiment_results = []

    if detected_aspects:
        sentiment_results = predict_multiple_aspects(
            text=text,
            aspects=detected_aspects,
            model=model,
            glove_model=glove_model,
        )

    rows = []

    for result in sentiment_results:
        rows.append(
            {
                "Review ID": review_id,
                "Review": text,
                "Detected Domain": domain_result["domain"],
                "Domain Confidence": round(
                    domain_result["confidence"] * 100,
                    2,
                ),
                "Aspect": result["aspect"],
                "Aspect Detection": result[
                    "aspect_detection_source"
                ],
                "Sentiment": result["sentiment"].title(),
                "Sentiment Confidence": round(
                    result["confidence"] * 100,
                    2,
                ),
                "Local Context": result["local_context"],
                "Feature Dimensions": result[
                    "feature_dimensions"
                ],
            }
        )

    return {
        "review_id": review_id,
        "text": text,
        "domain": domain_result,
        "detected_aspects": detected_aspects,
        "sentiments": sentiment_results,
        "rows": rows,
    }


def analyse_reviews(reviews):
    analyses = []
    all_rows = []

    total = len(reviews)

    if total > 1:
        st.caption(
            f"Input was analyzed as {total} text units so that different "
            "domains can be identified independently."
        )

    progress = st.progress(
        0,
        text="AspectIQ is analyzing your reviews..."
    )

    for index, text in enumerate(
        reviews,
        start=1,
    ):
        analysis = analyse_review(
            index,
            text,
        )

        analyses.append(analysis)
        all_rows.extend(analysis["rows"])

        progress.progress(
            index / total,
            text=f"Analyzing review {index} of {total}...",
        )

    progress.empty()

    st.session_state.analyses = analyses
    st.session_state.results_df = pd.DataFrame(all_rows)

    return analyses, st.session_state.results_df


# ============================================================
# COMPONENTS
# ============================================================

def go_to(page, mode=None):
    st.session_state.page = page

    if mode is not None:
        st.session_state.analyze_mode = mode

    st.rerun()


def render_hero():
    with st.container(key="hero_shell"):

        left, right = st.columns(
            [1.55, .65],
            gap="large",
        )

        with left:
            st.markdown(
                """
<div class="hero-kicker">Customer Experience Intelligence</div>
<div class="hero-title">AspectIQ</div>
<div class="hero-subtitle">
    Intelligent Aspect-Based Sentiment Analytics
</div>
<div class="hero-copy">
    Automatically detect review domains, identify customer-facing aspects,
    and measure the sentiment expressed toward each aspect.
</div>
""",
                unsafe_allow_html=True,
            )

        with right:
            st.markdown(
                """
<div class="hero-visual-card">
    <div class="hero-visual-title">SENTIMENT SIGNAL</div>
    <div class="hero-bars">
        <i style="height:28%"></i>
        <i style="height:40%"></i>
        <i style="height:52%"></i>
        <i style="height:63%"></i>
        <i style="height:76%"></i>
        <i style="height:88%"></i>
    </div>
    <div style="
        display:flex;
        justify-content:flex-end;
        gap:.45rem;
        margin-top:.5rem;
        font-size:.66rem;
        font-weight:700;
        color:#D9E7FF;">
        <span style="color:#79D692;">● Positive</span>
        <span style="color:#F5C56B;">● Neutral</span>
        <span style="color:#F48A8A;">● Negative</span>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )


def kpi_html(
    icon,
    icon_class,
    label,
    value,
):
    return f"""
<div class="kpi-card">
    <div class="kpi-wrap">
        <div class="kpi-icon {icon_class}">{icon}</div>
        <div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
    </div>
</div>
"""


def overall_mood(results_df):
    if results_df.empty:
        return "—"

    counts = results_df[
        "Sentiment"
    ].value_counts()

    if (
        len(counts) > 1
        and counts.iloc[0] / counts.sum() < .65
    ):
        return "Mixed"

    return counts.index[0]


def detected_domain(analyses):
    if not analyses:
        return "—"

    domains = sorted(
        {
            item["domain"]["domain"]
            for item in analyses
            if item["domain"]["domain"]
            not in {
                "",
                "Unknown",
            }
        }
    )

    if not domains:
        return "Unknown"

    if len(domains) == 1:
        return domains[0]

    return " + ".join(
        domains
    )


def render_kpis(
    analyses,
    results_df,
):
    values = [
        (
            "🧾",
            "i-blue",
            "Text Units Analyzed",
            len(analyses),
        ),
        (
            "☷",
            "i-purple",
            "Aspects Detected",
            len(results_df),
        ),
        (
            "💻",
            "i-teal",
            "Detected Domain",
            detected_domain(
                analyses
            ),
        ),
        (
            "◉",
            "i-orange",
            "Overall Sentiment",
            overall_mood(
                results_df
            ),
        ),
    ]

    cols = st.columns(4)

    for col, item in zip(
        cols,
        values,
    ):
        with col:
            st.markdown(
                kpi_html(*item),
                unsafe_allow_html=True,
            )


def sentiment_style(sentiment):
    sentiment = str(sentiment).lower()

    if sentiment == "positive":
        return (
            "aspect-positive",
            "p-positive",
            "#23A447",
            "😊",
        )

    if sentiment == "negative":
        return (
            "aspect-negative",
            "p-negative",
            "#E23939",
            "😟",
        )

    return (
        "aspect-neutral",
        "p-neutral",
        "#F59E0B",
        "😐",
    )


def render_aspect_card(result):
    (
        card_class,
        pill_class,
        color,
        emoji,
    ) = sentiment_style(
        result["sentiment"]
    )

    confidence = (
        float(
            result["confidence"]
        )
        * 100
    )

    st.markdown(
        f"""
<div class="aspect-card {card_class}">
    <div class="aspect-top">
        <div class="aspect-icon">{emoji}</div>
        <div>
            <div class="aspect-name">{result["aspect"].title()}</div>
            <span class="aspect-pill {pill_class}">
                {result["sentiment"].title()}
            </span>
        </div>
    </div>
    <div class="aspect-info">
        <div>
            <div class="meta-label">Confidence</div>
            <div class="meta-value" style="color:{color};font-size:1.2rem;">
                {confidence:.0f}%
            </div>
            <div class="progress-track">
                <div class="progress-fill"
                     style="width:{confidence:.1f}%;background:{color};">
                </div>
            </div>
        </div>
        <div>
            <div class="meta-label">Local Context</div>
            <div class="meta-value">
                {result["local_context"]}
            </div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_charts(results_df):
    left, right = st.columns(
        [0.9, 1.45],
        gap="large",
    )

    with left:
        st.markdown(
            '<div class="panel-shell">'
            '<div class="panel-title">Sentiment Distribution</div>',
            unsafe_allow_html=True,
        )

        sentiment_counts = (
            results_df["Sentiment"]
            .value_counts()
            .reindex(
                [
                    "Positive",
                    "Negative",
                    "Neutral",
                ],
                fill_value=0,
            )
            .rename_axis("Sentiment")
            .reset_index(name="Count")
        )

        figure = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Count",
            hole=.60,
            color="Sentiment",
            color_discrete_map={
                "Positive": "#23A447",
                "Negative": "#E23939",
                "Neutral": "#F59E0B",
            },
        )

        figure.update_layout(
            height=320,
            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10,
            ),
            legend=dict(
                orientation="h",
                y=-.04,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
            config={
                "displayModeBar": False
            },
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            '<div class="panel-shell">'
            '<div class="panel-title">Top Detected Aspects</div>',
            unsafe_allow_html=True,
        )

        aspects = (
            results_df["Aspect"]
            .str.lower()
            .value_counts()
            .head(10)
            .rename_axis("Aspect")
            .reset_index(name="Count")
        )

        figure = px.bar(
            aspects,
            x="Aspect",
            y="Count",
            text="Count",
        )

        figure.update_traces(
            marker_color="#2C6FF2"
        )

        figure.update_layout(
            height=320,
            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10,
            ),
            xaxis=dict(
                title=None,
            ),
            yaxis=dict(
                title=None,
                gridcolor="#E8EDF5",
                zeroline=False,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
            config={
                "displayModeBar": False
            },
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


def render_results(
    analyses,
    results_df,
):
    if not analyses:
        return

    render_kpis(
        analyses,
        results_df,
    )

    st.markdown(
        '<div class="section-title" style="margin-top:1.3rem;">'
        'What the customer is saying'
        '</div>',
        unsafe_allow_html=True,
    )

    domain_results = {}

    for analysis in analyses:
        domain_name = analysis[
            "domain"
        ][
            "domain"
        ]

        domain_results.setdefault(
            domain_name,
            [],
        )

        domain_results[
            domain_name
        ].extend(
            analysis[
                "sentiments"
            ]
        )

    aspect_count = sum(
        len(items)
        for items in domain_results.values()
    )

    if aspect_count == 0:
        st.warning(
            "No explicit aspect terms were detected."
        )
        return

    for domain_name in sorted(
        domain_results
    ):
        aspect_results = domain_results[
            domain_name
        ]

        if not aspect_results:
            continue

        st.markdown(
            f"### {domain_name} domain"
        )

        for start in range(
            0,
            len(aspect_results),
            2,
        ):
            row_items = aspect_results[
                start:
                start + 2
            ]

            cols = st.columns(2)

            for col, result in zip(
                cols,
                row_items,
            ):
                with col:
                    render_aspect_card(
                        result
                    )

    st.markdown(
        "<div style='height:.75rem'></div>",
        unsafe_allow_html=True,
    )

    render_charts(
        results_df
    )



# ============================================================
# REUSABLE REVIEW INPUT COMPONENT
# ============================================================

INPUT_WARNING = (
    "Please enter review or upload a review file first!"
)


def render_review_input(
    key_prefix: str,
    *,
    text_label: str = "Customer review",
    text_button_label: str = "Analyze Text",
    file_button_label: str = "Analyze Uploaded Reviews",
    text_height: int = 125,
    show_file_preview: bool = True,
):
    """
    Render the shared text/file analysis interface.

    This component is used by both Overview and Analyze so that
    input handling, file parsing, validation and analysis execution
    remain consistent across the application.
    """
    text_tab, file_tab = st.tabs(
        [
            "Analyze Text",
            "Upload Review File",
        ]
    )

    # --------------------------------------------------------
    # TEXT INPUT
    # --------------------------------------------------------
    with text_tab:
        review_text = st.text_area(
            text_label,
            value="",
            height=text_height,
            placeholder="Enter reviews here",
            key=f"{key_prefix}_text",
        )

        button_col, warning_col = st.columns(
            [1.25, 4.75]
        )

        with button_col:
            text_clicked = st.button(
                text_button_label,
                type="primary",
                key=f"{key_prefix}_run_text",
                use_container_width=True,
            )

        with warning_col:
            if text_clicked:
                if not review_text.strip():
                    st.warning(
                        INPUT_WARNING
                    )
                else:
                    analyse_reviews(
                        segment_input_text(
                            review_text
                        )
                    )
                    st.rerun()

    # --------------------------------------------------------
    # FILE INPUT
    # --------------------------------------------------------
    with file_tab:
        uploaded_file = st.file_uploader(
            "Upload TXT, CSV or XML",
            type=[
                "txt",
                "csv",
                "xml",
            ],
            key=f"{key_prefix}_file",
        )

        reviews = []
        file_ready = False

        if uploaded_file is not None:
            try:
                (
                    reviews,
                    file_type,
                    text_column,
                ) = read_uploaded_file(
                    uploaded_file
                )

                st.success(
                    f"Loaded {len(reviews):,} "
                    f"review(s) from {file_type}."
                )

                if text_column:
                    st.caption(
                        f"Detected review column: "
                        f"`{text_column}`"
                    )

                if reviews:
                    file_ready = True

                    if show_file_preview:
                        st.dataframe(
                            pd.DataFrame(
                                {
                                    "Review":
                                        reviews[:10]
                                }
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

            except Exception as exc:
                st.error(
                    "Could not process the uploaded file."
                )
                st.exception(
                    exc
                )

        button_col, warning_col = st.columns(
            [1.45, 4.55]
        )

        with button_col:
            file_clicked = st.button(
                file_button_label,
                type="primary",
                key=f"{key_prefix}_run_file",
                use_container_width=True,
            )

        with warning_col:
            if file_clicked:
                if (
                    uploaded_file is None
                    or not file_ready
                ):
                    st.warning(
                        INPUT_WARNING
                    )
                else:
                    analyse_reviews(
                        reviews
                    )
                    st.rerun()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        """
<div class="sidebar-brand">
    <div class="logo-box">A</div>
    <div>
        <div class="logo-name">AspectIQ</div>
        <div class="logo-sub">Customer Experience Analytics</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    navigation_groups = [
        (
            "OVERVIEW",
            [
                ("Overview", "Overview"),
            ],
        ),
        (
            "ANALYSIS",
            [
                ("Analyze", "Analyze"),
                ("Insights", "Insights"),
            ],
        ),
        (
            "DATA & METHODS",
            [
                ("Data Explorer", "Data Explorer"),
                ("Preprocessing", "Preprocessing"),
            ],
        ),
        (
            "MODEL & RESEARCH",
            [
                ("Model Lab", "Model Lab"),
                ("Methodology", "Methodology"),
            ],
        ),
    ]

    for group_name, nav_items in navigation_groups:

        st.markdown(
            f'<div class="nav-section-label">{group_name}</div>',
            unsafe_allow_html=True,
        )

        for label, destination in nav_items:
            is_active = (
                st.session_state.page
                == destination
            )

            if st.button(
                label,
                key=f"nav_{destination}",
                type=(
                    "primary"
                    if is_active
                    else "secondary"
                ),
                use_container_width=True,
            ):
                st.session_state.page = destination
                st.rerun()

    st.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True,
    )
# ============================================================
# PAGE — OVERVIEW
# ============================================================

if st.session_state.page == "Overview":

    render_hero()

    with st.container(
        key="quick_card"
    ):
        st.markdown(
            '<div class="section-kicker">Quick Analysis</div>'
            '<div class="section-title">Text or review file</div>',
            unsafe_allow_html=True,
        )

        render_review_input(
            "overview",
            text_label="Review text",
            text_button_label="Analyze Review  →",
            file_button_label="Analyze Uploaded Reviews",
            text_height=108,
            show_file_preview=True,
        )

    if st.session_state.analyses:

        render_results(
            st.session_state.analyses,
            st.session_state.results_df,
        )

    else:
        placeholder_values = [
            (
                "🧾",
                "i-blue",
                "Text Units Analyzed",
                "—",
            ),
            (
                "☷",
                "i-purple",
                "Aspects Detected",
                "—",
            ),
            (
                "💻",
                "i-teal",
                "Detected Domain",
                "—",
            ),
            (
                "◉",
                "i-orange",
                "Overall Sentiment",
                "—",
            ),
        ]

        cols = st.columns(4)

        for col, item in zip(
            cols,
            placeholder_values,
        ):
            with col:
                st.markdown(
                    kpi_html(*item),
                    unsafe_allow_html=True,
                )


# ============================================================
# PAGE — ANALYZE
# ============================================================

elif st.session_state.page == "Analyze":

    render_hero()

    # --------------------------------------------------------
    # If analysis exists, put the insights immediately below
    # the banner so they are visible without scrolling past
    # the input form again.
    # --------------------------------------------------------

    if st.session_state.analyses:

        st.markdown(
            '<div class="section-kicker">Analysis Results</div>'
            '<div class="section-title">Customer sentiment insights</div>',
            unsafe_allow_html=True,
        )

        render_results(
            st.session_state.analyses,
            st.session_state.results_df,
        )

        action_col, download_col, spacer = st.columns(
            [1.25, 1.45, 3.3]
        )

        with action_col:
            if st.button(
                "New Analysis",
                key="new_analysis_btn",
                use_container_width=True,
            ):
                st.session_state.analyses = []
                st.session_state.results_df = pd.DataFrame()
                st.rerun()

        with download_col:
            if not st.session_state.results_df.empty:
                st.download_button(
                    "Download Results",
                    data=(
                        st.session_state.results_df
                        .to_csv(index=False)
                        .encode("utf-8")
                    ),
                    file_name="aspectiq_analysis_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with st.expander(
            "Run another analysis",
            expanded=False,
        ):
            render_review_input(
                "analyze_after_results",
                text_label="Customer review",
                text_button_label="Analyze New Text",
                file_button_label="Analyze Uploaded Reviews",
                text_height=125,
                show_file_preview=False,
            )

    # --------------------------------------------------------
    # Initial state: show the analysis form prominently.
    # --------------------------------------------------------

    else:

        with st.container(
            key="analyze_card"
        ):
            st.markdown(
                '<div class="section-kicker">Analyze</div>'
                '<div class="section-title">Text or review file</div>',
                unsafe_allow_html=True,
            )

            render_review_input(
                "analyze_initial",
                text_label="Customer review",
                text_button_label="Analyze Text",
                file_button_label="Analyze Uploaded Reviews",
                text_height=125,
                show_file_preview=True,
            )


# ============================================================
# PAGE — INSIGHTS
# ============================================================

elif st.session_state.page == "Insights":

    st.markdown(
        '<div class="section-kicker">Executive Insights</div>'
        '<div class="section-title">Customer sentiment intelligence</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.results_df.empty:

        st.info(
            "Analyze text or upload reviews first. "
            "Your customer insights will appear here."
        )

    else:

        render_kpis(
            st.session_state.analyses,
            st.session_state.results_df,
        )

        st.markdown(
            "<div style='height:1rem'></div>",
            unsafe_allow_html=True,
        )

        render_charts(
            st.session_state.results_df
        )

        st.markdown(
            '<div class="section-title" style="margin-top:1.2rem;">'
            'Detailed Aspect Analysis'
            '</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            st.session_state.results_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# PAGE — DATA EXPLORER
# ============================================================

elif st.session_state.page == "Data Explorer":

    render_hero()

    st.markdown(
        '<div class="section-kicker">Data Explorer</div>'
        '<div class="section-title">Current analysis data</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.analyses:
        st.info(
            "No user data is available yet. Enter review text or upload a "
            "file on the Analyze page, run the analysis, and the submitted "
            "data will appear here."
        )

        if st.button(
            "Go to Analyze",
            type="primary",
            key="data_explorer_go_analyze",
        ):
            st.session_state.page = "Analyze"
            st.rerun()

    else:
        review_df = build_submitted_reviews_dataframe(
            st.session_state.analyses
        )

        aspect_df = st.session_state.results_df.copy()

        metric_cols = st.columns(4)

        with metric_cols[0]:
            st.metric(
                "Reviews Submitted",
                f"{len(review_df):,}",
            )

        with metric_cols[1]:
            st.metric(
                "Aspects Detected",
                (
                    f"{len(aspect_df):,}"
                    if not aspect_df.empty
                    else "0"
                ),
            )

        with metric_cols[2]:
            st.metric(
                "Domains Detected",
                (
                    review_df[
                        "Detected Domain"
                    ].nunique()
                    if not review_df.empty
                    else 0
                ),
            )

        with metric_cols[3]:
            st.metric(
                "Unique Aspects",
                (
                    aspect_df[
                        "Aspect"
                    ].nunique()
                    if (
                        not aspect_df.empty
                        and "Aspect"
                        in aspect_df.columns
                    )
                    else 0
                ),
            )

        review_tab, aspect_tab = st.tabs(
            [
                "Submitted Reviews",
                "Aspect-Level Analysis",
            ]
        )

        with review_tab:
            st.caption(
                "This table contains the text supplied by the user, whether "
                "entered directly or loaded from an uploaded file."
            )

            domain_values = sorted(
                review_df[
                    "Detected Domain"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            filter_col1, filter_col2 = st.columns(
                [1, 2]
            )

            with filter_col1:
                selected_domains = st.multiselect(
                    "Detected Domain",
                    domain_values,
                    default=domain_values,
                    key="submitted_review_domains",
                )

            with filter_col2:
                review_search = st.text_input(
                    "Search review text",
                    placeholder=(
                        "Search words or phrases in the submitted reviews"
                    ),
                    key="submitted_review_search",
                )

            filtered_reviews = review_df.copy()

            if selected_domains:
                filtered_reviews = filtered_reviews[
                    filtered_reviews[
                        "Detected Domain"
                    ]
                    .astype(str)
                    .isin(
                        selected_domains
                    )
                ]

            if review_search.strip():
                filtered_reviews = filtered_reviews[
                    filtered_reviews[
                        "Review"
                    ]
                    .astype(str)
                    .str.contains(
                        review_search.strip(),
                        case=False,
                        na=False,
                    )
                ]

            st.caption(
                f"Showing {len(filtered_reviews):,} of "
                f"{len(review_df):,} submitted review(s)."
            )

            st.dataframe(
                filtered_reviews,
                use_container_width=True,
                hide_index=True,
            )

            if not filtered_reviews.empty:
                domain_counts = (
                    filtered_reviews[
                        "Detected Domain"
                    ]
                    .value_counts()
                    .rename_axis("Domain")
                    .reset_index(name="Reviews")
                )

                fig = px.bar(
                    domain_counts,
                    x="Domain",
                    y="Reviews",
                    text="Reviews",
                    title="Detected Domain Distribution",
                )

                fig.update_traces(
                    marker_color="#2C6FF2"
                )

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    },
                )

        with aspect_tab:
            if aspect_df.empty:
                st.warning(
                    "The submitted review data is available, but no aspects "
                    "were detected for the current analysis."
                )

            else:
                filter_columns = st.columns(3)

                with filter_columns[0]:
                    domain_values = sorted(
                        aspect_df[
                            "Detected Domain"
                        ]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    selected_domains = st.multiselect(
                        "Domain",
                        domain_values,
                        default=domain_values,
                        key="aspect_data_domains",
                    )

                with filter_columns[1]:
                    sentiment_values = sorted(
                        aspect_df[
                            "Sentiment"
                        ]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    selected_sentiments = st.multiselect(
                        "Sentiment",
                        sentiment_values,
                        default=sentiment_values,
                        key="aspect_data_sentiments",
                    )

                with filter_columns[2]:
                    aspect_search = st.text_input(
                        "Aspect contains",
                        placeholder=(
                            "e.g. screen, battery, service"
                        ),
                        key="aspect_data_search",
                    )

                filtered_aspects = aspect_df.copy()

                if selected_domains:
                    filtered_aspects = filtered_aspects[
                        filtered_aspects[
                            "Detected Domain"
                        ]
                        .astype(str)
                        .isin(
                            selected_domains
                        )
                    ]

                if selected_sentiments:
                    filtered_aspects = filtered_aspects[
                        filtered_aspects[
                            "Sentiment"
                        ]
                        .astype(str)
                        .isin(
                            selected_sentiments
                        )
                    ]

                if aspect_search.strip():
                    filtered_aspects = filtered_aspects[
                        filtered_aspects[
                            "Aspect"
                        ]
                        .astype(str)
                        .str.contains(
                            aspect_search.strip(),
                            case=False,
                            na=False,
                        )
                    ]

                st.caption(
                    f"Showing {len(filtered_aspects):,} of "
                    f"{len(aspect_df):,} detected aspect result(s)."
                )

                st.dataframe(
                    filtered_aspects,
                    use_container_width=True,
                    hide_index=True,
                )

                if not filtered_aspects.empty:
                    chart_left, chart_right = st.columns(2)

                    with chart_left:
                        sentiment_counts = (
                            filtered_aspects[
                                "Sentiment"
                            ]
                            .value_counts()
                            .rename_axis("Sentiment")
                            .reset_index(name="Count")
                        )

                        fig = px.bar(
                            sentiment_counts,
                            x="Sentiment",
                            y="Count",
                            text="Count",
                            title="Current Sentiment Distribution",
                            color="Sentiment",
                            color_discrete_map={
                                "Positive": "#23A447",
                                "Negative": "#E23939",
                                "Neutral": "#F59E0B",
                            },
                        )

                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={
                                "displayModeBar": False
                            },
                        )

                    with chart_right:
                        top_aspects = (
                            filtered_aspects[
                                "Aspect"
                            ]
                            .dropna()
                            .astype(str)
                            .value_counts()
                            .head(12)
                            .rename_axis("Aspect")
                            .reset_index(name="Count")
                        )

                        fig = px.bar(
                            top_aspects,
                            x="Count",
                            y="Aspect",
                            orientation="h",
                            text="Count",
                            title="Current Detected Aspects",
                        )

                        fig.update_traces(
                            marker_color="#2C6FF2"
                        )

                        fig.update_layout(
                            yaxis={
                                "categoryorder":
                                    "total ascending"
                            },
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={
                                "displayModeBar": False
                            },
                        )

        # Training data is retained only as an optional academic reference.
        with st.expander(
            "Reference: SemEval training corpus",
            expanded=False,
        ):
            training_df = load_training_corpus()

            if training_df.empty:
                st.caption(
                    "Training corpus is not available in this deployment."
                )
            else:
                st.caption(
                    "This is the model-development reference dataset, not "
                    "the user's current analysis data."
                )

                st.dataframe(
                    training_df.head(50),
                    use_container_width=True,
                    hide_index=True,
                )


# ============================================================
# PAGE — PREPROCESSING
# ============================================================

elif st.session_state.page == "Preprocessing":

    render_hero()

    st.markdown(
        '<div class="section-kicker">Preprocessing</div>'
        '<div class="section-title">Current input transformation</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.analyses:
        st.info(
            "No submitted text is available yet. Enter text or upload a "
            "review file on the Analyze page and run the analysis first."
        )

        if st.button(
            "Go to Analyze",
            type="primary",
            key="preprocess_go_analyze",
        ):
            st.session_state.page = "Analyze"
            st.rerun()

    else:
        preprocessing_df = build_preprocessing_dataset(
            st.session_state.analyses
        )

        st.caption(
            "The preprocessing shown below is applied to the text that was "
            "actually entered or uploaded by the user in the current analysis."
        )

        if len(preprocessing_df) > 1:
            review_options = preprocessing_df[
                "Review ID"
            ].tolist()

            selected_review_id = st.selectbox(
                "Select submitted review",
                review_options,
                format_func=lambda value: (
                    f"Review {value}"
                ),
                key="preprocessing_review_selector",
            )
        else:
            selected_review_id = preprocessing_df[
                "Review ID"
            ].iloc[0]

        selected_row = preprocessing_df[
            preprocessing_df[
                "Review ID"
            ]
            == selected_review_id
        ].iloc[0]

        preprocessing_text = selected_row[
            "Original Text"
        ]

        steps = instructional_preprocessing_steps(
            preprocessing_text
        )

        final_model_text = selected_row[
            "Model-Ready Text"
        ]

        summary_cols = st.columns(3)

        with summary_cols[0]:
            st.metric(
                "Selected Review",
                selected_review_id,
            )

        with summary_cols[1]:
            st.metric(
                "Original Tokens",
                len(
                    steps[
                        "Tokens"
                    ]
                ),
            )

        with summary_cols[2]:
            model_tokens = (
                final_model_text.split()
                if final_model_text
                else []
            )

            st.metric(
                "Model-Ready Tokens",
                len(
                    model_tokens
                ),
            )

        st.markdown(
            '<div class="section-title" style="margin-top:1rem;">'
            'Transformation stages'
            '</div>',
            unsafe_allow_html=True,
        )

        step_rows = [
            {
                "Stage": "1. Original Text",
                "Output": steps[
                    "Original Text"
                ],
            },
            {
                "Stage": "2. Lowercase",
                "Output": steps[
                    "Lowercase"
                ],
            },
            {
                "Stage": (
                    "3. HTML / URL Normalisation"
                ),
                "Output": steps[
                    "HTML / URL Normalisation"
                ],
            },
            {
                "Stage": "4. Tokenisation",
                "Output": " | ".join(
                    steps[
                        "Tokens"
                    ]
                ),
            },
            {
                "Stage": (
                    "5. Stopword Inspection"
                ),
                "Output": " | ".join(
                    steps[
                        "Stopword-Filtered Tokens"
                    ]
                ),
            },
            {
                "Stage": (
                    "6. Final Model Preprocessing"
                ),
                "Output": final_model_text,
            },
        ]

        st.dataframe(
            pd.DataFrame(
                step_rows
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            '<div class="section-title" style="margin-top:1rem;">'
            'Before and after'
            '</div>',
            unsafe_allow_html=True,
        )

        before_col, after_col = st.columns(2)

        with before_col:
            st.text_area(
                "Submitted text",
                value=preprocessing_text,
                height=145,
                disabled=True,
                key=(
                    f"preprocess_original_"
                    f"{selected_review_id}"
                ),
            )

        with after_col:
            st.text_area(
                "Model-ready text",
                value=final_model_text,
                height=145,
                disabled=True,
                key=(
                    f"preprocess_final_"
                    f"{selected_review_id}"
                ),
            )

        token_col, filtered_col = st.columns(2)

        with token_col:
            st.markdown(
                "#### Tokens"
            )

            st.write(
                steps[
                    "Tokens"
                ]
            )

        with filtered_col:
            st.markdown(
                "#### Stopword inspection"
            )

            st.write(
                steps[
                    "Stopword-Filtered Tokens"
                ]
            )

        st.caption(
            "Negation, contrast and intensifier terms are preserved in the "
            "inspection view because they can materially change sentiment."
        )

        if len(preprocessing_df) > 1:
            st.markdown(
                '<div class="section-title" style="margin-top:1.2rem;">'
                'All submitted reviews after preprocessing'
                '</div>',
                unsafe_allow_html=True,
            )

            st.dataframe(
                preprocessing_df,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# PAGE — MODEL LAB
# ============================================================

elif st.session_state.page == "Model Lab":

    st.markdown(
        '<div class="section-kicker">Model Lab</div>'
        '<div class="section-title">Final model evaluation and representation experiments</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # FINAL PRODUCTION MODEL
    # --------------------------------------------------------
    production_model_file = (
        MODEL_DIR
        / "best_target_clause_local_3class.json"
    )

    production_model = {
        "task": "3-class",
        "model": "SVM",
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "class_weight": "balanced",
        "representation": "Target Clause + Local Context GloVe",
        "dimensions": 200,
        "context_window": 5,
        "contrast_aware": True,
        "distance_weighted_local_context": True,
        "selection_evidence": {
            "grouped_cv_accuracy": 0.679165,
            "grouped_cv_macro_f1": 0.630693,
            "grouped_cv_macro_roc_auc_ovr": 0.816772,
            "selection_basis": (
                "Highest grouped 5-fold training-only Macro F1 "
                "among tested target-context representations."
            ),
        },
    }

    if production_model_file.exists():
        try:
            with open(
                production_model_file,
                "r",
                encoding="utf-8",
            ) as file:
                production_model.update(
                    json.load(file)
                )
        except Exception as exc:
            st.warning(
                "Production model metadata could not be read. "
                "AspectIQ is showing the frozen development metadata instead."
            )
            st.caption(str(exc))

    selection_evidence = production_model.get(
        "selection_evidence",
        {},
    )

    grouped_accuracy = float(
        selection_evidence.get(
            "grouped_cv_accuracy",
            0.679165,
        )
    )

    grouped_macro_f1 = float(
        selection_evidence.get(
            "grouped_cv_macro_f1",
            0.630693,
        )
    )

    grouped_auc = float(
        selection_evidence.get(
            "grouped_cv_macro_roc_auc_ovr",
            0.816772,
        )
    )

    feature_dimensions = int(
        production_model.get(
            "dimensions",
            200,
        )
    )

    metrics = [
        (
            "🎯",
            "i-blue",
            "Grouped CV Accuracy",
            f"{grouped_accuracy * 100:.2f}%",
        ),
        (
            "📐",
            "i-purple",
            "Grouped CV Macro F1",
            f"{grouped_macro_f1:.3f}",
        ),
        (
            "📈",
            "i-teal",
            "Grouped CV ROC-AUC",
            f"{grouped_auc:.3f}",
        ),
        (
            "🧠",
            "i-orange",
            "Feature Dimensions",
            feature_dimensions,
        ),
    ]

    cols = st.columns(4)

    for col, item in zip(
        cols,
        metrics,
    ):
        with col:
            st.markdown(
                kpi_html(*item),
                unsafe_allow_html=True,
            )

    st.caption(
        "These are training-only grouped 5-fold cross-validation metrics "
        "used to select the final production representation. They are not "
        "presented as a new independent official-test result."
    )

    st.markdown(
        f"""
<div class="panel-shell" style="margin-top:1rem;">
    <div class="panel-title">Selected Production Model</div>
    <div style="line-height:1.85;color:#4D5970;">
        <b>Task:</b> 3-class aspect sentiment classification<br>
        <b>Classifier:</b> {html.escape(str(production_model.get("model", "SVM")))}
        ({html.escape(str(production_model.get("kernel", "rbf")).upper())} kernel)<br>
        <b>Representation:</b> {html.escape(str(production_model.get("representation", "Target Clause + Local Context GloVe")))}<br>
        <b>Dimensions:</b> {feature_dimensions}<br>
        <b>Context window:</b> {int(production_model.get("context_window", 5))}<br>
        <b>Class weighting:</b> {html.escape(str(production_model.get("class_weight", "balanced")))}<br>
        <b>Hyperparameters:</b> C={production_model.get("C", 1.0)},
        gamma={html.escape(str(production_model.get("gamma", "scale")))}<br>
        <b>Selection metric:</b> Grouped 5-fold CV Macro F1
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # REPRESENTATION EXPERIMENT
    # --------------------------------------------------------
    representation_file = (
        TARGET_CONTEXT_RESULTS_DIR
        / "representation_comparison.csv"
    )

    fallback_representation_results = pd.DataFrame(
        [
            {
                "representation": "Target Clause+Local 200d",
                "dimensions": 200,
                "accuracy_oof": 0.679165,
                "f1_macro_oof": 0.630693,
                "roc_auc_macro_ovr_oof": 0.816772,
            },
            {
                "representation": "Target Clause+Aspect 200d",
                "dimensions": 200,
                "accuracy_oof": 0.674542,
                "f1_macro_oof": 0.627110,
                "roc_auc_macro_ovr_oof": 0.813642,
            },
            {
                "representation": "Target Clause+Aspect+Local 300d",
                "dimensions": 300,
                "accuracy_oof": 0.672830,
                "f1_macro_oof": 0.626107,
                "roc_auc_macro_ovr_oof": 0.816958,
            },
            {
                "representation": "Current Full+Aspect+Local 300d",
                "dimensions": 300,
                "accuracy_oof": 0.670604,
                "f1_macro_oof": 0.623990,
                "roc_auc_macro_ovr_oof": 0.816268,
            },
            {
                "representation": "Local Context Only 100d",
                "dimensions": 100,
                "accuracy_oof": 0.620271,
                "f1_macro_oof": 0.578604,
                "roc_auc_macro_ovr_oof": 0.775565,
            },
            {
                "representation": "Aspect+Local 200d",
                "dimensions": 200,
                "accuracy_oof": 0.614621,
                "f1_macro_oof": 0.575528,
                "roc_auc_macro_ovr_oof": 0.770744,
            },
        ]
    )

    if representation_file.exists():
        try:
            representation_df = pd.read_csv(
                representation_file
            )
        except Exception:
            representation_df = (
                fallback_representation_results.copy()
            )
    else:
        representation_df = (
            fallback_representation_results.copy()
        )

    required_representation_columns = {
        "representation",
        "dimensions",
        "accuracy_oof",
        "f1_macro_oof",
        "roc_auc_macro_ovr_oof",
    }

    if not required_representation_columns.issubset(
        representation_df.columns
    ):
        representation_df = (
            fallback_representation_results.copy()
        )

    representation_df = (
        representation_df
        .sort_values(
            "f1_macro_oof",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    display_representation_df = (
        representation_df[
            [
                "representation",
                "dimensions",
                "accuracy_oof",
                "f1_macro_oof",
                "roc_auc_macro_ovr_oof",
            ]
        ]
        .rename(
            columns={
                "representation": "Representation",
                "dimensions": "Dimensions",
                "accuracy_oof": "Grouped CV Accuracy",
                "f1_macro_oof": "Grouped CV Macro F1",
                "roc_auc_macro_ovr_oof": "Grouped CV ROC-AUC",
            }
        )
    )

    st.markdown(
        '<div class="section-title" style="margin-top:1.25rem;">'
        'Training-Only Representation Comparison'
        '</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        display_representation_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Grouped CV Accuracy": st.column_config.NumberColumn(
                format="%.3f"
            ),
            "Grouped CV Macro F1": st.column_config.NumberColumn(
                format="%.3f"
            ),
            "Grouped CV ROC-AUC": st.column_config.NumberColumn(
                format="%.3f"
            ),
        },
    )

    representation_figure = px.bar(
        representation_df.sort_values(
            "f1_macro_oof",
            ascending=True,
        ),
        x="f1_macro_oof",
        y="representation",
        orientation="h",
        text_auto=".3f",
        title="Grouped 5-Fold CV Macro F1 by GloVe Representation",
        labels={
            "f1_macro_oof": "Macro F1",
            "representation": "Representation",
        },
    )

    representation_figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title=None,
        xaxis_title="Grouped CV Macro F1",
    )

    st.plotly_chart(
        representation_figure,
        use_container_width=True,
        config={
            "displayModeBar": False
        },
    )

    st.markdown(
        """
<div class="panel-shell" style="margin-top:1rem;">
    <div class="panel-title">Development Conclusion</div>
    <p style="color:#4D5970;line-height:1.75;margin-bottom:0;">
        The Target Clause + Local Context representation achieved the highest
        grouped cross-validation Macro F1 while reducing the sentiment feature
        space from 300 to 200 dimensions. The design removes the global
        full-sentence vector that was found to allow sentiment from a different
        aspect clause to influence the target aspect. It also avoids a standalone
        aspect vector, which did not improve grouped-CV performance.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # HISTORICAL INDEPENDENT BENCHMARK
    # --------------------------------------------------------
    prior_metrics = {
        "accuracy": 0.7031,
        "precision_macro": 0.6219,
        "recall_macro": 0.6437,
        "f1_macro": 0.6275,
        "roc_auc_macro_ovr": 0.8345,
    }

    prior_metrics_file = (
        PRIOR_EVALUATION_DIR
        / "official_test_metrics.json"
    )

    if prior_metrics_file.exists():
        try:
            with open(
                prior_metrics_file,
                "r",
                encoding="utf-8",
            ) as file:
                prior_metrics.update(
                    json.load(file)
                )
        except Exception:
            pass

    with st.expander(
        "Earlier independent-test benchmark — historical 300d model",
        expanded=False,
    ):
        st.caption(
            "This benchmark belongs to the earlier Contrast-Aware Enhanced "
            "GloVe 300d SVM. It is retained for research transparency and "
            "must not be interpreted as the independent-test score of the "
            "current 200d production representation."
        )

        historical_cols = st.columns(5)

        historical_values = [
            (
                "Accuracy",
                prior_metrics.get(
                    "accuracy",
                    0.7031,
                ),
            ),
            (
                "Precision",
                prior_metrics.get(
                    "precision_macro",
                    0.6219,
                ),
            ),
            (
                "Recall",
                prior_metrics.get(
                    "recall_macro",
                    0.6437,
                ),
            ),
            (
                "Macro F1",
                prior_metrics.get(
                    "f1_macro",
                    0.6275,
                ),
            ),
            (
                "ROC-AUC",
                prior_metrics.get(
                    "roc_auc_macro_ovr",
                    0.8345,
                ),
            ),
        ]

        for col, (
            label,
            value,
        ) in zip(
            historical_cols,
            historical_values,
        ):
            with col:
                if label == "Accuracy":
                    st.metric(
                        label,
                        f"{float(value) * 100:.2f}%",
                    )
                else:
                    st.metric(
                        label,
                        f"{float(value):.3f}",
                    )


# ============================================================
# PAGE — METHODOLOGY
# ============================================================

else:

    st.markdown(
        '<div class="section-kicker">Methodology</div>'
        '<div class="section-title">How AspectIQ works</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="panel-shell">
    <div class="panel-title">Automatic ABSA Pipeline</div>
    <div style="
        display:grid;
        grid-template-columns:repeat(5,minmax(0,1fr));
        gap:.75rem;
        margin-top:1rem;">
        <div class="kpi-card" style="text-align:center;">
            <div style="font-size:1.55rem;">📝</div>
            <b>Text / File Input</b>
            <div class="kpi-label">TXT, CSV, XML or direct text</div>
        </div>
        <div class="kpi-card" style="text-align:center;">
            <div style="font-size:1.55rem;">🧭</div>
            <b>Domain Detection</b>
            <div class="kpi-label">Laptop or Restaurant</div>
        </div>
        <div class="kpi-card" style="text-align:center;">
            <div style="font-size:1.55rem;">🔎</div>
            <b>Aspect Detection</b>
            <div class="kpi-label">Explicit aspect extraction</div>
        </div>
        <div class="kpi-card" style="text-align:center;">
            <div style="font-size:1.55rem;">🧠</div>
            <b>200d Target Context</b>
            <div class="kpi-label">Target clause + weighted local context</div>
        </div>
        <div class="kpi-card" style="text-align:center;">
            <div style="font-size:1.55rem;">😊</div>
            <b>SVM Sentiment</b>
            <div class="kpi-label">Positive, Negative, Neutral</div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="panel-shell" style="margin-top:1rem;">
    <div class="panel-title">Final Sentiment Model</div>
    <p style="color:#4D5970;line-height:1.75;">
        AspectIQ uses a 200-dimensional target-context GloVe representation.
        The first 100 dimensions represent the contrast-delimited clause
        containing the target aspect. The second 100 dimensions represent a
        distance-weighted local context around that aspect within the same
        clause. Sentiment is classified with a balanced RBF Support Vector
        Machine using C=1.0 and gamma='scale'.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="panel-shell" style="margin-top:1rem;">
    <div class="panel-title">Why Target-Context Representation?</div>
    <p style="color:#4D5970;line-height:1.75;">
        Aspect-based sentiment analysis must distinguish opinions expressed
        about different targets in the same sentence. For example, in
        <em>"The screen is beautiful but the battery life is disappointing"</em>,
        the sentiment for <em>screen</em> is positive while the sentiment for
        <em>battery life</em> is negative. The target-clause boundary prevents
        the positive screen clause from entering the battery-life
        representation. The weighted local-context vector then emphasizes
        words nearest to the target aspect.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="panel-shell" style="margin-top:1rem;">
    <div class="panel-title">Model Selection</div>
    <p style="color:#4D5970;line-height:1.75;">
        Alternative GloVe representations were compared using grouped
        five-fold cross-validation on the three-class training corpus.
        Target Clause + Local Context (200d) produced the highest grouped-CV
        Macro F1 (0.631) and accuracy (67.92%), compared with Macro F1 0.624
        and accuracy 67.06% for the earlier Full Sentence + Aspect + Local
        Context (300d) representation. The final classifier was therefore
        retrained on all 5,841 eligible three-class training observations.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="panel-shell" style="margin-top:1rem;">
    <div class="panel-title">Evaluation Interpretation</div>
    <p style="color:#4D5970;line-height:1.75;">
        The current 200d production representation was selected using
        training-only grouped cross-validation. An earlier 300d SVM had
        already been evaluated on the official SemEval test set; its
        independent-test scores are therefore retained only as a historical
        benchmark. This separation avoids presenting those earlier test
        results as if they belonged to the newly selected 200d production
        model.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="panel-shell" style="margin-top:1rem;">
    <div class="panel-title">Automatic Front-End Intelligence</div>
    <p style="color:#4D5970;line-height:1.75;">
        The user provides only review text or a file. AspectIQ detects explicit
        aspect terms automatically using the SemEval training aspect lexicon
        with a noun-phrase fallback, and infers the review domain using aspect
        evidence together with domain-specific vocabulary learned from the
        Laptop and Restaurant training corpora. These auxiliary detectors
        support routing and aspect discovery; the reported sentiment-model
        metrics apply to the SVM sentiment classifier rather than to these
        front-end heuristics.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

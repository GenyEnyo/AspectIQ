from __future__ import annotations

from pathlib import Path
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
    reviews = []

    for value in values:
        if pd.isna(value):
            continue

        text = str(value).strip()

        if text:
            reviews.append(text)

    return reviews


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
            [
                line.strip()
                for line in content.splitlines()
                if line.strip()
            ],
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

        return texts, "XML", None

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
        r"https?://\\S+|www\\.\\S+",
        " ",
        no_html,
    )

    normalized = re.sub(
        r"\\s+",
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

    domains = [
        item["domain"]["domain"]
        for item in analyses
    ]

    if len(set(domains)) == 1:
        return domains[0]

    return "Mixed"


def render_kpis(
    analyses,
    results_df,
):
    values = [
        (
            "🧾",
            "i-blue",
            "Reviews Analyzed",
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

    aspect_results = [
        result
        for analysis in analyses
        for result in analysis[
            "sentiments"
        ]
    ]

    if not aspect_results:
        st.warning(
            "No explicit aspect terms were detected."
        )
        return

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
            '<div class="section-title">Paste customer review</div>',
            unsafe_allow_html=True,
        )

        quick_text = st.text_area(
            "Review text",
            value=(
                "The screen is beautiful but "
                "the battery life is disappointing."
            ),
            height=108,
            label_visibility="collapsed",
            key="overview_text",
        )

        button_col, caption_col = st.columns(
            [1.05, 4.7]
        )

        with button_col:
            if st.button(
                "Analyze Review  →",
                type="primary",
                key="overview_run",
                use_container_width=True,
            ):
                if quick_text.strip():
                    analyse_reviews(
                        [
                            quick_text.strip()
                        ]
                    )

        with caption_col:
            st.caption(
                "AspectIQ automatically determines the domain, "
                "detects explicit aspects, and classifies their sentiment."
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
                "Reviews Analyzed",
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
            text_tab, file_tab = st.tabs(
                [
                    "Analyze Text",
                    "Upload Review File",
                ]
            )

            with text_tab:
                text = st.text_area(
                    "Customer review",
                    value="",
                    height=125,
                    key="analyze_text_after_results",
                    placeholder=(
                        "Enter another laptop or restaurant review..."
                    ),
                )

                if st.button(
                    "Analyze New Text",
                    type="primary",
                    key="run_text_after_results",
                ):
                    if text.strip():
                        analyse_reviews([text.strip()])
                        st.rerun()

            with file_tab:
                uploaded_file = st.file_uploader(
                    "Upload TXT, CSV or XML",
                    type=["txt", "csv", "xml"],
                    key="file_after_results",
                )

                if uploaded_file is not None:
                    try:
                        reviews, file_type, text_column = read_uploaded_file(
                            uploaded_file
                        )

                        st.success(
                            f"Loaded {len(reviews):,} review(s) from {file_type}."
                        )

                        if text_column:
                            st.caption(
                                f"Detected review column: `{text_column}`"
                            )

                        if reviews and st.button(
                            "Analyze Uploaded Reviews",
                            type="primary",
                            key="run_file_after_results",
                        ):
                            analyse_reviews(reviews)
                            st.rerun()

                    except Exception as exc:
                        st.error(
                            "Could not process the uploaded file."
                        )
                        st.exception(exc)

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

            text_tab, file_tab = st.tabs(
                [
                    "Analyze Text",
                    "Upload Review File",
                ]
            )

            with text_tab:
                text = st.text_area(
                    "Customer review",
                    value=(
                        "The food was excellent but "
                        "the service was painfully slow."
                    ),
                    height=125,
                    key="analyze_text",
                )

                if st.button(
                    "Analyze Text",
                    type="primary",
                    key="run_text",
                ):
                    if text.strip():
                        analyse_reviews(
                            [text.strip()]
                        )
                        st.rerun()

            with file_tab:
                uploaded_file = st.file_uploader(
                    "Upload TXT, CSV or XML",
                    type=[
                        "txt",
                        "csv",
                        "xml",
                    ],
                    key="initial_file_upload",
                )

                if uploaded_file is not None:
                    try:
                        reviews, file_type, text_column = (
                            read_uploaded_file(
                                uploaded_file
                            )
                        )

                        st.success(
                            f"Loaded {len(reviews):,} review(s) from {file_type}."
                        )

                        if text_column:
                            st.caption(
                                f"Detected review column: `{text_column}`"
                            )

                        if reviews:
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

                            if st.button(
                                "Analyze Uploaded Reviews",
                                type="primary",
                                key="run_file",
                            ):
                                analyse_reviews(
                                    reviews
                                )
                                st.rerun()

                    except Exception as exc:
                        st.error(
                            "Could not process the uploaded file."
                        )
                        st.exception(exc)


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
        '<div class="section-title">Model evaluation and experiments</div>',
        unsafe_allow_html=True,
    )

    best_model_file = (
        RESULTS_DIR
        / "best_3_class_model.json"
    )

    if best_model_file.exists():

        with open(
            best_model_file,
            "r",
            encoding="utf-8",
        ) as file:
            best_model = json.load(
                file
            )

        metrics = [
            (
                "🎯",
                "i-blue",
                "Test Accuracy",
                f"{best_model['test_accuracy'] * 100:.2f}%",
            ),
            (
                "📐",
                "i-purple",
                "Macro F1",
                f"{best_model['test_macro_f1']:.3f}",
            ),
            (
                "📈",
                "i-teal",
                "Macro ROC-AUC",
                f"{best_model['test_macro_roc_auc']:.3f}",
            ),
            (
                "🧠",
                "i-orange",
                "Feature Dimensions",
                best_model[
                    "dimensions"
                ],
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

        st.markdown(
            f"""
<div class="panel-shell" style="margin-top:1rem;">
    <div class="panel-title">Selected Final Model</div>
    <div style="line-height:1.8;color:#4D5970;">
        <b>Model:</b> {best_model['model']}<br>
        <b>Representation:</b> {best_model['representation']}<br>
        <b>Class weighting:</b> {best_model['class_weight']}<br>
        <b>Selection metric:</b> {best_model['selection_metric']}
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    comparison_file = (
        RESULTS_DIR
        / "controlled_experiment_results.csv"
    )

    if comparison_file.exists():

        comparison_df = pd.read_csv(
            comparison_file
        )

        st.markdown(
            '<div class="section-title" style="margin-top:1.25rem;">'
            'Controlled Experiment Comparison'
            '</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            comparison_df.sort_values(
                [
                    "task",
                    "cv_macro_f1_mean",
                ],
                ascending=[
                    True,
                    False,
                ],
            ),
            use_container_width=True,
            hide_index=True,
        )

        three_class = (
            comparison_df[
                comparison_df["task"]
                == "3-class"
            ]
            .copy()
        )

        figure = px.bar(
            three_class,
            x="model",
            y="f1_macro",
            color="representation",
            barmode="group",
            text_auto=".3f",
            title="Three-Class Test Macro F1",
        )

        figure.update_layout(
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

    confusion_file = (
        FIGURE_DIR
        / (
            "cm_3_class_logistic_regression_"
            "enhanced_300d.png"
        )
    )

    if confusion_file.exists():

        st.markdown(
            '<div class="section-title">Final Model Confusion Matrix</div>',
            unsafe_allow_html=True,
        )

        st.image(
            str(
                confusion_file
            ),
            use_container_width=True,
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
        grid-template-columns:repeat(5,1fr);
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
            <b>300d GloVe</b>
            <div class="kpi-label">Sentence + aspect + context</div>
        </div>
        <div class="kpi-card" style="text-align:center;">
            <div style="font-size:1.55rem;">😊</div>
            <b>Sentiment</b>
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
        AspectIQ uses a 300-dimensional aspect-aware GloVe representation:
        100 dimensions for the full sentence, 100 dimensions for the target
        aspect, and 100 dimensions for its local context. Logistic Regression
        was selected as the final three-class sentiment classifier using
        training cross-validation Macro F1.
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
        Laptop and Restaurant training corpora.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

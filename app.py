from __future__ import annotations

from pathlib import Path
import json
import xml.etree.ElementTree as ET

import nltk
import pandas as pd
import plotly.express as px
import streamlit as st


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
# GLOBAL CSS — CLOSELY MATCHES THE APPROVED MOCKUP
# ============================================================

st.markdown(
    """
<style>
:root {
    --navy-950: #061126;
    --navy-900: #08152D;
    --navy-800: #0E1D3B;
    --blue: #1769FF;
    --blue-2: #2B7CFF;
    --page: #F7F9FD;
    --card: #FFFFFF;
    --line: #E5EAF2;
    --text: #172033;
    --muted: #6E788B;
    --green: #23A447;
    --green-bg: #F2FCF4;
    --red: #E23939;
    --red-bg: #FFF3F3;
    --orange: #F59E0B;
    --purple: #7C4DFF;
    --teal: #159A9A;
}

/* ---------- global ---------- */
html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", Arial, sans-serif;
}

.stApp {
    background: var(--page);
    overflow-x: hidden;
}

.block-container {
    max-width: 1500px;
    padding-top: 0 !important;
    padding-bottom: 3rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

@media (max-width: 900px) {
    .block-container {
        padding-left: .8rem !important;
        padding-right: .8rem !important;
        padding-bottom: 2rem !important;
    }
}

@media (max-width: 600px) {
    .block-container {
        padding-left: .55rem !important;
        padding-right: .55rem !important;
    }
}

[data-testid="stMainBlockContainer"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

main[data-testid="stMain"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

div[data-testid="stAppViewContainer"] > section.main {
    padding-top: 0 !important;
}

#MainMenu,
footer,
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
button[data-testid="stBaseButton-headerNoPadding"] {
    display: none !important;
}

header[data-testid="stHeader"] {
    height: 0 !important;
    min-height: 0 !important;
    background: transparent !important;
}

button[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #061126 0%, #091733 100%);
    border-right: 1px solid rgba(255,255,255,.06);
}

@media (min-width: 901px) {
    section[data-testid="stSidebar"] {
        width: 260px !important;
        min-width: 260px !important;
    }
}

@media (max-width: 900px) {
    section[data-testid="stSidebar"] {
        width: min(82vw, 300px) !important;
        min-width: 0 !important;
        max-width: 300px !important;
        box-shadow: 12px 0 36px rgba(3,12,30,.22);
    }
}

section[data-testid="stSidebar"] > div {
    padding: 1.35rem 1.05rem 1.2rem 1.05rem;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: .8rem;
    margin: .15rem .15rem 1.55rem .15rem;
}

.logo-box {
    width: 44px;
    height: 44px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 1.45rem;
    font-weight: 900;
    background: linear-gradient(145deg, #3987FF, #2D61F4);
    box-shadow: 0 10px 24px rgba(45,97,244,.35);
}

.logo-name {
    color: white;
    font-size: 1.35rem;
    font-weight: 850;
    line-height: 1.05;
}

.logo-sub {
    color: #8FA4C8;
    font-size: .72rem;
    margin-top: .2rem;
}

/* sidebar Streamlit buttons */
section[data-testid="stSidebar"] div[data-testid="stButton"] {
    margin-bottom: .42rem;
}

section[data-testid="stSidebar"] div[data-testid="stButton"] button {
    width: 100%;
    min-height: 46px;
    border-radius: 11px;
    justify-content: flex-start;
    padding-left: .9rem;
    font-weight: 650;
    box-shadow: none;
}

/* secondary nav */
section[data-testid="stSidebar"] button[kind="secondary"] {
    color: #E5ECF8 !important;
    background: transparent !important;
    border: 1px solid transparent !important;
}

section[data-testid="stSidebar"] button[kind="secondary"]:hover {
    color: white !important;
    background: rgba(255,255,255,.06) !important;
}

/* active nav */
section[data-testid="stSidebar"] button[kind="primary"] {
    color: white !important;
    background: linear-gradient(90deg, #1E62E8, #2A72F4) !important;
    border: 0 !important;
    box-shadow: 0 8px 18px rgba(31,105,242,.28) !important;
}

.sidebar-divider {
    border-top: 1px solid rgba(255,255,255,.10);
    margin: 1.35rem .2rem 1rem .2rem;
}


/* ---------- hero ---------- */
.st-key-hero_shell {
    width: 92% !important;
    max-width: 1180px !important;
    margin: 0 auto 1rem auto !important;
    background:
      radial-gradient(circle at 78% 28%, rgba(56,115,255,.18), transparent 23%),
      radial-gradient(circle at 91% 78%, rgba(27,108,255,.12), transparent 22%),
      linear-gradient(135deg, #071226 0%, #0B1733 55%, #071126 100%);
    border-radius: 0 0 18px 18px;
    padding: 1.35rem 1.7rem 1.3rem 1.7rem;
    box-shadow: 0 14px 34px rgba(14,29,58,.14);
    overflow: hidden;
}

.st-key-hero_shell .hero-kicker {
    color:#8EB5FF !important;
    font-size:.78rem;
    letter-spacing:.14em;
    text-transform:uppercase;
    font-weight:850;
    margin-bottom:.45rem;
    text-shadow: 0 1px 10px rgba(85,140,255,.22);
}

.st-key-hero_shell .hero-title {
    color:#FFFFFF !important;
    font-size:2.85rem;
    line-height:1;
    letter-spacing:-.035em;
    font-weight:850;
    margin:0;
}

.st-key-hero_shell .hero-subtitle {
    color:#D9E7FF !important;
    font-size:1.05rem;
    font-weight:650;
    margin-top:.45rem;
}

.st-key-hero_shell .hero-copy {
    color:#F1F5FF !important;
    line-height:1.5;
    font-size:.90rem;
    margin-top:.75rem;
    max-width:700px;
    font-weight:450;
}

.st-key-hero_actions button {
    min-height: 47px;
    border-radius: 12px;
    font-weight: 800;
}

/* blue CTA */
.st-key-hero_analyze button {
    background: linear-gradient(90deg, #1769FF, #2381FF) !important;
    color:white !important;
    border:0 !important;
    box-shadow:0 8px 18px rgba(23,105,255,.25);
}

/* ghost CTA */
.st-key-hero_upload button {
    background: rgba(255,255,255,.03) !important;
    color:#F2F6FF !important;
    border:1px solid rgba(255,255,255,.35) !important;
}

.hero-visual-card {
    margin-left:auto;
    width:88%;
    max-width:300px;
    min-height:155px;
    border-radius:20px;
    padding:1rem 1.1rem;
    background:linear-gradient(145deg, rgba(27,49,90,.96), rgba(14,32,65,.90));
    border:1px solid rgba(108,155,255,.22);
    box-shadow:0 16px 40px rgba(0,0,0,.25);
}

.st-key-hero_shell .hero-visual-title {
    font-size:.73rem;
    color:#FFFFFF !important;
    font-weight:850;
    letter-spacing:.04em;
}

.hero-bars {
    height:78px;
    display:flex;
    align-items:flex-end;
    gap:10px;
    padding:.7rem .25rem .25rem .25rem;
}

.hero-bars i {
    display:block;
    flex:1;
    border-radius:8px 8px 2px 2px;
    background:linear-gradient(180deg,#7FE4FF,#3A80FF);
    box-shadow:0 3px 14px rgba(62,141,255,.30);
}

.hero-emojis {
    text-align:right;
    font-size:1.45rem;
    letter-spacing:.35rem;
    margin-top:.55rem;
}

/* ---------- quick input card ---------- */
.st-key-quick_card,
.st-key-analyze_card {
    background:white;
    border:1px solid var(--line);
    border-radius:18px;
    padding:1.05rem 1.2rem 1rem 1.2rem;
    box-shadow:0 10px 30px rgba(26,42,71,.07);
    margin-bottom:1rem;
}

.section-kicker {
    color:#276EF1;
    font-size:.75rem;
    letter-spacing:.12em;
    text-transform:uppercase;
    font-weight:850;
}

.section-title {
    color:var(--text);
    font-size:1.52rem;
    font-weight:850;
    margin:.18rem 0 .75rem 0;
}

.stTextArea textarea {
    background:#FBFCFE !important;
    border:1px solid #DDE4EE !important;
    border-radius:12px !important;
    color:#243047 !important;
    box-shadow:none !important;
}

.stTextArea textarea:focus {
    border:1px solid #9CBDF8 !important;
    box-shadow:0 0 0 3px rgba(37,105,255,.08) !important;
}

/* buttons main */
div[data-testid="stButton"] button[kind="primary"] {
    background:linear-gradient(90deg,#1769FF,#2B7CFF);
    color:white;
    border:0;
    border-radius:12px;
    font-weight:800;
    min-height:45px;
    box-shadow:0 8px 18px rgba(23,105,255,.22);
}

/* ---------- KPI cards ---------- */
.kpi-card {
    background:white;
    border:1px solid var(--line);
    border-radius:16px;
    padding:1rem 1rem;
    box-shadow:0 7px 23px rgba(25,38,67,.055);
    min-height:86px;
}

.kpi-wrap {
    display:flex;
    align-items:center;
    gap:.85rem;
}

.kpi-icon {
    width:48px;
    height:48px;
    border-radius:13px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:1.25rem;
}

.i-blue {background:#EAF2FF;color:#2468E8;}
.i-purple {background:#F0E9FF;color:#7C4DFF;}
.i-teal {background:#E8F7F6;color:#159A9A;}
.i-orange {background:#FFF3DE;color:#F08B00;}

.kpi-label {
    color:var(--muted);
    font-size:.76rem;
    margin-bottom:.08rem;
}

.kpi-value {
    color:var(--text);
    font-size:1.33rem;
    line-height:1.1;
    font-weight:900;
}

/* ---------- aspect cards ---------- */
.aspect-card {
    border-radius:18px;
    padding:.9rem 1rem;
    min-height:136px;
    box-shadow:0 7px 22px rgba(25,38,67,.045);
}

.aspect-positive {
    background:linear-gradient(180deg,#FAFFFB,#F1FBF4);
    border:1.5px solid #79CB8C;
}

.aspect-negative {
    background:linear-gradient(180deg,#FFF9F9,#FFF1F1);
    border:1.5px solid #EF8C8C;
}

.aspect-neutral {
    background:linear-gradient(180deg,#FFFDF8,#FFF7E8);
    border:1.5px solid #F0C36E;
}

.aspect-top {
    display:flex;
    align-items:center;
    gap:.78rem;
}

.aspect-icon {
    width:53px;
    height:53px;
    border-radius:50%;
    display:flex;
    align-items:center;
    justify-content:center;
    background:white;
    box-shadow:0 4px 14px rgba(25,38,67,.08);
    font-size:1.4rem;
}

.aspect-name {
    font-size:1.15rem;
    font-weight:900;
    color:var(--text);
}

.aspect-pill {
    display:inline-block;
    margin-top:.25rem;
    padding:.26rem .58rem;
    border-radius:999px;
    font-size:.72rem;
    font-weight:850;
}

.p-positive {background:#DDF5E3;color:#17803A;}
.p-negative {background:#FFE0E0;color:#C92C2C;}
.p-neutral {background:#FFF0C7;color:#AC6900;}

.aspect-info {
    display:grid;
    grid-template-columns:1fr 1.35fr;
    gap:1rem;
    margin-top:.85rem;
}

.meta-label {
    color:var(--muted);
    font-size:.71rem;
}

.meta-value {
    color:var(--text);
    font-size:.9rem;
    font-weight:750;
}

.progress-track {
    height:7px;
    border-radius:999px;
    background:rgba(116,124,142,.18);
    margin-top:.35rem;
    overflow:hidden;
}

.progress-fill {
    height:100%;
    border-radius:999px;
}

/* ---------- panels ---------- */
.panel-shell {
    background:white;
    border:1px solid var(--line);
    border-radius:18px;
    padding:1rem 1rem .7rem 1rem;
    box-shadow:0 7px 22px rgba(25,38,67,.05);
}

.panel-title {
    color:var(--text);
    font-size:1rem;
    font-weight:850;
    margin-bottom:.35rem;
}

/* ---------- tables ---------- */
div[data-testid="stDataFrame"] {
    border:1px solid var(--line);
    border-radius:14px;
    overflow:hidden;
}

/* ---------- typography ---------- */
small, .caption {
    color:var(--muted);
}

@media (max-width: 900px) {
    .hero-title {font-size:2.6rem;}
    .hero-visual-card {display:none;}
}

/* Hero contrast safeguards */
.st-key-hero_shell,
.st-key-hero_shell * {
    opacity: 1 !important;
}

.st-key-hero_shell button,
.st-key-hero_shell button p,
.st-key-hero_shell button span {
    color: #FFFFFF !important;
}

.st-key-hero_shell .hero-emojis {
    color: #FFFFFF !important;
    filter: saturate(1.15) brightness(1.08);
}


/* =========================================================
   RESPONSIVE / CORPORATE REFINEMENTS
   ========================================================= */

@media (max-width: 900px) {
    .st-key-hero_shell {
        width: 100% !important;
        max-width: none !important;
        margin: 0 0 .85rem 0 !important;
        padding: 1.2rem 1rem 1.25rem 1rem !important;
        border-radius: 0 0 14px 14px !important;
    }

    .st-key-hero_shell [data-testid="stHorizontalBlock"] {
        display: block !important;
    }

    .st-key-hero_shell [data-testid="column"] {
        width: 100% !important;
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }

    .st-key-hero_shell .hero-visual-card {
        display: none !important;
    }

    .st-key-hero_shell .hero-title {
        font-size: 2.45rem !important;
        line-height: 1.03 !important;
    }

    .st-key-hero_shell .hero-subtitle {
        font-size: 1rem !important;
        line-height: 1.4 !important;
    }

    .st-key-hero_shell .hero-copy {
        font-size: .92rem !important;
        line-height: 1.55 !important;
        margin-top: .7rem !important;
    }

    .st-key-hero_shell .hero-kicker {
        font-size: .69rem !important;
        letter-spacing: .10em !important;
    }
}

@media (max-width: 600px) {
    .st-key-hero_shell .hero-title {
        font-size: 2.15rem !important;
    }

    .section-title {
        font-size: 1.28rem !important;
    }

    .st-key-quick_card [data-testid="stHorizontalBlock"],
    .st-key-analyze_card [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }

    .st-key-quick_card [data-testid="column"],
    .st-key-analyze_card [data-testid="column"] {
        width: 100% !important;
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }

    .st-key-quick_card div[data-testid="stButton"] button,
    .st-key-analyze_card div[data-testid="stButton"] button {
        width: 100% !important;
    }

    .aspect-info {
        grid-template-columns: 1fr !important;
        gap: .65rem !important;
    }

    .kpi-card,
    .aspect-card,
    .panel-shell {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }
}

/* Compact landscape-phone layout */
@media (min-width: 601px) and (max-width: 900px) and (orientation: landscape) {
    .st-key-hero_shell {
        padding-top: .95rem !important;
        padding-bottom: 1rem !important;
    }

    .st-key-hero_shell .hero-title {
        font-size: 2.1rem !important;
    }

    .st-key-hero_shell .hero-copy {
        max-width: 94%;
    }
}


/* Mobile navigation access: keep Streamlit's sidebar opener available. */
@media (max-width: 900px) {
    header[data-testid="stHeader"] {
        display: flex !important;
        height: 3rem !important;
        min-height: 3rem !important;
        background: #F7F9FD !important;
        border-bottom: 1px solid #E5EAF2 !important;
    }

    button[data-testid="stSidebarCollapseButton"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# NLTK
# ============================================================

@st.cache_resource
def prepare_nltk_resources():
    for resource in [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger_eng",
    ]:
        nltk.download(resource, quiet=True)

    return True


prepare_nltk_resources()


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.aspect_domain_detector import detect_review_structure
from src.prediction import (
    load_prediction_assets,
    predict_multiple_aspects,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "outputs" / "results"
FIGURE_DIR = BASE_DIR / "outputs" / "figures"


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
        <div class="logo-sub">Sentiment Intelligence</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    nav_items = [
        ("Overview", "Overview"),
        ("Analyze", "Analyze"),
        ("Insights", "Insights"),
        ("Model Lab", "Model Lab"),
        ("Methodology", "Methodology"),
    ]

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

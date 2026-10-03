# AspectIQ: Project Walkthrough

A guide to what the system does, how it works, and where everything lives in the code.

---

## Part 1: What the system is about

### The problem it solves

Normal sentiment analysis gives a whole review one label. Real reviews are mixed:

> "The **screen** is beautiful but the **battery life** is disappointing."

Calling that review "positive" or "negative" misses the point. **Aspect-Based Sentiment Analysis (ABSA)** first finds the things being talked about (the *aspects*), then gives **each one** its own sentiment:

| Aspect | Sentiment |
|---|---|
| screen | Positive |
| battery life | Negative |

**AspectIQ** is a Streamlit web app that does this for **laptop** and **restaurant** reviews.

### The data

The project uses the **SemEval-2014 Task 4** benchmark, a standard academic dataset. It is stored in `data/raw/`:

- `Laptop_Train.xml`, `Laptops_Test.xml`
- `Restaurants_Train.xml`, `Restaurants_Test.xml`

Each sentence is labelled by humans with its aspect terms and a polarity for each: positive, negative, neutral or conflict.

After processing there are **5,841 training examples**, each one a sentence plus one aspect plus its label:

- 3,121 positive
- 1,633 negative
- 1,087 neutral

"Conflict" examples are dropped, so the final model predicts **3 classes**.

### The big picture: two halves

```
┌────────────── PART A: RESEARCH / TRAINING (offline, already done) ───────────────┐
│ XML data → load → clean → GloVe features → train many models → compare → pick best │
│                                                       ↓                           │
│                                    models/best_target_clause_local_3class.joblib  │
└───────────────────────────────────────────────────────────────────────────────────┘
                                                        ↓ (loaded at startup)
┌────────────── PART B: THE APP (app.py) ──────────────────────────────────────────┐
│ user review → split into units → detect domain + aspects → build features per    │
│ aspect → SVM predicts sentiment → cards, highlights, charts, tables              │
└───────────────────────────────────────────────────────────────────────────────────┘
```

- **Part A** is a set of scripts in `src/`. They were run once to produce the results and the saved model.
- **Part B** is `app.py`. It loads the saved model and uses it live.

### How a prediction works (the core idea of the project)

**Step 1: Turn words into numbers (GloVe).**
A model can't read words, so each word is replaced by a **GloVe vector**: 100 numbers learned from Wikipedia. Words with similar meanings get similar numbers ("great" is close to "excellent"). The `gensim` library downloads and loads `glove-wiki-gigaword-100`.

**Step 2: Only look at the part of the sentence that belongs to the aspect.**
This is the project's key idea. If you average the whole sentence, "beautiful" and "disappointing" blend together, and both aspects get the same mixed signal. This is called **cross-aspect sentiment leakage**.

To prevent it, the sentence is cut at **contrast words** (*but, however, although, though, yet, whereas, while…*):

```
the screen is beautiful | but | the battery life is disappointing
└──── clause for "screen" ┘     └──── clause for "battery life" ────┘
```

It is also cut at commas, semicolons and colons; see *Improvements added* below.

**Step 3: Build a 200-number "fingerprint" for each aspect.**

- **100 numbers:** the average GloVe vector of the aspect's **target clause**.
- **100 numbers:** a **local-context** average of the 5 words either side of the aspect (still inside the clause). Closer words are weighted more: weight = 1 / (1 + distance).

Together that is a **200-dimensional feature**. Negation words (*not, never, no*) are deliberately kept, and "isn't" is expanded to "is not", so negation is not lost.

**Step 4: Classify with an SVM.**
The fingerprint goes into a **Support Vector Machine** with an RBF kernel.

- It first passes through a `StandardScaler`.
- `class_weight="balanced"` stops the model from simply favouring "positive", the majority class.

The output is positive, negative or neutral.

### How good is it?

The model was chosen using **grouped 5-fold cross-validation on the training data only**. "Grouped" means sentences from the same review never appear in both the training and checking folds, which avoids an unfair advantage.

| Representation | Dims | Macro F1 | Accuracy | ROC-AUC |
|---|---|---|---|---|
| **Target Clause + Local (chosen)** | 200 | **0.631** | 0.679 | 0.817 |
| Target Clause + Aspect | 200 | 0.627 | 0.675 | 0.814 |
| Clause + Aspect + Local | 300 | 0.626 | 0.673 | 0.817 |
| Full sentence + Aspect + Local (older approach) | 300 | 0.624 | 0.671 | 0.816 |
| Local only | 100 | 0.579 | 0.620 | 0.776 |

An earlier 300d model scored **70.3% accuracy / 0.627 Macro F1** on the official SemEval test set. It is kept as a historical benchmark.

Points to say honestly:

- **Macro F1 is the headline metric**, not accuracy, because the classes are unbalanced.
- **Neutral is the hardest class.** This is typical for this dataset.
- The **confidence %** in the app is **not** a true probability. It is a softmax of the SVM's decision scores; the code marks this with `scores_are_calibrated: False`.
- Because GloVe averages word meanings, some words can mislead it. For example, "cold" in *"the pizza was cold"* is not always read as negative.
- Transformer models (e.g. BERT) score higher on this benchmark but are much heavier. This is a good **future work** point.

---

## Part 2: The codebase

### Folder map

```
ABSA/
├── app.py                 ← THE APP (the only one that matters)
├── app1.py … app16.py     ← old versions, kept as history
├── app_auth.py            ← same app + Google/Microsoft login
├── requirements.txt
├── assets/absabot.jpg     ← picture for the chatbot (bottom-right)
├── styles/main.css        ← the app's look
├── .streamlit/config.toml ← Streamlit theme settings
├── data/raw/              ← original SemEval XML files
├── data/processed/        ← cleaned CSVs produced by the pipeline
├── models/                ← saved trained models (.joblib) + metadata (.json)
├── outputs/               ← experiment results, charts, confusion matrices
├── notebooks/
└── src/                   ← all the logic
```

### `src/`, in the order the pipeline runs

**1. Getting the data ready**

| File | What it does |
|---|---|
| `data_loader.py` | Reads the 4 XML files and pulls out sentences, aspect terms and categories. Validates them, then writes `data/processed/*.csv` (e.g. `absa_model_dataset.csv`, one row per sentence and aspect). |
| `preprocessing.py` | Text cleaning: lowercase, expand "n't" to "not", strip URLs/digits/punctuation, tokenise, remove stopwords (but **keep** *not, but, very…*), and lemmatise ("running" → "run"). Produces `absa_train_preprocessed.csv` / `absa_test_preprocessed.csv`. |
| `exploratory_analysis.py` | Exploratory data analysis: sentiment distributions, top aspects, word clouds, sentence lengths. Charts are saved to `outputs/figures/`. |

**2. Features (turning text into numbers)**

| File | What it does |
|---|---|
| `glove_embeddings.py` | The first baseline: one average GloVe vector over the whole sentence. |
| `local_context_embeddings.py` | Adds a window of words around the aspect. Defines `GLOVE_MODEL_NAME`. |
| **`contrast_aware_features.py`** | ⭐ **The core.** `contrast_clause_bounds()` cuts the sentence at contrast words. `weighted_glove_average()` does the distance weighting. **`build_target_clause_local_feature()`** builds the production 200d feature. |

**3. Experiments (how the best model was chosen)**

| File | What it does |
|---|---|
| `model_training.py` | First experiments: SVM vs Logistic Regression, cross-validation, confusion matrices, ROC curves. |
| `controlled_experiments.py`, `improved_glove_controlled_experiments.py`, `contrast_aware_glove_controlled_experiments.py` | Fair, controlled comparisons: 3-class vs 4-class tasks, 200d vs 300d features, and different feature recipes. |
| `target_context_representation_experiment.py` | The final representation comparison (the table above). |
| `tune_target_context_svm.py` | Tunes the SVM settings (C, gamma, class weights) with nested grouped cross-validation. |
| `final_champion_evaluation.py`, `final_model_comparison.py` | Final scoring on the test set and comparison charts. |
| `error_analysis.py`, `battery_feature_diagnostic.py`, `contrast_context_diagnostic.py` | Check *why* the model gets certain examples wrong. |

**4. The production model**

| File | What it does |
|---|---|
| `train_production_target_clause_local.py` | Trains the final SVM on all the training data and saves `models/best_target_clause_local_3class.joblib` + `.json`. |
| **`prediction.py`** | ⭐ What the app calls. `load_prediction_assets()` loads the model and GloVe. `find_aspect_piece()` cuts the review at commas. `predict_aspect()` builds the 200d feature, runs the SVM, and returns the sentiment, confidence and the words used. `predict_multiple_aspects()` loops over aspects. |
| `smoke_test_production_model.py` | A quick check that the model still gives the expected answers for 4 examples. |
| **`aspect_domain_detector.py`** | ⭐ Finds aspects and the domain in **new** text. 1) It looks for aspect terms seen in training (a *lexicon*). 2) If none are found, it falls back to NLTK noun phrases. 3) It scores Laptop vs Restaurant from the aspects plus word frequencies. 4) It drops aspects that don't fit that domain. |

Files ending `_v1`, `_v2`, `_v3` are older versions, like `app1–16.py`.

### `app.py`: how the running app is built

Streamlit reruns the file from top to bottom on every click.

1. **Setup:** page config, load `styles/main.css`, download NLTK data.
2. **Load the model and GloVe once.** `@st.cache_resource` means they are not reloaded on every click.
3. **Input handling:**
   - `segment_input_text()` splits long or pasted text into units, so a document that mixes laptop and restaurant reviews is analysed per section.
   - `read_uploaded_file()` handles file uploads.
4. **The analysis engine:**
   ```
   analyse_review(text)
     → detect_review_structure(text)      # domain + aspects    (aspect_domain_detector.py)
     → predict_multiple_aspects(...)      # sentiment per aspect (prediction.py)
     → rows for the results table
   ```
5. **UI components:** hero banner, KPI tiles, highlighted review, per-aspect cards, Plotly charts.
6. **The sidebar and the 7 pages:**

| Page | What it shows |
|---|---|
| **Overview** | Landing page / introduction |
| **Analyze** | Type a review or upload a file → aspects, sentiments, charts |
| **Insights** | Summary of the current analysis |
| **Data Explorer** | EDA of the SemEval dataset and of your submitted reviews |
| **Preprocessing** | Step-by-step demo of the text cleaning |
| **Model Lab** | Model metrics, representation comparison, confusion matrices |
| **Methodology** | Written explanation of the approach |

7. **The floating chatbot** (bottom-right, on every page). See below.

### One-sentence summary

> AspectIQ detects the aspects mentioned in a laptop or restaurant review. It builds a 200-dimensional GloVe representation of each aspect's own clause and nearby words, so sentiment from other aspects doesn't leak in. A class-balanced RBF SVM, trained on SemEval-2014 and selected by grouped cross-validation (Macro F1 ≈ 0.63), then classifies each aspect as positive, negative or neutral.

---

## Part 3: Improvements added

### 1. Fix: commas now separate clauses (`src/prediction.py` → `find_aspect_piece()`)

**The problem.** In *"The laptop has an excellent screen, but the battery life is disappointing, and it comes with 8GB of RAM"* every aspect came out **positive**. The sentence was only cut at contrast words like *but*. So "battery life" was judged on *"…is disappointing **and it comes with 8 gb of ram**"*, and the extra words drowned out "disappointing".

**The fix.** Before building the features, `find_aspect_piece()` cuts the review at commas, semicolons and colons, and keeps only the piece that mentions the aspect. If that piece is shorter than 4 words (e.g. *"The food,"*), it is too short to hold an opinion, so the full sentence is used instead.

**Result.**

| Aspect | Before | After |
|---|---|---|
| screen | Positive | Positive |
| battery life | Positive ❌ | **Negative** ✅ |
| 8GB of RAM | Positive ❌ | **Neutral** ✅ |

On the official SemEval test set the overall scores are essentially unchanged:

- Macro F1: 0.622 → 0.623
- Accuracy: 70.1% → 69.9%

Cutting at *every* comma without the 4-word rule made scores worse (Macro F1 0.605), which is why the rule exists. The model was **not retrained**; only how the input text is prepared changed.

### 2. Explainability: highlighted review (`app.py` → `highlight_review()`)

Above the result cards, the review is shown with each aspect **coloured by its sentiment**: green = positive, red = negative, amber = neutral. The function walks through the review from left to right and wraps each aspect in a coloured `<mark>` tag. The text is escaped with `html.escape`, so a review can't inject HTML into the page.

On each card, the **"Words the model used"** line shows exactly which part of the sentence the model looked at. Together, these show *why* the model made each decision.

### 3. Confetti and balloons (`app.py` → `render_aspect_card()`, `all_positive()`)

- **Positive cards** get a small burst of 🎉✨🎊 that falls once inside the card. Negative and neutral cards stay plain. The animation is pure CSS (`@keyframes confetti-fall` in `styles/main.css`).
- **If every aspect is positive**, `all_positive()` returns True and Streamlit's `st.balloons()` flies balloons over the whole page.

### 4. Chatbot (`app.py` → `render_chatbot()`, `chatbot_reply()`)

- A chatbot button sits in the **bottom-right corner of every page**. It uses the picture in `assets/absabot.jpg`, or a 🤖 emoji if the picture is missing.
- Clicking it opens a small chat window. You type a review, and the bot replies with each aspect, its sentiment and confidence, plus the highlighted review.
- It uses the **same model and pipeline** as the Analyze page (`analyse_review()`); it is a friendlier way to enter text. The original text area still works as before.
- It is built only from Streamlit's own chat components (`st.popover`, `st.chat_input`, `st.chat_message`). There is no external AI service and no API key.
- CSS pins it to the corner: `.st-key-chatbot_widget { position: fixed; bottom: …; right: …; }`.

---

## Part 4: How the app works, page by page

### 4.1 The "template": Streamlit, the theme and the CSS

There is no HTML template. The whole app is **one Python file**, `app.py`, built with **Streamlit**, a library that turns a Python script into a web page:

```python
st.title("Hello")
name = st.text_input("Your name")
if st.button("Say hi"):
    st.write("Hi", name)
```

**The most important idea:** every time the user clicks a button or types something, **Streamlit runs `app.py` again from top to bottom.** Because of that, ordinary variables are forgotten after each click. Anything the app must remember is stored in **`st.session_state`**, a dictionary that survives reruns.

The look comes from three places:

| Where | What it controls |
|---|---|
| `.streamlit/config.toml` | Base theme: light mode, blue primary colour `#1769FF`, background `#F7F9FD`, font. `runOnSave = true` reloads the app when a file is saved. |
| `styles/main.css` | All the custom design: dark sidebar, banner, cards, KPI tiles, confetti, chatbot position. |
| HTML snippets in `app.py` | Pieces like `<div class="aspect-card">` that use the CSS classes. |

How the CSS gets in (`load_css()` near the top of `app.py`):

```python
def load_css(css_path):
    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
```

The function reads the CSS file and puts it into the page inside a `<style>` tag. `unsafe_allow_html=True` means "this is real HTML, don't show it as text". The app uses the same trick everywhere to draw custom boxes:

```python
st.markdown('<div class="section-title">Text or review file</div>', unsafe_allow_html=True)
```

**Containers with a key:** `st.container(key="hero_shell")` gives that box the CSS class `st-key-hero_shell`, which the CSS can then style. That's how:

- the banner stays stuck at the top (`position: sticky`)
- the chatbot floats bottom-right (`position: fixed`)

### 4.2 What happens when `app.py` runs, top to bottom

```
1. Imports                         (streamlit, pandas, plotly, nltk…)
2. st.set_page_config(...)         browser tab title "AspectIQ", wide layout
3. load_css(...)                   apply styles/main.css
4. prepare_nltk_resources()        download NLTK word lists (once)
5. import from src/                detector, prediction, preprocessing
6. load_assets()                   load the SVM model + GloVe (once)
7. session_state defaults          page="Overview", analyses=[], results_df=empty
8. Helper functions                (defined only, not run yet)
9. Sidebar / navbar                draw the menu buttons
10. Balloons check                 celebrate if everything was positive
11. Page router (if / elif)        draw ONLY the selected page
12. render_chatbot()               floating bot on every page
```

**Caching (steps 4 and 6).** Loading GloVe takes several seconds. Without caching it would happen on every click.

```python
@st.cache_resource
def load_assets():
    return load_prediction_assets()
```

`@st.cache_resource` means: run once, keep the result in memory, and reuse it.

- `@st.cache_resource` is for big objects such as models.
- `@st.cache_data` is for data tables, e.g. `load_training_corpus()`.

**The app's memory (step 7).**

```python
if "page" not in st.session_state:
    st.session_state.page = "Overview"      # which page is open
if "analyses" not in st.session_state:
    st.session_state.analyses = []          # results of the last analysis
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()   # same results as a table
```

The `if … not in` check sets these values only the first time. Every page reads from them, which is why the Insights page can show what you analysed on the Analyze page.

### 4.3 The navbar (sidebar)

```python
with st.sidebar:
    navigation_groups = [
        ("OVERVIEW",         [("Overview", "Overview")]),
        ("ANALYSIS",         [("Analyze", "Analyze"), ("Insights", "Insights")]),
        ("DATA & METHODS",   [("Data Explorer", "Data Explorer"), ("Preprocessing", "Preprocessing")]),
        ("MODEL & RESEARCH", [("Model Lab", "Model Lab"), ("Methodology", "Methodology")]),
    ]

    for group_name, nav_items in navigation_groups:
        st.markdown(f'<div class="nav-section-label">{group_name}</div>', unsafe_allow_html=True)

        for label, destination in nav_items:
            is_active = st.session_state.page == destination
            if st.button(label, key=f"nav_{destination}",
                         type="primary" if is_active else "secondary"):
                st.session_state.page = destination
                st.rerun()
```

1. `with st.sidebar:` puts everything inside it into the dark left panel.
2. `navigation_groups` is a **list of menu sections**. Each section has a heading and its buttons, written as `(text shown, page name)`.
3. The outer loop writes each heading. The inner loop makes one button per page.
4. The current page's button gets `type="primary"`, which the CSS draws as the highlighted button.
5. A click saves the page name in `st.session_state.page`, and `st.rerun()` restarts the script.

**The page router** comes next:

```python
if st.session_state.page == "Overview":
    ...
elif st.session_state.page == "Analyze":
    ...
elif st.session_state.page == "Insights":
    ...
# Data Explorer, Preprocessing, Model Lab, Methodology
```

Only one branch runs on each rerun. The whole "multi-page" system is **a variable plus an if/elif chain.** To add a page, add one tuple to `navigation_groups` and one `elif` block.

What one click looks like:

```
You click "Insights"
  → st.session_state.page = "Insights"
  → st.rerun()
  → app.py runs again (model/GloVe come from the cache, so it's instant)
  → sidebar redrawn, with "Insights" highlighted
  → router reaches  elif page == "Insights":  and draws that page
  → chatbot drawn at the end
```

### 4.4 Shared building blocks

| Function | What it does |
|---|---|
| `render_hero()` | The blue **AspectIQ banner**: title, subtitle and the "Sentiment Signal" picture. It stays stuck at the top while you scroll. |
| `render_review_input(key_prefix, ...)` | The input box, with two tabs: **Analyze Text** and **Upload Review File** (TXT/CSV/XML). Overview and Analyze both use it. `key_prefix` gives each widget a unique name. |
| `segment_input_text(text)` | Splits long pasted text into chunks of about 700 characters or fewer, so mixed laptop/restaurant documents are analysed piece by piece. |
| `read_uploaded_file(file)` | Reads TXT, CSV or XML and returns a list of reviews. It finds the review column in a CSV automatically. |
| `analyse_reviews(reviews)` | Loops over the reviews with a progress bar and calls `analyse_review()` on each. It **saves the results in `session_state`** and sets the balloons flag. |
| `analyse_review(id, text)` | **The engine.** `detect_review_structure()` finds the domain and aspects, then `predict_multiple_aspects()` gives each aspect a sentiment. |
| `render_results(...)` | KPI tiles, the highlighted review, a card per aspect grouped by domain, then charts. |
| `render_kpis()` / `kpi_html()` | The four tiles: Text Units Analyzed, Aspects Detected, Detected Domain, Overall Sentiment. |
| `overall_mood(df)` | The most common sentiment, or **"Mixed"** if it covers less than 65% of the results. |
| `render_aspect_card(result)` | One card: emoji, aspect, sentiment pill, confidence bar, "Words the model used", and confetti if positive. |
| `render_charts(df)` | Plotly charts: **Sentiment Distribution** and **Top Detected Aspects**. |

What happens when you press "Analyze":

```
Type a review → click "Analyze Text"
  render_review_input: empty box → warning
                       else → analyse_reviews(segment_input_text(text))
     analyse_review: detect domain + aspects → predict sentiment per aspect
     results saved in st.session_state.analyses / .results_df
  st.rerun() → the page redraws and shows render_results(...)
```

### 4.5 Each page in the navbar

**🏠 Overview, the landing page.** This is the first page you see, because `page` starts as `"Overview"`.

- It shows the banner, plus a **Quick Analysis** card with `render_review_input("overview", ...)`.
- If results exist, it shows them with `render_results`.
- Before any analysis, it shows four **placeholder KPI tiles** with "—". `st.columns(4)` splits the row into four columns, and a `for … zip` loop puts one tile in each.

**🔍 Analyze, the main working page.** It has two states:

- **No results yet:** an "Analyze — Text or review file" card with the input box.
- **Results exist:** shown at the top, followed by:
  - **New Analysis**, which empties `analyses` and `results_df` and reruns.
  - **Download Results**, which uses `st.download_button` to save the results table as `aspectiq_analysis_results.csv`.
  - An expander, **Run another analysis**, holding the input box again.

**📊 Insights, the summary.** If nothing has been analysed it shows an info message. Otherwise it shows KPI tiles, charts and the full results table (`st.dataframe`). It doesn't analyse anything itself; it reads what Analyze saved.

**🗂️ Data Explorer, exploring the data.** It has two tabs, made with `st.tabs([...])`.

- **SemEval Dataset EDA.** `load_training_corpus()` (`@st.cache_data`) reads `data/processed/absa_train_preprocessed.csv`, and `prepare_training_eda_data()` prepares the tables. Five sections follow:
  1. **Corpus Overview:** `st.metric` tiles for sentences, aspect records, unique aspects and domains.
  2. **Sentiment and Domain Distributions:** bar charts. The unequal class counts are why Macro F1 is reported.
  3. **Frequent Aspects:** the top 15 aspect terms.
  4. **Review Length Analysis:** mean, median, shortest and longest token counts, plus a histogram.
  5. **Word Frequency Analysis:** the top 20 words (`training_token_frequencies`) and an expander with the full table.
- **Current Analysis.** The same kind of analysis for your own submitted reviews: tiles, the submitted-reviews table, an aspect table, and domain, sentiment and aspect charts. If nothing has been analysed yet, it shows a **Go to Analyze** button.

`px.bar(...)` builds a Plotly chart, and `st.plotly_chart(fig)` shows it.

**🧹 Preprocessing, how text is cleaned.** It needs an analysis first. `st.selectbox` picks a review, and `instructional_preprocessing_steps(text)` shows each stage:

```
Original:    "The battery isn't great, but the screen is AMAZING!!!"
Lowercase:   "the battery isn't great, but the screen is amazing!!!"
Expand neg.: "the battery is not great, but the screen is amazing!!!"
Clean:       "the battery is not great but the screen is amazing"
Tokens:      [the, battery, is, not, great, but, the, screen, is, amazing]
Stopwords:   [battery, not, great, but, screen, amazing]   ← "not" and "but" kept on purpose
Lemmatise:   [battery, not, great, but, screen, amazing]
```

The sections are: Transformation stages, Before and after, tokens versus filtered tokens, and All submitted reviews after preprocessing. Negation, contrast and intensifier words are kept because removing "not" would flip the meaning.

**🧪 Model Lab, the evidence.** It reads result files from `outputs/` and `models/`. If a file is missing, it uses built-in fallback values.

1. **Detailed Class-Level Evaluation:** precision, recall and F1 per class. There are tabs for **SVM — Best Model** and **Logistic Regression**.
2. **Logistic Regression vs SVM:** a comparison table and chart, "SVM Δ" metric tiles, and a green box naming SVM the best model.
3. **Confusion Matrices:** images from `outputs/figures/` (`st.image`).
4. **Selected Production Model:** details from `models/best_target_clause_local_3class.json`.
5. **GloVe Representation Experiment:** the Macro F1 chart per feature recipe, showing why Target Clause + Local 200d won, plus a **Development Conclusion** panel.

An expander at the end holds the earlier 300d model's official test scores, kept for transparency.

**📘 Methodology, the written explanation.** Static text made of `st.markdown` panels with no calculations:

- Automatic ABSA Pipeline
- Final Sentiment Model
- Why Target-Context Representation?
- Exploratory Text Analytics
- Classifier Evaluation and Comparison
- Model Selection
- Evaluation Interpretation
- Automatic Front-End Intelligence

To change it, edit the text inside the `"""…"""` strings.

### 4.6 The end of the file: the chatbot

`render_chatbot()` runs after the router, outside the if/elif, so it appears on **every page**.

- `st.container(key="chatbot_widget")` is pinned bottom-right by the CSS.
- The container holds the bot picture and an `st.popover("Chat with AspectIQ")`.
- Inside the popover, `st.chat_input` takes a review, `analyse_review()` analyses it, and `st.chat_message` shows the conversation.
- Messages are kept in `st.session_state.chat_messages`.

### 4.7 The whole flow in one picture

```
Browser opens app
   │
   ▼
app.py runs top → bottom
   ├─ theme (config.toml) + CSS (main.css)
   ├─ model + GloVe loaded once (cached)
   ├─ session_state: page, analyses, results_df
   ├─ SIDEBAR: buttons ─ click ─► page = "X" ─► rerun ──┐
   ├─ ROUTER: if/elif draws page "X"   ◄────────────────┘
   │     Overview / Analyze ─► render_review_input ─► analyse_reviews
   │                              ─► detect aspects + domain (src/aspect_domain_detector.py)
   │                              ─► predict sentiment     (src/prediction.py)
   │                              ─► save in session_state ─► rerun ─► render_results
   │     Insights / Data Explorer / Preprocessing ─► read session_state
   │     Model Lab ─► read saved results in outputs/ and models/
   │     Methodology ─► static text
   └─ CHATBOT (every page) ─► analyse_review ─► reply in chat
```

### Five Streamlit ideas to remember

1. **The script reruns on every click.**
2. **`st.session_state` is the memory** shared between reruns and pages.
3. **`@st.cache_resource` / `@st.cache_data`** do slow work only once.
4. **Navigation is a variable plus if/elif**, and the sidebar buttons change that variable.
5. **`st.markdown(..., unsafe_allow_html=True)` plus `main.css`** produce the custom design.

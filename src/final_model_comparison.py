"""Assignment-ready Logistic Regression vs SVM comparison for AspectIQ.

Both classifiers use the same 3-class data, the same final Target Clause +
distance-weighted Local Context GloVe 200d representation, and identical
grouped 5-fold CV splits. Reported metrics are out-of-fold training metrics,
not a new independent official-test result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.contrast_aware_features import build_target_clause_local_matrix
from src.contrast_aware_glove_controlled_experiments import (
    ASPECT_COLUMN_CANDIDATES,
    ID_COLUMN_CANDIDATES,
    LABEL_COLUMN_CANDIDATES,
    TEXT_COLUMN_CANDIDATES,
    build_task_dataset,
    detect_fourth_label,
    load_glove_model,
    minimum_class_group_count,
    prepare_dataset,
    read_dataset,
    resolve_column,
)

RANDOM_STATE = 42
PREFERRED_LABEL_ORDER = ["negative", "neutral", "positive"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/absa_train_preprocessed.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/final_model_comparison"),
    )
    parser.add_argument("--glove-model", default="glove-wiki-gigaword-100")
    parser.add_argument("--glove-cache-dir", type=Path, default=None)
    parser.add_argument("--context-window", type=int, default=5)
    parser.add_argument("--cv-folds", type=int, default=5)
    return parser.parse_args()


def resolve_columns(df):
    return (
        resolve_column(df, None, TEXT_COLUMN_CANDIDATES, "text"),
        resolve_column(df, None, ASPECT_COLUMN_CANDIDATES, "aspect"),
        resolve_column(df, None, LABEL_COLUMN_CANDIDATES, "label"),
        resolve_column(df, None, ID_COLUMN_CANDIDATES, "id", required=False),
    )


def softmax_rows(values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = np.column_stack([-values, values])
    values = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(values)
    totals = exp_values.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return exp_values / totals


def ordered_labels(labels):
    unique = list(pd.unique(pd.Series(labels)))
    preferred = [x for x in PREFERRED_LABEL_ORDER if x in unique]
    remaining = sorted(x for x in unique if x not in preferred)
    return preferred + remaining


def build_models():
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=5000,
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                        decision_function_shape="ovr",
                    ),
                ),
            ]
        ),
    }


def get_probabilities(model, X, global_classes):
    clf = model.named_steps["classifier"]
    model_classes = list(clf.classes_)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
    else:
        probs = softmax_rows(model.decision_function(X))

    aligned = np.zeros((len(X), len(global_classes)), dtype=float)
    class_index = {label: i for i, label in enumerate(model_classes)}
    for j, label in enumerate(global_classes):
        if label in class_index:
            aligned[:, j] = probs[:, class_index[label]]
    totals = aligned.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return aligned / totals


def calculate_metrics(y_true, y_pred, probs, classes):
    try:
        auc = roc_auc_score(
            y_true,
            probs,
            labels=classes,
            average="macro",
            multi_class="ovr",
        )
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "roc_auc_macro_ovr": float(auc),
    }


def slug(name):
    return name.casefold().replace(" ", "_").replace("-", "_")


def save_confusion(y_true, y_pred, labels, model_name, output_dir):
    file_slug = slug(model_name)

    for normalized, suffix, value_format in [
        (None, "confusion_matrix", "d"),
        ("true", "confusion_matrix_normalized", ".2f"),
    ]:
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalized)
        pd.DataFrame(
            cm,
            index=[f"actual_{x}" for x in labels],
            columns=[f"predicted_{x}" for x in labels],
        ).to_csv(output_dir / f"{file_slug}_{suffix}.csv")

        fig, ax = plt.subplots(figsize=(7, 5.6))
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(
            ax=ax,
            values_format=value_format,
        )
        title = "Normalized " if normalized else ""
        ax.set_title(f"{model_name} — {title}Grouped-CV Confusion Matrix")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{file_slug}_{suffix}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)


def evaluate_model(name, estimator, X, data, groups, splits, classes, output_dir):
    y = data["label"].to_numpy()
    predictions = np.empty(len(data), dtype=object)
    probabilities = np.zeros((len(data), len(classes)), dtype=float)
    fold_rows = []

    print(f"\nEvaluating {name} across {len(splits)} grouped folds...")

    for fold_no, (train_idx, val_idx) in enumerate(splits, start=1):
        estimator.fit(X[train_idx], y[train_idx])
        fold_pred = estimator.predict(X[val_idx])
        fold_prob = get_probabilities(estimator, X[val_idx], classes)

        predictions[val_idx] = fold_pred
        probabilities[val_idx] = fold_prob

        fold_metrics = calculate_metrics(y[val_idx], fold_pred, fold_prob, classes)
        fold_rows.append(
            {
                "model": name,
                "fold": fold_no,
                "train_records": len(train_idx),
                "validation_records": len(val_idx),
                **fold_metrics,
            }
        )
        print(
            f"  Fold {fold_no}: Accuracy={fold_metrics['accuracy']:.4f} | "
            f"Macro F1={fold_metrics['f1_macro']:.4f}"
        )

    metrics = calculate_metrics(y, predictions, probabilities, classes)
    file_slug = slug(name)

    pd.DataFrame(
        classification_report(
            y,
            predictions,
            labels=classes,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        )
    ).transpose().to_csv(
        output_dir / f"{file_slug}_classification_report.csv",
        index_label="class",
    )

    oof = data[["group_id", "text", "aspect", "label"]].copy()
    oof["predicted_label"] = predictions
    oof["correct"] = oof["label"] == oof["predicted_label"]
    for j, label in enumerate(classes):
        oof[f"score_{label}"] = probabilities[:, j]
    oof.to_csv(
        output_dir / f"{file_slug}_grouped_cv_predictions.csv",
        index=False,
    )

    save_confusion(y, predictions, classes, name, output_dir)

    return (
        {
            "model": name,
            "representation": "Target Clause + Local Context GloVe",
            "dimensions": int(X.shape[1]),
            "cv_scheme": f"{len(splits)}-fold StratifiedGroupKFold",
            **metrics,
        },
        pd.DataFrame(fold_rows),
    )


def save_comparison_chart(results, output_dir):
    metric_cols = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc_macro_ovr",
    ]
    metric_labels = [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Macro ROC-AUC",
    ]

    x = np.arange(len(metric_cols))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.8))

    names = results["model"].tolist()
    for i, name in enumerate(names):
        values = results.loc[results["model"] == name, metric_cols].iloc[0].to_numpy(float)
        offset = (i - (len(names) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=name)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_title("Logistic Regression vs SVM — Same 200d GloVe Representation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "model_comparison_metrics.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\nFINAL ASSIGNMENT MODEL COMPARISON")
    print("Evaluation: training-only grouped cross-validation")
    print("Representation: Target Clause + Local Context GloVe 200d")

    raw = read_dataset(args.input)
    text_col, aspect_col, label_col, id_col = resolve_columns(raw)

    prepared = prepare_dataset(
        raw,
        text_column=text_col,
        aspect_column=aspect_col,
        label_column=label_col,
        id_column=id_col,
        keep_duplicates=False,
    )

    fourth_label = detect_fourth_label(prepared["label"].tolist(), None)
    data = build_task_dataset(
        prepared,
        task="3-class",
        fourth_label=fourth_label,
        three_class_policy="drop",
    )

    print(f"\nUsable 3-class records: {len(data):,}")
    print(data["label"].value_counts().to_string())

    groups = data["group_id"].to_numpy()
    folds = min(
        args.cv_folds,
        minimum_class_group_count(data["label"].to_numpy(), groups),
    )
    if folds < 2:
        raise ValueError("Not enough independent groups for grouped cross-validation.")

    print(f"\nGrouped CV folds: {folds}")

    glove = load_glove_model(args.glove_model, args.glove_cache_dir)
    X, metadata_rows, oov = build_target_clause_local_matrix(
        data[["text", "aspect"]],
        glove,
        context_window=args.context_window,
    )

    if X.shape[1] != 200:
        raise ValueError(f"Expected 200 feature dimensions; found {X.shape[1]}.")

    metadata = pd.DataFrame(metadata_rows)
    metadata.insert(0, "group_id", data["group_id"].to_numpy())
    metadata.insert(1, "text", data["text"].to_numpy())
    metadata.insert(2, "aspect", data["aspect"].to_numpy())
    metadata.insert(3, "label", data["label"].to_numpy())
    metadata.to_csv(args.output_dir / "target_clause_local_metadata.csv", index=False)

    if oov:
        pd.DataFrame(oov.most_common(), columns=["token", "count"]).to_csv(
            args.output_dir / "target_clause_local_oov.csv",
            index=False,
        )

    classes = ordered_labels(data["label"].tolist())
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    splits = list(splitter.split(X, data["label"].to_numpy(), groups))

    results = []
    fold_frames = []
    for name, estimator in build_models().items():
        result, fold_df = evaluate_model(
            name,
            estimator,
            X,
            data,
            groups,
            splits,
            classes,
            args.output_dir,
        )
        results.append(result)
        fold_frames.append(fold_df)

    results_df = (
        pd.DataFrame(results)
        .sort_values(["f1_macro", "accuracy"], ascending=False)
        .reset_index(drop=True)
    )
    results_df["rank_by_macro_f1"] = np.arange(1, len(results_df) + 1)
    results_df.to_csv(args.output_dir / "model_comparison.csv", index=False)

    pd.concat(fold_frames, ignore_index=True).to_csv(
        args.output_dir / "fold_metrics.csv",
        index=False,
    )

    save_comparison_chart(results_df, args.output_dir)

    champion = results_df.iloc[0].to_dict()
    summary = {
        "evaluation_scope": "training-only grouped cross-validation",
        "records": int(len(data)),
        "classes": classes,
        "cv_folds": int(folds),
        "representation": (
            "Target Clause 100d mean GloVe + distance-weighted Local Context 100d GloVe"
        ),
        "dimensions": int(X.shape[1]),
        "context_window": int(args.context_window),
        "champion_by_macro_f1": champion,
        "methodological_note": (
            "Both classifiers use identical features and identical grouped CV folds. "
            "No official test data is used in this final comparison."
        ),
    }
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL MODEL COMPARISON")
    print(
        results_df[
            [
                "model",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "roc_auc_macro_ovr",
                "rank_by_macro_f1",
            ]
        ].to_string(index=False)
    )
    print(f"\nBest model by Macro F1: {champion['model']}")
    print(f"\nOutputs written to: {args.output_dir}")
    print(
        "\nThese are grouped out-of-fold training comparison metrics, "
        "not a new independent official-test evaluation."
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
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
    prepare_dataset,
    read_dataset,
    resolve_column,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the selected AspectIQ production model: "
            "SVM + Target Clause + Local Context 200d."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/processed/absa_train_preprocessed.csv"
        ),
    )

    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path(
            "models/best_target_clause_local_3class.joblib"
        ),
    )

    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path(
            "models/best_target_clause_local_3class.json"
        ),
    )

    parser.add_argument(
        "--glove-model",
        default="glove-wiki-gigaword-100",
    )

    parser.add_argument(
        "--glove-cache-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--context-window",
        type=int,
        default=5,
    )

    return parser.parse_args()


def resolve_columns(dataframe):
    text_column = resolve_column(
        dataframe,
        None,
        TEXT_COLUMN_CANDIDATES,
        "text",
    )

    aspect_column = resolve_column(
        dataframe,
        None,
        ASPECT_COLUMN_CANDIDATES,
        "aspect",
    )

    label_column = resolve_column(
        dataframe,
        None,
        LABEL_COLUMN_CANDIDATES,
        "label",
    )

    id_column = resolve_column(
        dataframe,
        None,
        ID_COLUMN_CANDIDATES,
        "id",
        required=False,
    )

    return (
        text_column,
        aspect_column,
        label_column,
        id_column,
    )


def build_production_pipeline():
    return Pipeline(
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
    )


def main():
    args = parse_args()

    if args.context_window < 1:
        raise ValueError(
            "--context-window must be at least 1."
        )

    raw = read_dataset(args.input)

    (
        text_column,
        aspect_column,
        label_column,
        id_column,
    ) = resolve_columns(raw)

    prepared = prepare_dataset(
        raw,
        text_column=text_column,
        aspect_column=aspect_column,
        label_column=label_column,
        id_column=id_column,
        keep_duplicates=False,
    )

    fourth_label = detect_fourth_label(
        prepared["label"].tolist(),
        None,
    )

    train_data = build_task_dataset(
        prepared,
        task="3-class",
        fourth_label=fourth_label,
        three_class_policy="drop",
    )

    print("\nPRODUCTION TRAINING")
    print(
        "Model: SVM RBF | C=1.0 | gamma=scale | "
        "class_weight=balanced"
    )
    print(
        "Representation: Target Clause + Local Context "
        "GloVe 200d"
    )
    print(f"Training records: {len(train_data):,}")
    print(train_data["label"].value_counts().to_string())

    glove_model = load_glove_model(
        args.glove_model,
        args.glove_cache_dir,
    )

    features, _, oov = build_target_clause_local_matrix(
        train_data,
        glove_model,
        context_window=args.context_window,
    )

    if features.shape[1] != 200:
        raise ValueError(
            "Expected 200-dimensional production features, "
            f"got {features.shape[1]}."
        )

    model = build_production_pipeline()

    print(
        "\nFitting production model on all "
        "3-class training data..."
    )

    model.fit(
        features,
        train_data["label"].to_numpy(),
    )

    args.model_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        args.model_output,
    )

    metadata = {
        "task": "3-class",
        "model": "SVM",
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "class_weight": "balanced",
        "representation": (
            "Target Clause + Local Context GloVe"
        ),
        "dimensions": 200,
        "context_window": int(args.context_window),
        "contrast_aware": True,
        "distance_weighted_local_context": True,
        "glove_model": args.glove_model,
        "training_samples": int(len(train_data)),
        "training_label_counts": {
            str(label): int(count)
            for label, count
            in train_data["label"].value_counts().items()
        },
        "selection_evidence": {
            "grouped_cv_accuracy": 0.679165,
            "grouped_cv_macro_f1": 0.630693,
            "grouped_cv_macro_roc_auc_ovr": 0.816772,
            "selection_basis": (
                "Highest grouped 5-fold training-only "
                "Macro F1 among tested target-context "
                "representations."
            ),
        },
        "official_test_note": (
            "The official test set was previously used for "
            "the earlier 300d champion. This 200d production "
            "representation was selected using training-only "
            "grouped cross-validation."
        ),
        "model_file": str(args.model_output),
    }

    args.metadata_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.metadata_output.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    oov_path = (
        args.metadata_output.parent
        / "best_target_clause_local_3class_oov.csv"
    )

    pd.DataFrame(
        oov.most_common(100),
        columns=["token", "count"],
    ).to_csv(
        oov_path,
        index=False,
    )

    print("\nProduction model saved:")
    print(args.model_output)
    print("\nMetadata saved:")
    print(args.metadata_output)
    print(
        "\nNext: python "
        "src\\smoke_test_production_model.py"
    )


if __name__ == "__main__":
    main()

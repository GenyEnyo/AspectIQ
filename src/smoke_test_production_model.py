from __future__ import annotations

from src.prediction import (
    load_prediction_assets,
    predict_multiple_aspects,
)


def main():
    model, glove_model = load_prediction_assets()

    cases = [
        (
            "The screen is beautiful but "
            "the battery life is disappointing.",
            ["screen", "battery life"],
        ),
        (
            "The food was excellent but "
            "the service was painfully slow.",
            ["food", "service"],
        ),
    ]

    expected = {
        "screen": "positive",
        "battery life": "negative",
        "food": "positive",
        "service": "negative",
    }

    failures = []

    print("\nPRODUCTION MODEL SMOKE TEST")

    for text, aspects in cases:
        results = predict_multiple_aspects(
            text=text,
            aspects=aspects,
            model=model,
            glove_model=glove_model,
        )

        print(f"\nText: {text}")

        for result in results:
            aspect = result["aspect"]
            predicted = result["sentiment"]
            expected_label = expected[aspect]
            status = (
                "PASS"
                if predicted == expected_label
                else "FAIL"
            )

            print(
                f"  {aspect:<13} -> {predicted:<8} "
                f"expected={expected_label:<8} [{status}]"
            )
            print(
                f"      target clause: "
                f"{result['target_clause']}"
            )
            print(
                f"      local context: "
                f"{result['local_context']}"
            )
            print(
                f"      dimensions: "
                f"{result['feature_dimensions']}"
            )

            if status == "FAIL":
                failures.append(aspect)

    if failures:
        raise SystemExit(
            "\nSmoke test failed for: "
            + ", ".join(failures)
        )

    print("\nAll production smoke tests passed.")


if __name__ == "__main__":
    main()

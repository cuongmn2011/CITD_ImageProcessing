import csv

from lpr.cli import main
from lpr.metrics import edit_distance, evaluate_ocr_pairs


def test_edit_distance_and_ocr_metrics() -> None:
    assert edit_distance("ABC", "ADC") == 1
    metrics = evaluate_ocr_pairs([
        {"ground_truth": "29A-123.45", "prediction": "29A12345"},
        {"ground_truth": "51F12345", "prediction": "51F1234"},
    ])
    assert metrics["samples"] == 2
    assert metrics["exact_accuracy"] == 0.5
    assert metrics["cer"] == 1 / 16


def test_evaluate_ocr_cli(tmp_path, capsys) -> None:
    path = tmp_path / "ocr.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["ground_truth", "prediction"])
        writer.writeheader()
        writer.writerow({"ground_truth": "29A12345", "prediction": "29A12345"})
    assert main(["evaluate-ocr", "--csv", str(path)]) == 0
    assert '"exact_accuracy": 1.0' in capsys.readouterr().out

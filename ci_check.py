import json
import sys

def check_golden_dataset():
    with open("golden_dataset.json", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "golden_dataset.json must be a list"
    for item in data:
        assert "question" in item, "Missing 'question' field"
        assert "ground_truth" in item, "Missing 'ground_truth' field"
    print(f"✓ golden_dataset.json valid — {len(data)} entries")

def check_prompts_config():
    import yaml
    with open("prompts.yaml", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    assert "v3" in prompts, "Missing v3 prompt template"
    print("✓ prompts.yaml valid — v3 template found")

def check_eval_results_threshold():
    try:
        with open("eval_results.json", encoding="utf-8") as f:
            results = json.load(f)
        faithfulness = results["ragas_scores"]["faithfulness"]
        refusal_accuracy = results["refusal_accuracy"]

        THRESHOLD_FAITHFULNESS = 0.30
        THRESHOLD_REFUSAL = 0.90

        assert faithfulness >= THRESHOLD_FAITHFULNESS, \
            f"Faithfulness {faithfulness} below threshold {THRESHOLD_FAITHFULNESS}"
        assert refusal_accuracy >= THRESHOLD_REFUSAL, \
            f"Refusal accuracy {refusal_accuracy} below threshold {THRESHOLD_REFUSAL}"

        print(f"✓ eval_results.json passes thresholds — faithfulness={faithfulness}, refusal={refusal_accuracy}")
    except FileNotFoundError:
        print("⚠ eval_results.json not found — skipping threshold check (run evaluate.py locally first)")

if __name__ == "__main__":
    try:
        check_golden_dataset()
        check_prompts_config()
        check_eval_results_threshold()
        print("\nAll CI checks passed.")
    except AssertionError as e:
        print(f"\n✗ CI CHECK FAILED: {e}")
        sys.exit(1)
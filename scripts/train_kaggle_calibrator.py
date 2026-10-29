import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import List, Tuple
import joblib
import requests
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

HC3_DIRECT_URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"
LOCAL_HC3_CACHE = Path("data/hc3_all.jsonl")
BENCHMARK_CSV = Path("data/AI_vs_Human_Text_Classification_Benchmark.csv")


def clean_hc3_artifacts(text: str) -> str:
    """Removes Reddit/tokenizer spacing artifacts so the classifier learns real human syntax."""
    t = re.sub(r"\s+([,.;:!?])", r"\1", text)
    t = re.sub(r"\b(do|does|did|is|are|was|were|have|has|had|can|could|will|would|should)\s+n't\b", r"\1n't", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+'(s|re|ve|ll|d|m)\b", r"'\1", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def load_or_download_hc3() -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []

    if not LOCAL_HC3_CACHE.exists():
        print(f"Auto-downloading HC3 Human-vs-AI Corpus (~73 MB) to {LOCAL_HC3_CACHE}...")
        LOCAL_HC3_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(HC3_DIRECT_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(LOCAL_HC3_CACHE, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        print("Download complete!")

    print(f"Parsing Human & AI essays from {LOCAL_HC3_CACHE}...")
    with open(LOCAL_HC3_CACHE, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            for h_ans in obj.get("human_answers", [])[:1]:
                cleaned_h = clean_hc3_artifacts(h_ans or "")
                if len(cleaned_h.split()) >= 20:
                    texts.append(cleaned_h)
                    labels.append(0)  # 0 = Human

            for ai_ans in obj.get("chatgpt_answers", [])[:1]:
                cleaned_ai = clean_hc3_artifacts(ai_ans or "")
                if len(cleaned_ai.split()) >= 20:
                    texts.append(cleaned_ai)
                    labels.append(1)  # 1 = AI

    return texts, labels


def load_custom_csv(csv_path: Path, text_col: str, label_col: str) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Support custom text_col or fallback to Text / essay columns
            t = (row.get(text_col) or row.get("Text") or row.get("essay") or "").strip()
            raw_lbl = str(
                row.get(label_col)
                or row.get("Label")
                or row.get("generated")
                or ""
            ).strip().lower()

            if not t or len(t.split()) < 15:
                continue

            if raw_lbl in ("1", "ai", "generated", "true", "ai-generated"):
                lbl = 1
            elif raw_lbl in ("0", "human", "false", "human-written"):
                lbl = 0
            elif "Detection_Score_Benchmark_Model_A" in row:
                try:
                    score = float(row["Detection_Score_Benchmark_Model_A"])
                    lbl = 1 if score >= 60.0 else 0
                except ValueError:
                    continue
            else:
                continue

            texts.append(t)
            labels.append(lbl)
    return texts, labels


def train_calibrator(csv_arg: str = "", text_col: str = "text", label_col: str = "label"):
    texts: List[str] = []
    labels: List[int] = []

    if csv_arg and Path(csv_arg).exists():
        print(f"Loading local CSV corpus from {csv_arg}...")
        t_csv, l_csv = load_custom_csv(Path(csv_arg), text_col, label_col)
        texts.extend(t_csv)
        labels.extend(l_csv)
    else:
        if csv_arg:
            print(f"Note: '{csv_arg}' not found locally — switching to automatic HC3 dataset download!")
        try:
            t_hc3, l_hc3 = load_or_download_hc3()
            texts.extend(t_hc3)
            labels.extend(l_hc3)
        except Exception as exc:
            print(f"HC3 download skipped ({exc}). Checking local benchmark CSV...")

    # Also merge local AI_vs_Human_Text_Classification_Benchmark.csv if it has a text column
    if BENCHMARK_CSV.exists():
        t_b, l_b = load_custom_csv(BENCHMARK_CSV, text_col, label_col)
        if t_b:
            texts.extend(t_b)
            labels.extend(l_b)

    ai_total = sum(labels)
    hu_total = len(labels) - ai_total
    print(f"Total Training Samples: {len(texts):,} ({ai_total:,} AI | {hu_total:,} Human)")

    if len(texts) < 40 or ai_total < 10 or hu_total < 10:
        raise RuntimeError("Insufficient samples loaded. Check your internet connection so HC3 can download.")

    # 1. Compute Monroe's Log-Odds Ratio with Dirichlet Prior on 2-to-4 word phrases
    print("Mining empirical AI 2-to-4 word N-Grams (Monroe Log-Odds)...")
    vec = CountVectorizer(ngram_range=(2, 4), min_df=8, max_features=40000, token_pattern=r"\b[a-z']+\b")
    X_counts = vec.fit_transform(texts)
    vocab = vec.get_feature_names_out()

    ai_mask = [i for i, y in enumerate(labels) if y == 1]
    hu_mask = [i for i, y in enumerate(labels) if y == 0]

    ai_sums = X_counts[ai_mask].sum(axis=0).A1
    hu_sums = X_counts[hu_mask].sum(axis=0).A1
    bg_sums = ai_sums + hu_sums

    n_ai = float(ai_sums.sum())
    n_hu = float(hu_sums.sum())
    n_bg = float(bg_sums.sum())

    log_odds_dict = {}
    for idx, ngram in enumerate(vocab):
        y_ai = ai_sums[idx]
        y_hu = hu_sums[idx]
        alpha_w = max(1.0, bg_sums[idx] * 0.05)
        alpha_0 = max(1.0, n_bg * 0.05)

        delta = math.log((y_ai + alpha_w) / (n_ai + alpha_0 - y_ai - alpha_w)) - math.log(
            (y_hu + alpha_w) / (n_hu + alpha_0 - y_hu - alpha_w)
        )
        var = (1.0 / (y_ai + alpha_w)) + (1.0 / (y_hu + alpha_w))
        z_score = delta / math.sqrt(var)

        if z_score >= 4.2:
            log_odds_dict[ngram] = round(z_score, 2)

    top_ai_ngrams = dict(sorted(log_odds_dict.items(), key=lambda x: x[1], reverse=True)[:1500])
    out_json = Path("data/ai_ngram_log_odds.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(top_ai_ngrams, f, indent=2)
    print(f"Saved {len(top_ai_ngrams):,} empirical AI N-Grams -> {out_json}")

    # 2. Train Zero-LLM Word + Character TF-IDF Logistic Scorer
    print("Training local scikit-learn Human-vs-AI probability scorer...")
    clf_pipe = Pipeline([
        ("feats", FeatureUnion([
            ("word_tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=35000, sublinear_tf=True)),
            ("char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=25000, sublinear_tf=True)),
        ])),
        ("clf", SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=40, random_state=42))
    ])
    clf_pipe.fit(texts, labels)

    out_model = Path("data/local_detector_scorer.joblib")
    joblib.dump(clf_pipe, out_model)
    print(f"Saved trained detector scorer -> {out_model}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-Downloading Zero-LLM Human vs AI Calibrator")
    parser.add_argument("--csv", default="", help="Optional path to custom Kaggle CSV (auto-downloads HC3 if omitted)")
    parser.add_argument("--text-col", default="text", help="Text column name")
    parser.add_argument("--label-col", default="label", help="Label column name")
    args = parser.parse_args()
    train_calibrator(args.csv, args.text_col, args.label_col)
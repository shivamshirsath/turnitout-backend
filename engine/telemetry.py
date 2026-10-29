import json
import zlib
import re
from pathlib import Path
from typing import Dict, Optional

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

from engine.similarity import compute_tfidf_cosine, compute_shingle_jaccard
from engine.benchmark_calibrator import GLOBAL_CALIBRATOR

AI_TRIGGER_LEXICON = [
    "delve", "tapestry", "pivotal", "testament", "furthermore", "moreover",
    "additionally", "consequently", "nevertheless", "realm", "underscore",
    "multifaceted", "meticulous", "unwavering", "symphony", "beacon", "paradigm",
    "seamless", "robust framework", "in conclusion", "it is important to note",
    "it is worth noting", "holistic", "synergistic", "overarching",
    "increasingly important", "significant challenges", "shown promise",
    "practical deployment", "constrained by", "does not by itself"
]

PREDICTABLE_4GRAMS = [
    "in the context of", "due to the fact", "plays a key role", "plays an important role",
    "it can be seen", "it is evident that", "on the other hand", "as a result of",
    "for the purpose of", "a wide range of", "a significant number of", "has the potential to",
    "warrants further investigation", "implications of this are", "evidence supports this conclusion"
]

_LOG_ODDS_PATH = Path("data/ai_ngram_log_odds.json")
_SKLEARN_MODEL_PATH = Path("data/local_detector_scorer.joblib")

KAGGLE_LOG_ODDS_NGRAMS: Dict[str, float] = {}
if _LOG_ODDS_PATH.exists():
    try:
        with open(_LOG_ODDS_PATH, "r", encoding="utf-8") as f:
            KAGGLE_LOG_ODDS_NGRAMS = json.load(f)
    except Exception:
        KAGGLE_LOG_ODDS_NGRAMS = {}

LOCAL_SKLEARN_SCORER = None
if HAS_JOBLIB and _SKLEARN_MODEL_PATH.exists():
    try:
        LOCAL_SKLEARN_SCORER = joblib.load(_SKLEARN_MODEL_PATH)
    except Exception:
        LOCAL_SKLEARN_SCORER = None


def compute_zippy_metrics(text: str, reference_text: Optional[str] = None) -> Dict[str, float]:
    if not text or not text.strip():
        return {
            "ai_probability": 0,
            "burstiness": 0.0,
            "entropy_score": 0.0,
            "semantic_retention": 100.0,
            "word_count": 0,
            "tell_hits": 0
        }

    words = text.split()
    word_count = len(words)
    encoded = text.encode("utf-8")
    compressed = zlib.compress(encoded, level=9)

    alpha_tokens = re.findall(r'[a-z]+', text.lower())
    ttr = (len(set(alpha_tokens)) / max(len(alpha_tokens), 1)) * 100.0
    raw_ratio = len(compressed) / max(len(encoded), 1)
    entropy_score = round(min((raw_ratio * 65.0) + (ttr * 0.45), 99.0), 1)

    csv_feats = GLOBAL_CALIBRATOR.extract_csv_features(text)
    burstiness = round(csv_feats["Burstiness_Score"], 1)
    csv_human_likeness = GLOBAL_CALIBRATOR.score_human_likeness(text)

    lower_text = text.lower()
    tell_hits = sum(lower_text.count(trigger) for trigger in AI_TRIGGER_LEXICON)
    ngram_hits = sum(lower_text.count(ng) for ng in PREDICTABLE_4GRAMS)

    kaggle_odds_penalty = 0.0
    if KAGGLE_LOG_ODDS_NGRAMS:
        for ng, z_score in KAGGLE_LOG_ODDS_NGRAMS.items():
            if ng in lower_text:
                kaggle_odds_penalty += min(6.0, z_score * 0.45)
                ngram_hits += 1

    contractions_cnt = len(re.findall(r"\b\w+'(?:t|re|ve|ll|d|s|m)\b", text))
    natural_bonus = min(10.0, contractions_cnt * 2.5)

    csv_ai_deficit = max(0.0, (82.0 - csv_human_likeness) * 0.85)
    burst_penalty = max(0.0, (58.0 - burstiness) * 0.55)
    pattern_penalty = (tell_hits * 11.0) + (ngram_hits * 7.5) + min(25.0, kaggle_odds_penalty)

    overlap_penalty = 0.0
    semantic_retention = 100.0
    if reference_text and reference_text.strip():
        cos = compute_tfidf_cosine(reference_text, text)
        shingle_3 = compute_shingle_jaccard(reference_text, text, n=3)
        if shingle_3 > 42.0:
            overlap_penalty = (shingle_3 - 42.0) * 0.60
        semantic_retention = round(min(100.0, max(84.0, 70.0 + (cos * 0.40))), 1)

    heuristic_risk = csv_ai_deficit + burst_penalty + pattern_penalty + overlap_penalty - natural_bonus

    # Blend with trained scikit-learn Kaggle DAIGT/HC3 classifier if available
    if LOCAL_SKLEARN_SCORER is not None:
        try:
            sklearn_prob = float(LOCAL_SKLEARN_SCORER.predict_proba([text])[0][1]) * 100.0
            raw_risk = (heuristic_risk * 0.45) + (sklearn_prob * 0.55)
        except Exception:
            raw_risk = heuristic_risk
    else:
        raw_risk = heuristic_risk

    ai_prob = int(min(99.0, max(0.0, round(raw_risk))))

    return {
        "ai_probability": ai_prob,
        "burstiness": burstiness,
        "entropy_score": entropy_score,
        "semantic_retention": semantic_retention,
        "word_count": word_count,
        "tell_hits": tell_hits + ngram_hits
    }
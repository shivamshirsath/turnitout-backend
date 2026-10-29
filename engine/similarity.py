import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, Tuple

def _tokenize(text: str) -> list:
    return re.findall(r'\b[a-z0-9]+\b', text.lower())

def compute_shingle_jaccard(text_a: str, text_b: str, n: int = 3) -> float:
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if len(tokens_a) < n or len(tokens_b) < n:
        set_a, set_b = set(tokens_a), set(tokens_b)
    else:
        set_a = {" ".join(tokens_a[i:i+n]) for i in range(len(tokens_a) - n + 1)}
        set_b = {" ".join(tokens_b[i:i+n]) for i in range(len(tokens_b) - n + 1)}
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return round((intersection / max(union, 1)) * 100.0, 2)

def compute_tfidf_cosine(text_a: str, text_b: str) -> float:
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    tf_a = Counter(tokens_a)
    tf_b = Counter(tokens_b)
    vocab = set(tf_a.keys()).union(set(tf_b.keys()))
    vec_a, vec_b = [], []
    for w in vocab:
        df = (1 if w in tf_a else 0) + (1 if w in tf_b else 0)
        idf = math.log((1 + 2) / (1 + df)) + 1.0
        vec_a.append((tf_a[w] / len(tokens_a)) * idf)
        vec_b.append((tf_b[w] / len(tokens_b)) * idf)
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return round((dot / (norm_a * norm_b)) * 100.0, 2)

def compute_sequence_overlap(text_a: str, text_b: str) -> Tuple[float, str]:
    words_a = text_a.split()
    words_b = text_b.split()
    if not words_a or not words_b:
        return 0.0, ""
    matcher = SequenceMatcher(None, [w.lower() for w in words_a], [w.lower() for w in words_b])
    ratio = round(matcher.ratio() * 100.0, 2)
    highlighted_b = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        segment = " ".join(words_b[j1:j2])
        if tag == "equal" and (j2 - j1) >= 3:
            highlighted_b.append(
                f'<span style="background-color: rgba(239, 68, 68, 0.28); border-bottom: 2px solid #EF4444; padding: 1px 4px; border-radius: 4px;">{segment}</span>'
            )
        else:
            highlighted_b.append(segment)
    return ratio, " ".join(highlighted_b)

def run_pairwise_analysis(doc_a: str, doc_b: str) -> Dict:
    jaccard_3 = compute_shingle_jaccard(doc_a, doc_b, n=3)
    jaccard_4 = compute_shingle_jaccard(doc_a, doc_b, n=4)
    cosine = compute_tfidf_cosine(doc_a, doc_b)
    lcs_ratio, html_diff = compute_sequence_overlap(doc_a, doc_b)
    composite = round((jaccard_3 * 0.35) + (cosine * 0.30) + (lcs_ratio * 0.35), 1)
    return {
        "composite_similarity": composite,
        "jaccard_3gram": jaccard_3,
        "jaccard_4gram": jaccard_4,
        "tfidf_cosine": cosine,
        "lcs_overlap": lcs_ratio,
        "highlighted_html": html_diff
    }

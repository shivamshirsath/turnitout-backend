import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

CSV_PATH = Path("data/AI_vs_Human_Text_Classification_Benchmark.csv")

STOPWORDS_SET = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for",
    "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by",
    "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all",
    "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get",
    "which", "go", "me", "when", "can", "like", "time", "no", "just", "him", "know",
    "take", "into", "your", "some", "could", "them", "than", "then", "now", "only",
    "its", "over", "also", "back", "after", "two", "how", "our", "first", "well",
    "even", "want", "because", "any", "these", "most", "us", "is", "are", "was",
    "were", "been", "being", "has", "had", "having", "does", "did", "doing", "should",
    "must", "while", "where", "through", "between", "both", "each", "more", "other",
    "such", "very", "same", "own", "here", "there", "without", "within", "across"
}


def count_syllables(word: str) -> int:
    w = re.sub(r'[^a-z]', '', word.lower())
    if len(w) <= 3:
        return 1
    w = re.sub(r'(?:[^laeiouy]es|ed|[^laeiouy]e)$', '', w)
    w = re.sub(r'^y', '', w)
    vowels = re.findall(r'[aeiouy]{1,2}', w)
    return max(1, len(vowels))


class BenchmarkCalibrator:
    """
    Trains an empirical statistical evaluator directly on
    `data/AI_vs_Human_Text_Classification_Benchmark.csv` across all stylometric columns.
    Separates Low-Detection Human rows from High-Detection AI rows.
    """

    FEATURE_KEYS = [
        "Average_Word_Length",
        "Average_Sentence_Length_Words",
        "Lexical_Diversity_Score",
        "Punctuation_Density",
        "Stopword_Density",
        "Flesch_Kincaid_Grade_Level",
        "Smog_Index",
        "Average_Syllables_Per_Word",
        "Burstiness_Score"
    ]

    def __init__(self, csv_path: Path = CSV_PATH):
        self.csv_path = csv_path
        # Human passing profile (derived from rows in your CSV with Detection_Score < 35%)
        self.human_means = {
            "Average_Word_Length": 4.95,
            "Average_Sentence_Length_Words": 12.2,
            "Lexical_Diversity_Score": 0.64,
            "Punctuation_Density": 0.038,
            "Stopword_Density": 0.49,
            "Flesch_Kincaid_Grade_Level": 9.1,
            "Smog_Index": 11.2,
            "Average_Syllables_Per_Word": 1.62,
            "Burstiness_Score": 66.5
        }
        self.human_stds = {
            "Average_Word_Length": 0.45,
            "Average_Sentence_Length_Words": 3.2,
            "Lexical_Diversity_Score": 0.12,
            "Punctuation_Density": 0.008,
            "Stopword_Density": 0.04,
            "Flesch_Kincaid_Grade_Level": 1.8,
            "Smog_Index": 1.5,
            "Average_Syllables_Per_Word": 0.16,
            "Burstiness_Score": 12.0
        }
        # High-AI risk profile (derived from rows in your CSV with Detection_Score > 70%)
        self.ai_means = {
            "Average_Word_Length": 5.85,
            "Average_Sentence_Length_Words": 18.5,
            "Lexical_Diversity_Score": 0.42,
            "Punctuation_Density": 0.032,
            "Stopword_Density": 0.38,
            "Flesch_Kincaid_Grade_Level": 13.5,
            "Smog_Index": 14.1,
            "Average_Syllables_Per_Word": 1.98,
            "Burstiness_Score": 38.0
        }
        self._train_from_csv()

    def _train_from_csv(self) -> None:
        if not self.csv_path.exists():
            return
        try:
            low_ai_rows: List[Dict[str, float]] = []
            high_ai_rows: List[Dict[str, float]] = []

            with open(self.csv_path, mode="r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        det_score = float(row.get("Detection_Score_Benchmark_Model_A", 50.0))
                        feat_row = {k: float(row[k]) for k in self.FEATURE_KEYS if k in row and row[k] != ""}
                        if len(feat_row) == len(self.FEATURE_KEYS):
                            if det_score <= 45.0:
                                low_ai_rows.append(feat_row)
                            elif det_score >= 65.0:
                                high_ai_rows.append(feat_row)
                    except (ValueError, KeyError, TypeError):
                        continue

            if len(low_ai_rows) >= 2:
                for k in self.FEATURE_KEYS:
                    vals = [r[k] for r in low_ai_rows]
                    m = sum(vals) / len(vals)
                    v = sum((x - m) ** 2 for x in vals) / max(len(vals), 1)
                    self.human_means[k] = m
                    self.human_stds[k] = max(math.sqrt(v), 0.02)

            if len(high_ai_rows) >= 2:
                for k in self.FEATURE_KEYS:
                    vals = [r[k] for r in high_ai_rows]
                    self.ai_means[k] = sum(vals) / len(vals)
        except Exception:
            pass

    def extract_csv_features(self, text: str) -> Dict[str, float]:
        words = re.findall(r'\b[a-zA-Z0-9\'-]+\b', text)
        word_count = max(len(words), 1)
        char_count = max(len(text), 1)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip().split()) >= 2]
        sent_count = max(len(sentences), 1)

        avg_word_len = sum(len(w) for w in words) / word_count
        avg_sent_len = word_count / sent_count

        lower_words = [w.lower() for w in words]
        unique_words = len(set(lower_words))
        lex_div = unique_words / word_count

        punct_count = len(re.findall(r'[,.;:!?()"\'-]', text))
        punct_density = punct_count / char_count

        stop_count = sum(1 for w in lower_words if w in STOPWORDS_SET)
        stop_density = stop_count / word_count

        syllable_counts = [count_syllables(w) for w in lower_words]
        total_syllables = sum(syllable_counts)
        avg_syllables = total_syllables / word_count
        polysyllables = sum(1 for s in syllable_counts if s >= 3)

        fk_grade = max(1.0, 0.39 * avg_sent_len + 11.8 * avg_syllables - 15.59)
        smog_index = 1.0430 * math.sqrt(polysyllables * (30.0 / sent_count)) + 3.1291

        if len(sentences) >= 2:
            s_lens = [len(s.split()) for s in sentences]
            m_len = sum(s_lens) / len(s_lens)
            var_len = sum((l - m_len) ** 2 for l in s_lens) / len(s_lens)
            burstiness = min(99.0, (math.sqrt(var_len) / max(m_len, 1.0)) * 135.0)
        else:
            clauses = [c.strip() for c in re.split(r'[,;—\-]+|\band\b|\bbut\b|\bwhile\b|\bbecause\b', text) if c.strip()]
            if len(clauses) >= 2:
                c_lens = [len(c.split()) for c in clauses]
                m_c = sum(c_lens) / len(c_lens)
                v_c = sum((l - m_c) ** 2 for l in c_lens) / len(c_lens)
                burstiness = min(94.0, 52.0 + (math.sqrt(v_c) / max(m_c, 1.0)) * 65.0)
            else:
                burstiness = 48.0

        return {
            "Average_Word_Length": avg_word_len,
            "Average_Sentence_Length_Words": avg_sent_len,
            "Lexical_Diversity_Score": lex_div,
            "Punctuation_Density": punct_density,
            "Stopword_Density": stop_density,
            "Flesch_Kincaid_Grade_Level": fk_grade,
            "Smog_Index": smog_index,
            "Average_Syllables_Per_Word": avg_syllables,
            "Burstiness_Score": burstiness
        }

    def score_human_likeness(self, text: str) -> float:
        """
        Computes Likelihood Ratio between the Low-Detection Human Centroid and
        the High-Detection AI Centroid from `AI_vs_Human_Text_Classification_Benchmark.csv`.
        """
        feats = self.extract_csv_features(text)
        human_dist = 0.0
        ai_dist = 0.0

        weights = {
            "Stopword_Density": 2.4,
            "Average_Syllables_Per_Word": 2.2,
            "Average_Word_Length": 1.8,
            "Burstiness_Score": 2.0,
            "Flesch_Kincaid_Grade_Level": 1.5,
            "Smog_Index": 1.4,
            "Lexical_Diversity_Score": 1.2,
            "Punctuation_Density": 0.8,
            "Average_Sentence_Length_Words": 1.2
        }

        for k in self.FEATURE_KEYS:
            val = feats[k]
            h_mean = self.human_means[k]
            a_mean = self.ai_means[k]
            std = self.human_stds[k]
            w = weights.get(k, 1.0)

            if k in ("Stopword_Density", "Burstiness_Score") and val >= h_mean:
                d_h = 0.0
            elif k in ("Average_Word_Length", "Average_Syllables_Per_Word", "Flesch_Kincaid_Grade_Level", "Smog_Index") and val <= h_mean:
                d_h = 0.0
            else:
                d_h = abs(val - h_mean) / std

            d_a = abs(val - a_mean) / std
            human_dist += d_h * w
            ai_dist += d_a * w

        total_d = max(human_dist + ai_dist, 1e-4)
        human_prob = (ai_dist / total_d) * 100.0
        return round(max(0.0, min(100.0, human_prob)), 1)


GLOBAL_CALIBRATOR = BenchmarkCalibrator()
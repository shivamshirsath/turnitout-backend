import argparse
import gzip
import json
import sqlite3
from pathlib import Path
import nltk
from nltk.corpus import wordnet as nltk_wn
from wordfreq import zipf_frequency

DB_PATH = Path("data/lexical_kb.sqlite")
RULES_PATH = Path("data/rules_catalog.json")


def init_db(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS lexical_synonyms (
            source_lemma TEXT NOT NULL,
            pos TEXT NOT NULL,
            sense_id TEXT NOT NULL,
            gloss TEXT,
            target_synonym TEXT NOT NULL,
            target_zipf REAL NOT NULL,
            is_phrasal INTEGER DEFAULT 0,
            register_tag TEXT DEFAULT 'general',
            sense_rank INTEGER DEFAULT 1
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lemma_pos ON lexical_synonyms(source_lemma, pos);")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS phrasal_paraphrases (
            source_phrase TEXT NOT NULL,
            lhs_tag TEXT NOT NULL,
            target_phrase TEXT NOT NULL,
            equiv_score REAL NOT NULL,
            target_zipf REAL NOT NULL
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_src_phrase ON phrasal_paraphrases(source_phrase);")
    conn.commit()


def build_wordnet_and_builtin_phrasal_kb(conn: sqlite3.Connection):
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    cur = conn.cursor()
    cur.execute("DELETE FROM lexical_synonyms WHERE register_tag LIKE 'oewn%';")
    cur.execute("DELETE FROM phrasal_paraphrases WHERE lhs_tag IN ('WN_PHRASAL', 'CATALOG_BREAKER');")

    lex_rows = []
    phrasal_rows = []
    lex_count = 0
    phr_count = 0

    # 1. Seed verified N-Gram predictability breakers from rules_catalog.json
    if RULES_PATH.exists():
        with open(RULES_PATH, "r", encoding="utf-8") as f:
            cat = json.load(f)
        breakers = cat.get("ngram_Predictability_breakers", {})
        unsafe_keys = {
            "the interplay between", "in illustrating that", "illustrating that",
            "allows for", "allow for", "leads to", "lead to", "contributes to",
            "contribute to", "in terms of", "with regard to", "in the context of"
        }
        for src_p, tgt_list in breakers.items():
            s_clean = src_p.strip().lower()
            if s_clean in unsafe_keys or len(s_clean.split()) < 2:
                continue
            for tgt_p in tgt_list:
                t_clean = tgt_p.strip().lower()
                z = sum(zipf_frequency(w, "en") for w in t_clean.split()) / max(len(t_clean.split()), 1)
                phrasal_rows.append((s_clean, "CATALOG_BREAKER", t_clean, 0.92, round(z, 2)))
                phr_count += 1

    print("Indexing WordNet single-word & multi-word phrasal paraphrases...")

    # 2. Index both Single-Word Synonyms AND Multi-Word Phrasal Paraphrases from WordNet
    for synset in nltk_wn.all_synsets():
        pos_map = {"n": "NOUN", "v": "VERB", "a": "ADJ", "s": "ADJ", "r": "ADV"}
        pos = pos_map.get(synset.pos(), "NOUN")
        sense_id = synset.name()
        gloss = synset.definition() or ""

        lemmas = synset.lemmas()
        if len(lemmas) < 2:
            continue

        lemma_pool = []
        for l in lemmas:
            raw_name = l.name().replace("_", " ").strip().lower()
            words_in_name = raw_name.split()
            if not (1 <= len(words_in_name) <= 4) or not all(w.isalpha() for w in words_in_name):
                continue
            z = (
                zipf_frequency(raw_name, "en")
                if len(words_in_name) == 1
                else sum(zipf_frequency(w, "en") for w in words_in_name) / len(words_in_name)
            )
            is_phrasal = 1 if len(words_in_name) > 1 else 0
            if (is_phrasal == 0 and z < 3.1) or (is_phrasal == 1 and z < 3.8):
                continue
            lemma_pool.append((raw_name, z, is_phrasal, l.count()))

        if len(lemma_pool) < 2:
            continue

        for src_name, src_z, src_phr, _ in lemma_pool:
            for tgt_name, tgt_z, tgt_phr, tgt_cnt in lemma_pool:
                if src_name == tgt_name:
                    continue

                if src_phr == 0:
                    lex_rows.append((
                        src_name, pos, sense_id, gloss, tgt_name,
                        round(tgt_z, 2), tgt_phr, "oewn_verified", tgt_cnt
                    ))
                    lex_count += 1
                elif len(src_name.split()) >= 2 and pos in ("VERB", "ADV", "ADJ") and tgt_z >= 4.5 and tgt_z >= src_z:
                    # Multi-word phrasal verb/adverbial -> natural higher-Zipf equivalent!
                    equiv = min(0.85, 0.42 + (0.05 * min(tgt_cnt, 8)))
                    phrasal_rows.append((
                        src_name, "WN_PHRASAL", tgt_name, round(equiv, 3), round(tgt_z, 2)
                    ))
                    phr_count += 1

                if len(lex_rows) >= 10000:
                    cur.executemany("""
                        INSERT INTO lexical_synonyms
                        (source_lemma, pos, sense_id, gloss, target_synonym, target_zipf, is_phrasal, register_tag, sense_rank)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, lex_rows)
                    conn.commit()
                    lex_rows.clear()

                if len(phrasal_rows) >= 5000:
                    cur.executemany("INSERT INTO phrasal_paraphrases VALUES (?, ?, ?, ?, ?);", phrasal_rows)
                    conn.commit()
                    phrasal_rows.clear()

    if lex_rows:
        cur.executemany("""
            INSERT INTO lexical_synonyms
            (source_lemma, pos, sense_id, gloss, target_synonym, target_zipf, is_phrasal, register_tag, sense_rank)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, lex_rows)
    if phrasal_rows:
        cur.executemany("INSERT INTO phrasal_paraphrases VALUES (?, ?, ?, ?, ?);", phrasal_rows)
    conn.commit()

    print(f"Indexed {lex_count:,} single-word synonyms + {phr_count:,} multi-word phrasal paraphrases into {DB_PATH}!")


def ingest_ppdb_phrasal_pack(conn: sqlite3.Connection, ppdb_path: str):
    path = Path(ppdb_path)
    if not path.exists():
        print(f"Optional PPDB file '{ppdb_path}' not present — using built-in WordNet + Catalog phrasal table.")
        return

    print(f"Streaming external PPDB 2.0 pack from {ppdb_path}...")
    cur = conn.cursor()
    opener = gzip.open if path.suffix == ".gz" else open
    batch = []
    added = 0

    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split(" ||| ")
            if len(parts) < 4:
                continue
            lhs = parts[0].strip("[]")
            src = parts[1].strip().lower()
            tgt = parts[2].strip().lower()
            feats = parts[3]

            if src == tgt or not (2 <= len(src.split()) <= 5) or not (1 <= len(tgt.split()) <= 5):
                continue
            if not all(w.isalpha() for w in src.split() + tgt.split()):
                continue

            equiv = 0.0
            for kv in feats.split():
                if kv.startswith("Equivalence="):
                    try:
                        equiv = float(kv.split("=")[1])
                    except ValueError:
                        pass
                    break

            if equiv < 0.35:
                continue

            src_z = sum(zipf_frequency(w, "en") for w in src.split()) / len(src.split())
            tgt_z = sum(zipf_frequency(w, "en") for w in tgt.split()) / len(tgt.split())
            if tgt_z < 4.3 or tgt_z < src_z - 0.15:
                continue

            batch.append((src, lhs, tgt, round(equiv, 3), round(tgt_z, 2)))
            added += 1
            if len(batch) >= 10000:
                cur.executemany("INSERT INTO phrasal_paraphrases VALUES (?, ?, ?, ?, ?);", batch)
                conn.commit()
                batch.clear()

    if batch:
        cur.executemany("INSERT INTO phrasal_paraphrases VALUES (?, ?, ?, ?, ?);", batch)
        conn.commit()
    print(f"Added {added:,} external PPDB 2.0 phrases into {DB_PATH}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SQLite Lexical + Phrasal Paraphrase Knowledge Base")
    parser.add_argument("--ppdb", type=str, default="", help="Optional path to external ppdb-2.0-s-phrasal.gz")
    args = parser.parse_args()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    init_db(conn)
    build_wordnet_and_builtin_phrasal_kb(conn)
    if args.ppdb:
        ingest_ppdb_phrasal_pack(conn, args.ppdb)
    conn.close()
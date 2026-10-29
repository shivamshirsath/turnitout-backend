import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

def generate_citations_and_packages(
    text: str,
    discipline: str = "Computer Science & AI",
    style: str = "IEEE",
    max_citations: int = 3
) -> Tuple[str, str, str, str]:
    with open(Path("data/citations_taxonomy.json"), "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    refs: List[Dict[str, str]] = taxonomy.get(discipline, list(taxonomy.values())[0])[:max_citations]
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    bib_entries = []
    latex_bibitems = []
    json_records = []

    for idx, ref in enumerate(refs):
        if sentences:
            target_idx = min((idx * 2) + 1, len(sentences) - 1) if len(sentences) > 1 else 0
            sent = sentences[target_idx]
            if style == "IEEE":
                marker = f" [{idx + 1}]"
            elif style == "APA":
                marker = f" ({ref['author']}, {ref['year']})"
            elif style == "MLA":
                surname = ref['author'].split()[0].rstrip(",")
                marker = f" ({surname} {ref.get('pages', '1').split('-')[0]})"
            else:
                marker = f" \\cite{{{ref['key']}}}"

            if marker not in sent and sent.endswith("."):
                sentences[target_idx] = sent[:-1] + marker + "."

        bib_entries.append(
            f"@article{{{ref['key']},\n"
            f"  author  = {{{ref['author']}}},\n"
            f"  title   = {{{ref['title']}}},\n"
            f"  journal = {{{ref['journal']}}},\n"
            f"  volume  = {{{ref.get('volume', '1')}}},\n"
            f"  pages   = {{{ref.get('pages', '1-10')}}},\n"
            f"  year    = {{{ref['year']}}},\n"
            f"  doi     = {{{ref.get('doi', '')}}}\n"
            f"}}"
        )
        latex_bibitems.append(
            f"\\bibitem{{{ref['key']}}} {ref['author']}, ``{ref['title']},'' \\emph{{{ref['journal']}}}, vol. {ref.get('volume', '1')}, pp. {ref.get('pages', '1-10')}, {ref['year']}."
        )
        json_records.append(ref)

    cited_text = " ".join(sentences)
    bibtex_str = "\n\n".join(bib_entries)
    latex_bib_str = "\\begin{thebibliography}{99}\n" + "\n".join(latex_bibitems) + "\n\\end{thebibliography}"
    json_str = json.dumps({"discipline": discipline, "style": style, "references": json_records}, indent=2)

    return cited_text, bibtex_str, latex_bib_str, json_str

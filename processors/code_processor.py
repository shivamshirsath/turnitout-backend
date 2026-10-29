import json
import re
from pathlib import Path
from typing import Dict, Tuple

with open(Path("data/rules_catalog.json"), "r", encoding="utf-8") as f:
    _CATALOG = json.load(f)
    CODE_PERSONAS = _CATALOG.get("code_personas", {})
    AI_TELLS = _CATALOG.get("ai_tells_38", {})

def _transform_comment_text(comment_text: str, persona: str) -> str:
    persona_key = "senior" if "senior" in persona.lower() or "architect" in persona.lower() else persona.lower()
    rules = CODE_PERSONAS.get(persona_key, CODE_PERSONAS.get("intermediate", {}))
    out = comment_text
    for tell, reps in AI_TELLS.items():
        out = re.sub(rf'\b{re.escape(tell)}\b', reps[0], out, flags=re.IGNORECASE)
    for src, target in sorted(rules.items(), key=lambda x: len(x[0]), reverse=True):
        def _sub_p(m):
            orig = m.group(0)
            return target.capitalize() if orig[0].isupper() else target
        out = re.sub(rf'\b{re.escape(src)}\b', _sub_p, out, flags=re.IGNORECASE)
    return out

def humanize_source_code(code: str, persona: str = "Intermediate", language: str = "auto") -> Tuple[str, Dict[str, int]]:
    n = len(code)
    i = 0
    out = []
    stats = {"line_comments_humanized": 0, "block_docstrings_humanized": 0, "strings_protected": 0}

    while i < n:
        if code[i:i+3] in ('"""', "'''"):
            quote_delim = code[i:i+3]
            end_idx = code.find(quote_delim, i + 3)
            if end_idx == -1:
                out.append(code[i:])
                break
            inner = code[i+3:end_idx]
            transformed_inner = _transform_comment_text(inner, persona)
            if transformed_inner != inner:
                stats["block_docstrings_humanized"] += 1
            out.append(quote_delim + transformed_inner + quote_delim)
            i = end_idx + 3
            continue

        if code[i:i+2] == "/*":
            end_idx = code.find("*/", i + 2)
            if end_idx == -1:
                out.append(code[i:])
                break
            inner = code[i+2:end_idx]
            transformed_inner = _transform_comment_text(inner, persona)
            if transformed_inner != inner:
                stats["block_docstrings_humanized"] += 1
            out.append("/*" + transformed_inner + "*/")
            i = end_idx + 2
            continue

        if code[i] in ('"', "'", "`"):
            quote_ch = code[i]
            start_str = i
            i += 1
            while i < n:
                if code[i] == "\\":
                    i += 2
                    continue
                if code[i] == quote_ch:
                    i += 1
                    break
                if quote_ch != "`" and code[i] == "\n":
                    break
                i += 1
            out.append(code[start_str:i])
            stats["strings_protected"] += 1
            continue

        is_slash = (code[i:i+2] == "//")
        is_sql = (code[i:i+2] == "--")
        is_hash = (code[i] == "#" and not code[i:i+8].startswith("#include") and not code[i:i+7].startswith("#define"))

        if is_slash or is_sql or is_hash:
            marker_len = 2 if (is_slash or is_sql) else 1
            marker = code[i:i+marker_len]
            nl_idx = code.find("\n", i + marker_len)
            if nl_idx == -1:
                nl_idx = n
            comment_body = code[i+marker_len:nl_idx]
            transformed_body = _transform_comment_text(comment_body, persona)
            if transformed_body != comment_body:
                stats["line_comments_humanized"] += 1
            out.append(marker + transformed_body)
            i = nl_idx
            continue

        out.append(code[i])
        i += 1

    return "".join(out), stats

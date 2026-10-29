import base64
import io
import os
import zipfile
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from engine.humanizer import TurnitoutEngine
from engine.similarity import run_pairwise_analysis
from engine.surprisal_scorer import GLOBAL_SURPRISAL_SCORER
from engine.telemetry import compute_zippy_metrics
from engine.voice_matcher import EmpiricalVoiceMatcher
from processors.citation_generator import generate_citations_and_packages
from processors.code_processor import humanize_source_code
from processors.doc_processor import process_docx_in_place
from processors.latex_processor import process_latex_manuscript

# Secret key shared ONLY between your Next.js backend and this FastAPI server
INTERNAL_API_SECRET = os.getenv("INTERNAL_API_SECRET", "turnitout-super-secret-key-2026")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")

GLOBAL_ENGINE: Optional[TurnitoutEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global GLOBAL_ENGINE
    print("⚡ Loading Turnitout v35.0 NLP Engine into RAM...")
    GLOBAL_ENGINE = TurnitoutEngine()
    print("✅ Turnitout v35.0 Engine Ready!")
    yield


app = FastAPI(
    title="Turnitout v35.0 NLP Engine API",
    version="35.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN] if FRONTEND_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def verify_internal_secret(x_internal_secret: Optional[str]):
    if x_internal_secret != INTERNAL_API_SECRET:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Direct API access is blocked. Use the official Turnitout web application."
        )


# ==========================================
# REQUEST & RESPONSE SCHEMAS
# ==========================================

class HumanizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    intensity: int = Field(95, ge=20, le=100)
    register_tone: str = Field("professional")
    use_zipf: bool = Field(True)
    clean_dashes: bool = Field(True)
    adaptive_mode: bool = Field(True)
    voice_sample: Optional[str] = Field("")
    seed: Optional[int] = Field(None)


class CodeHumanizeRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=100000)
    persona: str = Field("intermediate")
    language: str = Field("python")


class CitationRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=30000)
    discipline: str = Field("Computer Science & AI")
    style: str = Field("IEEE")
    max_citations: int = Field(3, ge=1, le=5)


class SimilarityRequest(BaseModel):
    doc_a: str = Field(..., min_length=5, max_length=50000)
    doc_b: str = Field(..., min_length=5, max_length=50000)


# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/health")
async def health_check():
    return {"status": "ok", "engine": "Turnitout v35.0 Ready"}


@app.post("/api/v1/humanize")
async def api_humanize_text(
    req: HumanizeRequest,
    x_internal_secret: Optional[str] = Header(None)
):
    verify_internal_secret(x_internal_secret)

    def _run_sync():
        voice_prof = EmpiricalVoiceMatcher.extract_profile(req.voice_sample or "")
        pre_metrics = compute_zippy_metrics(req.text)
        pre_highlight_rate, _ = GLOBAL_SURPRISAL_SCORER.paragraph_sentence_highlight_rate(req.text)

        humanized_out, telemetry = GLOBAL_ENGINE.humanize_text(
            text=req.text,
            intensity=req.intensity,
            register=req.register_tone,
            use_zipf_synsets=req.use_zipf,
            clean_em_dashes=req.clean_dashes,
            voice_profile=voice_prof,
            adaptive_mode=req.adaptive_mode,
            seed=req.seed,
        )

        # Thread-safe quality report generation
        from engine.pipeline_stages import GLOBAL_PIPELINE_VALIDATOR
        norm_reg = GLOBAL_ENGINE._normalize_register(req.register_tone)
        quality_report = GLOBAL_PIPELINE_VALIDATOR.generate_quality_report(
            req.text, humanized_out, register=norm_reg, telemetry=telemetry
        )

        post_metrics = compute_zippy_metrics(humanized_out, reference_text=req.text)
        post_highlight_rate, sent_scores = GLOBAL_SURPRISAL_SCORER.paragraph_sentence_highlight_rate(
            humanized_out, req.text
        )

        return {
            "original_text": req.text,
            "humanized_text": humanized_out,
            "telemetry": telemetry,
            "quality_report": quality_report,
            "pre_metrics": {
                **pre_metrics,
                "highlight_risk": pre_highlight_rate,
            },
            "post_metrics": {
                **post_metrics,
                "highlight_risk": post_highlight_rate,
                "sentence_scores": sent_scores,
            },
        }

    return await run_in_threadpool(_run_sync)


@app.post("/api/v1/process-file")
async def api_process_file(
    file: UploadFile = File(...),
    intensity: int = Form(90),
    register_tone: str = Form("academic"),
    use_zipf: bool = Form(True),
    clean_dashes: bool = Form(True),
    voice_sample: str = Form(""),
    x_internal_secret: Optional[str] = Header(None),
):
    verify_internal_secret(x_internal_secret)
    raw_bytes = await file.read()
    filename = file.filename or "document.txt"

    def _process_sync():
        voice_prof = EmpiricalVoiceMatcher.extract_profile(voice_sample)

        if filename.endswith(".docx"):
            out_bytes, tel = process_docx_in_place(
                raw_bytes, GLOBAL_ENGINE, intensity, register_tone, use_zipf, voice_prof
            )
            mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            out_name = f"humanized_{filename}"

        elif filename.endswith(".tex"):
            tex_str = raw_bytes.decode("utf-8", errors="ignore")
            out_tex, tel = process_latex_manuscript(
                tex_str, GLOBAL_ENGINE, intensity, register_tone, use_zipf, voice_prof
            )
            out_bytes = out_tex.encode("utf-8")
            mime = "text/plain"
            out_name = f"humanized_{filename}"

        elif filename.endswith(".zip"):
            master_zip = io.BytesIO()
            agg_tel: Dict[str, int] = {}
            with zipfile.ZipFile(master_zip, "w", zipfile.ZIP_DEFLATED) as zout:
                with zipfile.ZipFile(io.BytesIO(raw_bytes), "r") as zin:
                    for member in zin.namelist():
                        if member.endswith("/") or "__MACOSX" in member:
                            continue
                        m_bytes = zin.read(member)
                        base_name = os.path.basename(member)
                        if member.endswith(".docx"):
                            hb, t = process_docx_in_place(
                                m_bytes, GLOBAL_ENGINE, intensity, register_tone, use_zipf, voice_prof
                            )
                            zout.writestr(f"humanized_{base_name}", hb)
                        elif member.endswith(".tex"):
                            ht, t = process_latex_manuscript(
                                m_bytes.decode("utf-8", errors="ignore"),
                                GLOBAL_ENGINE, intensity, register_tone, use_zipf, voice_prof
                            )
                            zout.writestr(f"humanized_{base_name}", ht.encode("utf-8"))
                        elif member.endswith((".md", ".txt")):
                            htxt, t = GLOBAL_ENGINE.humanize_text(
                                m_bytes.decode("utf-8", errors="ignore"),
                                intensity, register_tone, use_zipf, clean_dashes, voice_prof
                            )
                            zout.writestr(f"humanized_{base_name}", htxt.encode("utf-8"))
                        else:
                            continue
                        for k, v in t.items():
                            agg_tel[k] = agg_tel.get(k, 0) + v
            out_bytes = master_zip.getvalue()
            tel = agg_tel
            mime = "application/zip"
            out_name = f"humanized_{filename}"

        else:
            txt_str = raw_bytes.decode("utf-8", errors="ignore")
            out_txt, tel = GLOBAL_ENGINE.humanize_text(
                txt_str, intensity, register_tone, use_zipf, clean_dashes, voice_prof
            )
            out_bytes = out_txt.encode("utf-8")
            mime = "text/plain"
            out_name = f"humanized_{filename}"

        return {
            "filename": out_name,
            "mime_type": mime,
            "file_base64": base64.b64encode(out_bytes).decode("ascii"),
            "telemetry": tel,
        }

    return await run_in_threadpool(_process_sync)


@app.post("/api/v1/code-comments")
async def api_humanize_code(
    req: CodeHumanizeRequest,
    x_internal_secret: Optional[str] = Header(None),
):
    verify_internal_secret(x_internal_secret)

    def _run_code():
        sanitized, stats = humanize_source_code(req.code, persona=req.persona, language=req.language)
        return {"humanized_code": sanitized, "stats": stats}

    return await run_in_threadpool(_run_code)


@app.post("/api/v1/citations")
async def api_citations(
    req: CitationRequest,
    x_internal_secret: Optional[str] = Header(None),
):
    verify_internal_secret(x_internal_secret)

    def _run_cite():
        cited_text, bibtex, latex_bib, json_meta = generate_citations_and_packages(
            req.text, discipline=req.discipline, style=req.style, max_citations=req.max_citations
        )
        return {
            "cited_text": cited_text,
            "bibtex": bibtex,
            "latex_bibitem": latex_bib,
            "json_metadata": json_meta,
        }

    return await run_in_threadpool(_run_cite)


@app.post("/api/v1/similarity")
async def api_similarity(
    req: SimilarityRequest,
    x_internal_secret: Optional[str] = Header(None),
):
    verify_internal_secret(x_internal_secret)
    return await run_in_threadpool(run_pairwise_analysis, req.doc_a, req.doc_b)